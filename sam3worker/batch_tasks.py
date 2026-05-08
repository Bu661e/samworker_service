from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np
from PIL import Image


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam3worker.client import Sam3WorkerClient


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASKS_DIR = PACKAGE_ROOT / "测试"
DEFAULT_RUNS_DIR = PACKAGE_ROOT / "tests" / "runs"
DEFAULT_PYTHON_EXECUTABLE = Path("/opt/conda/bin/python")
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_STARTUP_TIMEOUT = 180.0


@dataclass(frozen=True)
class TaskInput:
    task_id: str
    task_dir: Path
    image_path: Path
    bboxes: list[dict[str, Any]]
    task_name: str | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 on every task under 测试/ and save combined mask PNGs."
    )
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON_EXECUTABLE)
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    return parser.parse_args()


def create_run_root(runs_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_root = runs_dir / timestamp
    suffix = 0
    while run_root.exists():
        suffix += 1
        run_root = runs_dir / f"{timestamp}-{suffix:02d}"
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def load_task_inputs(tasks_dir: Path) -> list[TaskInput]:
    if not tasks_dir.exists() or not tasks_dir.is_dir():
        raise ValueError(f"tasks_dir does not exist: {tasks_dir}")

    task_inputs: list[TaskInput] = []
    for task_dir in sorted((path for path in tasks_dir.iterdir() if path.is_dir()), key=_task_sort_key):
        image_path = task_dir / "rgb.png"
        bboxes_path = task_dir / "res_bboxes.json"
        task_name_path = task_dir / "任务名称.md"

        if not image_path.is_file():
            raise ValueError(f"missing task image: {image_path}")
        if not bboxes_path.is_file():
            raise ValueError(f"missing task bbox file: {bboxes_path}")

        raw_bboxes = json.loads(bboxes_path.read_text(encoding="utf-8"))
        if not isinstance(raw_bboxes, list) or not raw_bboxes:
            raise ValueError(f"task bbox file must contain a non-empty list: {bboxes_path}")

        task_name = None
        if task_name_path.is_file():
            task_name = task_name_path.read_text(encoding="utf-8").strip() or None

        task_inputs.append(
            TaskInput(
                task_id=task_dir.name,
                task_dir=task_dir,
                image_path=image_path.resolve(),
                bboxes=raw_bboxes,
                task_name=task_name,
            )
        )

    if not task_inputs:
        raise ValueError(f"no task directories found under: {tasks_dir}")
    return task_inputs


def save_combined_mask_png(
    mask_paths: Sequence[Path],
    output_path: Path,
    *,
    image_path: Path,
) -> None:
    width, height = _read_image_size(image_path)
    canvas = np.zeros((height, width), dtype=np.uint8)

    for mask_path in mask_paths:
        with Image.open(mask_path) as mask_image:
            mask_array = np.asarray(mask_image.convert("L"), dtype=np.uint8)

        if mask_array.shape != (height, width):
            raise ValueError(
                f"mask size {mask_array.shape} does not match image size {(height, width)}: {mask_path}"
            )

        canvas[mask_array > 0] = 255

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(output_path, format="PNG")


def run_all_tasks(
    *,
    tasks_dir: Path,
    runs_dir: Path,
    python_executable: Path,
    startup_timeout: float,
    timeout: float,
) -> Path:
    task_inputs = load_task_inputs(tasks_dir)
    run_root = create_run_root(runs_dir)
    worker_runtime_dir = run_root / "worker-runtime"
    worker_runtime_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    with Sam3WorkerClient(
        socket_path=worker_runtime_dir / "sam3.sock",
        trace_path=worker_runtime_dir / "sam3-trace.jsonl",
        python_executable=python_executable,
        startup_timeout=startup_timeout,
        cwd=PACKAGE_ROOT.parent,
    ) as client:
        for task in task_inputs:
            with TemporaryDirectory(prefix=f"sam3-task-{task.task_id}-", dir=str(run_root)) as temp_dir:
                masks_dir = Path(temp_dir)
                response = client.infer(
                    image_path=task.image_path,
                    output_dir=masks_dir,
                    bboxes=task.bboxes,
                    request_id=f"task-{task.task_id}",
                    timeout=timeout,
                )

                results = response.get("results", [])
                if not isinstance(results, list):
                    raise ValueError(f"sam3 response results is not a list for task {task.task_id}")

                mask_paths = [
                    Path(str(item["mask_path"]))
                    for item in results
                    if isinstance(item, dict) and item.get("found") and item.get("mask_path")
                ]
                combined_mask_path = task.task_dir / "combined_masks.png"
                save_combined_mask_png(mask_paths, combined_mask_path, image_path=task.image_path)

                task_run_dir = run_root / f"task-{task.task_id}"
                response_path = task_run_dir / "infer_response.json"
                response_path.parent.mkdir(parents=True, exist_ok=True)
                response_path.write_text(
                    json.dumps(response, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            summary.append(
                {
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "task_dir": str(task.task_dir.resolve()),
                    "image_path": str(task.image_path),
                    "bbox_count": len(task.bboxes),
                    "found_count": len(mask_paths),
                    "combined_mask_path": str(combined_mask_path.resolve()),
                    "infer_response_path": str(response_path.resolve()),
                }
            )

    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_root


def _read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _task_sort_key(task_dir: Path) -> tuple[int, str]:
    try:
        return (0, f"{int(task_dir.name):09d}")
    except ValueError:
        return (1, task_dir.name)


def main() -> int:
    args = _parse_args()
    run_root = run_all_tasks(
        tasks_dir=args.tasks_dir.resolve(),
        runs_dir=args.runs_dir.resolve(),
        python_executable=args.python_executable.resolve(),
        startup_timeout=args.startup_timeout,
        timeout=args.timeout,
    )
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
