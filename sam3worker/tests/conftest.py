from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from sam3worker import Sam3WorkerClient


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXECUTABLE = Path("/root/autodl-tmp/conda/envs/sam3d-objects/bin/python")
SAM3_WEIGHT_PATH = Path("/root/sam3.pt")
INPUT_DIR = REPO_ROOT / "sam3worker" / "tests" / "inputs"
RUNS_DIR = REPO_ROOT / "sam3worker" / "tests" / "runs"
IMAGE_PATH = INPUT_DIR / "example.png"
BBOXES_PATH = INPUT_DIR / "bboxes.json"


def _gpu_test_unavailable_reason() -> str | None:
    if not PYTHON_EXECUTABLE.exists():
        return f"missing python executable: {PYTHON_EXECUTABLE}"
    if not SAM3_WEIGHT_PATH.exists():
        return f"missing sam3 weight file: {SAM3_WEIGHT_PATH}"
    if not IMAGE_PATH.exists():
        return f"missing test image: {IMAGE_PATH}"
    if not BBOXES_PATH.exists():
        return f"missing test bbox file: {BBOXES_PATH}"

    probe = subprocess.run(
        [
            str(PYTHON_EXECUTABLE),
            "-c",
            (
                "import importlib, sys, torch; "
                "mods=('ultralytics', 'pytest'); "
                "[importlib.import_module(name) for name in mods]; "
                "sys.exit(0 if torch.cuda.is_available() else 1)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        details = (probe.stderr or probe.stdout).strip()
        return "sam3 GPU test environment is not ready" + (f": {details}" if details else "")
    return None


@pytest.fixture(scope="session")
def gpu_test_environment() -> dict[str, str]:
    reason = _gpu_test_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    return {"python_executable": str(PYTHON_EXECUTABLE)}


@pytest.fixture(scope="session")
def sam3_test_inputs() -> dict[str, object]:
    return {
        "image_path": IMAGE_PATH,
        "bboxes": json.loads(BBOXES_PATH.read_text(encoding="utf-8")),
    }


@pytest.fixture(scope="session")
def sam3_run_root() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_root = RUNS_DIR / timestamp
    suffix = 0
    while run_root.exists():
        suffix += 1
        run_root = RUNS_DIR / f"{timestamp}-{suffix:02d}"
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


@pytest.fixture(scope="session")
def sam3_client(
    gpu_test_environment: dict[str, str],
    sam3_run_root: Path,
) -> Sam3WorkerClient:
    worker_dir = sam3_run_root / "worker-runtime"
    worker_dir.mkdir(parents=True, exist_ok=True)
    client = Sam3WorkerClient(
        socket_path=worker_dir / "sam3.sock",
        trace_path=worker_dir / "sam3-trace.jsonl",
        python_executable=gpu_test_environment["python_executable"],
        startup_timeout=180.0,
        cwd=REPO_ROOT,
    )
    client.start()
    try:
        yield client
    finally:
        client.stop()


@pytest.fixture(scope="session")
def sam3_infer_response(
    sam3_client: Sam3WorkerClient,
    sam3_test_inputs: dict[str, object],
    sam3_run_root: Path,
) -> dict[str, object]:
    output_root = sam3_run_root / "fixture-infer"
    output_root.mkdir(parents=True, exist_ok=True)
    return sam3_client.infer(
        image_path=sam3_test_inputs["image_path"],
        output_dir=output_root / "req-1",
        bboxes=sam3_test_inputs["bboxes"],
        request_id="sam3-example-infer",
        timeout=300.0,
    )
