from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from sam3dworker import Sam3dWorkerClient


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXECUTABLE = Path("/opt/conda/bin/python")
INPUT_DIR = REPO_ROOT / "sam3dworker" / "tests" / "inputs" / "emp_default_tableoverview"
RUNS_DIR = REPO_ROOT / "sam3dworker" / "tests" / "runs"
PAYLOADS_PATH = INPUT_DIR / "payloads.json"
SAM3D_CONFIG_PATH = REPO_ROOT / "third_party" / "SAM3D-object" / "checkpoints" / "hf" / "pipeline.yaml"
HF_DIR = Path("/root/hf")


def _gpu_test_unavailable_reason() -> str | None:
    if not PYTHON_EXECUTABLE.exists():
        return f"missing python executable: {PYTHON_EXECUTABLE}"
    if not HF_DIR.exists():
        return f"missing sam3d checkpoint directory: {HF_DIR}"
    if not SAM3D_CONFIG_PATH.exists():
        return f"missing sam3d config file: {SAM3D_CONFIG_PATH}"
    if not PAYLOADS_PATH.exists():
        return f"missing sam3d payload file: {PAYLOADS_PATH}"

    probe = subprocess.run(
        [
            str(PYTHON_EXECUTABLE),
            "-c",
            (
                "import importlib, sys, torch; "
                "mods=('pytest', 'omegaconf', 'hydra', 'numpy', 'PIL'); "
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
        return "sam3d GPU test environment is not ready" + (f": {details}" if details else "")
    return None


def _load_payload_templates() -> list[dict[str, object]]:
    payloads = json.loads(PAYLOADS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payloads, list) or not payloads:
        raise ValueError(f"payload file must contain a non-empty list: {PAYLOADS_PATH}")
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            raise ValueError(f"payloads[{index}] must be an object")
        for field_name in ("image_path", "depth_path", "mask_path", "label", "fx", "fy", "cx", "cy"):
            if field_name not in payload:
                raise ValueError(f"payloads[{index}] missing required field: {field_name}")
        for path_field in ("image_path", "depth_path", "mask_path"):
            path = Path(str(payload[path_field]))
            if not path.exists():
                raise ValueError(f"payloads[{index}] path does not exist: {path}")
    return payloads


@pytest.fixture(scope="session")
def gpu_test_environment() -> dict[str, str]:
    reason = _gpu_test_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    return {"python_executable": str(PYTHON_EXECUTABLE)}


@pytest.fixture(scope="session")
def sam3d_run_root() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_root = RUNS_DIR / timestamp
    suffix = 0
    while run_root.exists():
        suffix += 1
        run_root = RUNS_DIR / f"{timestamp}-{suffix:02d}"
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


@pytest.fixture(scope="session")
def sam3d_test_payloads(sam3d_run_root: Path) -> list[dict[str, object]]:
    payloads = _load_payload_templates()
    materialized: list[dict[str, object]] = []
    for payload in payloads:
        label = str(payload["label"])
        item = dict(payload)
        item["output_dir"] = str(sam3d_run_root / "reconstruct" / label)
        materialized.append(item)
    return materialized


@pytest.fixture(scope="session")
def sam3d_client(
    gpu_test_environment: dict[str, str],
    sam3d_run_root: Path,
) -> Sam3dWorkerClient:
    worker_dir = sam3d_run_root / "worker-runtime"
    worker_dir.mkdir(parents=True, exist_ok=True)
    client = Sam3dWorkerClient(
        socket_path=worker_dir / "sam3d.sock",
        trace_path=worker_dir / "sam3d-trace.jsonl",
        python_executable=gpu_test_environment["python_executable"],
        startup_timeout=900.0,
        cwd=REPO_ROOT,
    )
    client.start()
    try:
        yield client
    finally:
        client.stop()


@pytest.fixture(scope="session")
def sam3d_reconstruct_responses(
    sam3d_client: Sam3dWorkerClient,
    sam3d_test_payloads: list[dict[str, object]],
) -> list[dict[str, object]]:
    responses: list[dict[str, object]] = []
    for payload in sam3d_test_payloads:
        response = sam3d_client.reconstruct(
            image_path=str(payload["image_path"]),
            depth_path=str(payload["depth_path"]),
            mask_path=str(payload["mask_path"]),
            output_dir=str(payload["output_dir"]),
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            label=str(payload["label"]),
            artifact_types=list(payload.get("artifact_types", [])),
            request_id=f"sam3d-{payload['label']}",
            timeout=1200.0,
        )
        responses.append(response)
    return responses
