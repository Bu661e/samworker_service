from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_SAM3_DIR = REPO_ROOT / "third_party" / "sam3-ultralytics"
SAM3_RUNNER = THIRD_PARTY_SAM3_DIR / "run_sam3_inference.py"


def describe_service() -> dict[str, Any]:
    return {
        "worker": "sam3",
        "third_party_dir": str(THIRD_PARTY_SAM3_DIR),
        "runner_script": str(SAM3_RUNNER),
        "status": "skeleton",
    }


def handle_command(command: str, payload: dict[str, Any]) -> dict[str, Any]:
    if command == "ping":
        return {"status": "ready"}
    if command == "describe":
        return describe_service()
    raise ValueError(f"unknown command: {command}")
