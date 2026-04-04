from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_SAM3D_DIR = REPO_ROOT / "third_party" / "SAM3D-object"


def describe_service() -> dict[str, Any]:
    return {
        "worker": "sam3d",
        "third_party_dir": str(THIRD_PARTY_SAM3D_DIR),
        "status": "skeleton",
    }


def handle_command(command: str, payload: dict[str, Any]) -> dict[str, Any]:
    if command == "ping":
        return {"status": "ready"}
    if command == "describe":
        return describe_service()
    raise ValueError(f"unknown command: {command}")
