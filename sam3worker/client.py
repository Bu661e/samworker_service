from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "worker_ipc"))

from worker_ipc import ManagedChildProcess, Response  # noqa: E402


class Sam3WorkerCommandError(RuntimeError):
    pass


class Sam3WorkerClient:
    def __init__(
        self,
        *,
        socket_path: str | Path,
        python_executable: str | Path | None = None,
        trace_path: str | Path | None = None,
        startup_timeout: float = 30.0,
        cwd: str | Path | None = None,
    ) -> None:
        worker_cwd = Path(cwd) if cwd is not None else REPO_ROOT
        executable = Path(python_executable) if python_executable is not None else Path(sys.executable)

        self._worker = ManagedChildProcess(
            command=[str(executable), str(REPO_ROOT / "sam3worker" / "worker.py")],
            socket_path=Path(socket_path),
            trace_path=trace_path,
            worker_name="sam3",
            startup_timeout=startup_timeout,
            cwd=worker_cwd,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._worker.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._worker.stop()
        self._started = False

    def __enter__(self) -> "Sam3WorkerClient":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def call_raw(
        self,
        command: str,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> Response:
        return self._worker.call(command, payload, request_id=request_id, timeout=timeout)

    def call(
        self,
        command: str,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        response = self.call_raw(command, payload, request_id=request_id, timeout=timeout)
        if not response.ok:
            raise Sam3WorkerCommandError(response.error or f"sam3 worker command failed: {command}")
        return response.payload

    def ping(
        self,
        *,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.call("ping", {}, request_id=request_id, timeout=timeout)

    def describe(
        self,
        *,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.call("describe", {}, request_id=request_id, timeout=timeout)

    def infer(
        self,
        *,
        image_path: str | Path,
        output_dir: str | Path,
        bboxes: list[dict[str, Any]],
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "image_path": str(Path(image_path)),
            "output_dir": str(Path(output_dir)),
            "bboxes": bboxes,
        }
        return self.call("infer", payload, request_id=request_id, timeout=timeout)
