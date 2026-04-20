from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_TURBO_PATH = Path("/etc/network_turbo")
sys.path.insert(0, str(REPO_ROOT / "worker_ipc"))

from worker_ipc import ManagedChildProcess, Response  # noqa: E402


class Sam3dWorkerCommandError(RuntimeError):
    pass


class Sam3dWorkerClient:
    def __init__(
        self,
        *,
        socket_path: str | Path,
        python_executable: str | Path | None = None,
        trace_path: str | Path | None = None,
        startup_timeout: float = 30.0,
        cwd: str | Path | None = None,
        network_turbo_path: str | Path | None = NETWORK_TURBO_PATH,
    ) -> None:
        worker_cwd = Path(cwd) if cwd is not None else REPO_ROOT
        executable = Path(python_executable) if python_executable is not None else Path(sys.executable)
        turbo_path = Path(network_turbo_path) if network_turbo_path is not None else None

        self._worker = ManagedChildProcess(
            command=_build_worker_command(executable, REPO_ROOT / "sam3dworker" / "worker.py", turbo_path),
            socket_path=Path(socket_path),
            trace_path=trace_path,
            worker_name="sam3d",
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

    def __enter__(self) -> "Sam3dWorkerClient":
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
            raise Sam3dWorkerCommandError(response.error or f"sam3d worker command failed: {command}")
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

    def reconstruct(
        self,
        *,
        image_path: str | Path,
        depth_path: str | Path,
        mask_path: str | Path,
        output_dir: str | Path,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        label: str,
        artifact_types: list[str] | None = None,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "image_path": str(Path(image_path)),
            "depth_path": str(Path(depth_path)),
            "mask_path": str(Path(mask_path)),
            "output_dir": str(Path(output_dir)),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "label": label,
        }
        if artifact_types is not None:
            payload["artifact_types"] = artifact_types
        return self.call("reconstruct", payload, request_id=request_id, timeout=timeout)


def _build_worker_command(
    executable: Path,
    worker_script: Path,
    network_turbo_path: Path | None,
) -> list[str]:
    exec_cmd = f"exec {shlex.quote(str(executable))} {shlex.quote(str(worker_script))}"
    if network_turbo_path is None:
        shell_cmd = exec_cmd
    else:
        shell_cmd = (
            f"if [ -f {shlex.quote(str(network_turbo_path))} ]; then "
            f"source {shlex.quote(str(network_turbo_path))}; "
            f"fi; {exec_cmd}"
        )
    return ["bash", "-lc", shell_cmd]
