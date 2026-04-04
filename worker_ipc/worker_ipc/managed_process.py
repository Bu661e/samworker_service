from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .client import UdsJsonlClient
from .exceptions import WorkerStartError


class ManagedChildProcess:
    def __init__(
        self,
        *,
        command: list[str],
        socket_path: str | Path,
        trace_path: str | Path | None = None,
        worker_name: str | None = None,
        startup_timeout: float = 10.0,
        cwd: str | Path | None = None,
    ) -> None:
        self.command = command
        self.socket_path = Path(socket_path)
        self.trace_path = Path(trace_path) if trace_path else None
        self.worker_name = worker_name or "worker"
        self.startup_timeout = startup_timeout
        self.cwd = str(cwd) if cwd is not None else None
        self._process: subprocess.Popen[str] | None = None
        self._client: UdsJsonlClient | None = None

    def start(self) -> None:
        env = os.environ.copy()
        env["WORKER_IPC_SOCKET_PATH"] = str(self.socket_path)
        env["WORKER_IPC_WORKER_NAME"] = self.worker_name
        if self.trace_path is not None:
            env["WORKER_IPC_TRACE_PATH"] = str(self.trace_path)

        self._process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        deadline = time.time() + self.startup_timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise WorkerStartError("worker process exited before becoming ready")

            client = UdsJsonlClient(self.socket_path)
            try:
                client.connect()
                response = client.call("ping", {}, request_id="startup-ping", timeout=0.5)
                if not response.ok:
                    raise WorkerStartError("worker ping failed during startup")
                self._client = client
                return
            except Exception as exc:
                client.close()
                last_error = exc
                time.sleep(0.05)

        self.stop()
        raise WorkerStartError("worker did not become ready: %s" % last_error)

    def call(
        self,
        command: str,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        timeout: float | None = None,
    ):
        if self._client is None:
            raise RuntimeError("managed worker is not started")
        return self._client.call(command, payload, request_id=request_id, timeout=timeout)

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            self._process = None
