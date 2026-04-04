# Worker IPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable internal Python package that supports synchronous parent-child IPC over Unix Domain Sockets with JSON Lines framing, optional protocol trace files, and a managed child-process wrapper.

**Architecture:** Create a repository-root package named `worker_ipc` beside `samworker`. Keep transport and process-management logic inside the package and keep SAM-specific business handlers in worker-specific modules. Use TDD for the message models, JSONL codec, server loop, client, and managed child-process wrapper, with a demo worker and integration tests proving the package works end to end.

**Tech Stack:** Python 3, `dataclasses`, `socket`, `subprocess`, `pathlib`, `logging`, `pytest`

---

This workspace does not have a `.git` directory, so commit steps are intentionally omitted from the plan.

## File Structure

- Create: `worker_ipc/pyproject.toml`
- Create: `worker_ipc/README.md`
- Create: `worker_ipc/worker_ipc/__init__.py`
- Create: `worker_ipc/worker_ipc/exceptions.py`
- Create: `worker_ipc/worker_ipc/messages.py`
- Create: `worker_ipc/worker_ipc/jsonl.py`
- Create: `worker_ipc/worker_ipc/server.py`
- Create: `worker_ipc/worker_ipc/client.py`
- Create: `worker_ipc/worker_ipc/managed_process.py`
- Create: `worker_ipc/examples/echo_worker.py`
- Create: `worker_ipc/examples/echo_parent.py`
- Create: `worker_ipc/tests/test_messages.py`
- Create: `worker_ipc/tests/test_jsonl.py`
- Create: `worker_ipc/tests/test_server.py`
- Create: `worker_ipc/tests/test_client.py`
- Create: `worker_ipc/tests/test_managed_process.py`

### Task 1: Package Skeleton And Message Models

**Files:**
- Create: `worker_ipc/pyproject.toml`
- Create: `worker_ipc/worker_ipc/__init__.py`
- Create: `worker_ipc/worker_ipc/exceptions.py`
- Create: `worker_ipc/worker_ipc/messages.py`
- Test: `worker_ipc/tests/test_messages.py`

- [ ] **Step 1: Write the failing tests for request and response validation**

```python
from worker_ipc.messages import Request, Response


def test_request_from_dict_requires_request_id_command_and_payload():
    request = Request.from_dict(
        {
            "request_id": "req-1",
            "command": "ping",
            "payload": {"value": 1},
        }
    )
    assert request.request_id == "req-1"
    assert request.command == "ping"
    assert request.payload == {"value": 1}


def test_request_from_dict_rejects_non_object_payload():
    with pytest.raises(ValueError, match="payload must be a JSON object"):
        Request.from_dict({"request_id": "req-1", "command": "ping", "payload": []})


def test_response_success_and_error_to_dict():
    success = Response.success("req-1", {"status": "ready"})
    error = Response.error("req-1", "boom")

    assert success.to_dict() == {
        "request_id": "req-1",
        "ok": True,
        "payload": {"status": "ready"},
    }
    assert error.to_dict() == {
        "request_id": "req-1",
        "ok": False,
        "payload": {},
        "error": "boom",
    }
```

- [ ] **Step 2: Run the focused tests and verify they fail because the package does not exist yet**

Run:

```bash
python3 -m pytest worker_ipc/tests/test_messages.py -q
```

Expected:

```text
ERROR ... ModuleNotFoundError: No module named 'worker_ipc'
```

- [ ] **Step 3: Create the package metadata and minimal message-model implementation**

`worker_ipc/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "worker-ipc"
version = "0.1.0"
description = "Internal Unix domain socket IPC helpers for parent-child workers"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`worker_ipc/worker_ipc/exceptions.py`

```python
class WorkerIpcError(RuntimeError):
    """Base exception for the worker_ipc package."""


class ProtocolError(WorkerIpcError):
    """Raised when a protocol message is malformed."""


class ClientTimeoutError(WorkerIpcError):
    """Raised when the client times out waiting for a response."""


class WorkerStartError(WorkerIpcError):
    """Raised when a managed child process cannot be started cleanly."""
```

`worker_ipc/worker_ipc/messages.py`

```python
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class Request:
    request_id: str
    command: str
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, value: object) -> "Request":
        if not isinstance(value, dict):
            raise ValueError("message must be a JSON object")
        request_id = value.get("request_id")
        command = value.get("command")
        payload = value.get("payload")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(command, str) or not command:
            raise ValueError("command must be a non-empty string")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        return cls(request_id=request_id, command=command, payload=payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "command": self.command,
            "payload": self.payload,
        }


@dataclass(slots=True, frozen=True)
class Response:
    request_id: str | None
    ok: bool
    payload: dict[str, Any]
    error: str | None = None

    @classmethod
    def success(cls, request_id: str | None, payload: dict[str, Any]) -> "Response":
        return cls(request_id=request_id, ok=True, payload=payload)

    @classmethod
    def error(
        cls,
        request_id: str | None,
        error: str,
        payload: dict[str, Any] | None = None,
    ) -> "Response":
        return cls(request_id=request_id, ok=False, payload=payload or {}, error=error)

    def to_dict(self) -> dict[str, Any]:
        message = {"ok": self.ok, "payload": self.payload}
        if self.request_id is not None:
            message["request_id"] = self.request_id
        if self.error is not None:
            message["error"] = self.error
        return message
```

`worker_ipc/worker_ipc/__init__.py`

```python
from .messages import Request, Response

__all__ = ["Request", "Response"]
```

- [ ] **Step 4: Run the message-model tests and verify they pass**

Run:

```bash
python3 -m pytest worker_ipc/tests/test_messages.py -q
```

Expected:

```text
3 passed
```

### Task 2: JSON Lines Codec

**Files:**
- Create: `worker_ipc/worker_ipc/jsonl.py`
- Test: `worker_ipc/tests/test_jsonl.py`

- [ ] **Step 1: Write failing JSONL codec tests**

```python
import io
import pytest

from worker_ipc.exceptions import ProtocolError
from worker_ipc.jsonl import read_json_line, write_json_line


def test_write_json_line_appends_newline():
    stream = io.StringIO()
    write_json_line(stream, {"request_id": "req-1", "command": "ping", "payload": {}})
    assert stream.getvalue().endswith("\n")


def test_read_json_line_returns_dict():
    stream = io.StringIO('{"request_id":"req-1","command":"ping","payload":{}}\n')
    assert read_json_line(stream) == {"request_id": "req-1", "command": "ping", "payload": {}}


def test_read_json_line_rejects_non_object_json():
    stream = io.StringIO('["bad"]\n')
    with pytest.raises(ProtocolError, match="top-level JSON value must be an object"):
        read_json_line(stream)
```

- [ ] **Step 2: Run the focused tests and verify they fail because the codec module is missing**

Run:

```bash
python3 -m pytest worker_ipc/tests/test_jsonl.py -q
```

Expected:

```text
ERROR ... ModuleNotFoundError: No module named 'worker_ipc.jsonl'
```

- [ ] **Step 3: Implement JSONL read and write helpers**

`worker_ipc/worker_ipc/jsonl.py`

```python
import json
from typing import Any, TextIO

from .exceptions import ProtocolError


def write_json_line(stream: TextIO, message: dict[str, Any]) -> None:
    if not isinstance(message, dict):
        raise ProtocolError("top-level JSON value must be an object")
    stream.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


def read_json_line(stream: TextIO) -> dict[str, Any] | None:
    line = stream.readline()
    if line == "":
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid JSON line") from exc
    if not isinstance(value, dict):
        raise ProtocolError("top-level JSON value must be an object")
    return value
```

- [ ] **Step 4: Run the JSONL tests and verify they pass**

Run:

```bash
python3 -m pytest worker_ipc/tests/test_jsonl.py -q
```

Expected:

```text
3 passed
```

### Task 3: UDS Server With Trace Support

**Files:**
- Create: `worker_ipc/worker_ipc/server.py`
- Test: `worker_ipc/tests/test_server.py`

- [ ] **Step 1: Write failing server tests for request handling, trace output, and stale socket cleanup**

```python
import json
import socket
import threading
import time
from pathlib import Path

from worker_ipc.messages import Response
from worker_ipc.server import UdsJsonlServer


def _start_server(server, handler):
    stop_flag = {"value": False}

    def should_stop():
        return stop_flag["value"]

    thread = threading.Thread(
        target=server.serve_forever,
        args=(handler, should_stop),
        daemon=True,
    )
    thread.start()

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if server.socket_path.exists():
            return thread, stop_flag
        time.sleep(0.01)
    raise AssertionError("server socket did not appear")


def test_server_handles_ping_request(tmp_path):
    socket_path = tmp_path / "server.sock"
    server = UdsJsonlServer(socket_path=socket_path)

    def handler(request):
        return Response.success(request.request_id, {"status": "ready"})

    thread, stop_flag = _start_server(server, handler)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        writer = client.makefile("w", encoding="utf-8")
        reader = client.makefile("r", encoding="utf-8")
        writer.write('{"request_id":"req-1","command":"ping","payload":{}}\n')
        writer.flush()
        response = json.loads(reader.readline())
    assert response == {
        "request_id": "req-1",
        "ok": True,
        "payload": {"status": "ready"},
    }
    stop_flag["value"] = True
    thread.join(timeout=1)
    server.close()


def test_server_appends_trace_records_when_trace_path_is_set(tmp_path):
    socket_path = tmp_path / "trace.sock"
    trace_path = tmp_path / "trace.jsonl"
    server = UdsJsonlServer(socket_path=socket_path, trace_path=trace_path, worker_name="echo")

    def handler(request):
        return Response.success(request.request_id, {"echo": request.payload["value"]})

    thread, stop_flag = _start_server(server, handler)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        writer = client.makefile("w", encoding="utf-8")
        reader = client.makefile("r", encoding="utf-8")
        writer.write('{"request_id":"req-2","command":"echo","payload":{"value":3}}\n')
        writer.flush()
        _ = reader.readline()

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["request", "response"]
    assert records[0]["request_id"] == "req-2"
    assert records[1]["message"]["payload"] == {"echo": 3}
    stop_flag["value"] = True
    thread.join(timeout=1)
    server.close()


def test_server_removes_stale_socket_path(tmp_path):
    socket_path = tmp_path / "stale.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(socket_path))
    stale.close()

    server = UdsJsonlServer(socket_path=socket_path)
    server.start()
    assert socket_path.exists()
    server.close()
```

- [ ] **Step 2: Run the server tests and verify they fail because the server does not exist yet**

Run:

```bash
python3 -m pytest worker_ipc/tests/test_server.py -q
```

Expected:

```text
ERROR ... ModuleNotFoundError: No module named 'worker_ipc.server'
```

- [ ] **Step 3: Implement the synchronous UDS server**

`worker_ipc/worker_ipc/server.py`

```python
import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .exceptions import ProtocolError
from .jsonl import read_json_line, write_json_line
from .messages import Request, Response


class UdsJsonlServer:
    def __init__(
        self,
        socket_path: str | Path,
        *,
        trace_path: str | Path | None = None,
        worker_name: str | None = None,
        logger: logging.Logger | None = None,
        accept_timeout: float = 0.1,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.trace_path = Path(trace_path) if trace_path else None
        self.worker_name = worker_name or "worker"
        self.logger = logger or logging.getLogger(__name__)
        self.accept_timeout = accept_timeout
        self._server_socket: socket.socket | None = None

    @classmethod
    def from_env(cls) -> "UdsJsonlServer":
        return cls(
            socket_path=os.environ["WORKER_IPC_SOCKET_PATH"],
            trace_path=os.environ.get("WORKER_IPC_TRACE_PATH"),
            worker_name=os.environ.get("WORKER_IPC_WORKER_NAME"),
        )

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            if self._path_is_live_socket():
                raise RuntimeError(f"socket already in use: {self.socket_path}")
            self.socket_path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.socket_path))
        sock.listen(1)
        sock.settimeout(self.accept_timeout)
        self._server_socket = sock

    def serve_forever(self, handler: Callable[[Request], Response], should_stop=None) -> None:
        if self._server_socket is None:
            self.start()
        should_stop = should_stop or (lambda: False)
        assert self._server_socket is not None
        while not should_stop():
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            with conn, conn.makefile("r", encoding="utf-8") as reader, conn.makefile("w", encoding="utf-8") as writer:
                while not should_stop():
                    raw = read_json_line(reader)
                    if raw is None:
                        break
                    request_id = raw.get("request_id") if isinstance(raw.get("request_id"), str) else None
                    command = raw.get("command") if isinstance(raw.get("command"), str) else None
                    self._write_trace("request", request_id, command, raw)
                    try:
                        request = Request.from_dict(raw)
                        response = handler(request)
                    except ValueError as exc:
                        response = Response.error(request_id, str(exc))
                    except Exception:
                        self.logger.exception("unhandled worker request error")
                        response = Response.error(request_id, "internal worker error")
                    self._write_trace("response", request_id, command, response.to_dict())
                    try:
                        write_json_line(writer, response.to_dict())
                    except BrokenPipeError:
                        break

    def close(self) -> None:
        if self._server_socket is not None:
            self._server_socket.close()
            self._server_socket = None
        if self.socket_path.exists():
            self.socket_path.unlink()

    def _path_is_live_socket(self) -> bool:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.connect(str(self.socket_path))
            return True
        except OSError:
            return False

    def _write_trace(self, event: str, request_id: str | None, command: str | None, message: dict[str, object]) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "request_id": request_id,
            "command": command,
            "message": message,
        }
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
```

- [ ] **Step 4: Run the server tests and verify they pass**

Run:

```bash
python3 -m pytest worker_ipc/tests/test_server.py -q
```

Expected:

```text
3 passed
```

### Task 4: UDS Client

**Files:**
- Create: `worker_ipc/worker_ipc/client.py`
- Modify: `worker_ipc/worker_ipc/__init__.py`
- Test: `worker_ipc/tests/test_client.py`

- [ ] **Step 1: Write failing client tests for successful RPC and timeout behavior**

```python
import threading
import time

import pytest

from worker_ipc.client import UdsJsonlClient
from worker_ipc.exceptions import ClientTimeoutError
from worker_ipc.messages import Response
from worker_ipc.server import UdsJsonlServer


def test_client_call_returns_response(tmp_path):
    socket_path = tmp_path / "client.sock"
    server = UdsJsonlServer(socket_path=socket_path)

    def handler(request):
        return Response.success(request.request_id, {"echo": request.payload["value"]})

    thread = threading.Thread(target=server.serve_forever, args=(handler,), daemon=True)
    thread.start()

    client = UdsJsonlClient(socket_path)
    client.connect()
    response = client.call("echo", {"value": 7}, request_id="req-7")
    assert response.payload == {"echo": 7}
    client.close()
    server.close()


def test_client_call_raises_timeout(tmp_path):
    socket_path = tmp_path / "timeout.sock"
    server = UdsJsonlServer(socket_path=socket_path)

    def handler(request):
        time.sleep(0.2)
        return Response.success(request.request_id, {"status": "late"})

    thread = threading.Thread(target=server.serve_forever, args=(handler,), daemon=True)
    thread.start()

    client = UdsJsonlClient(socket_path)
    client.connect()
    with pytest.raises(ClientTimeoutError):
        client.call("slow", {}, request_id="req-timeout", timeout=0.05)
```

- [ ] **Step 2: Run the client tests and verify they fail because the client module is missing**

Run:

```bash
python3 -m pytest worker_ipc/tests/test_client.py -q
```

Expected:

```text
ERROR ... ModuleNotFoundError: No module named 'worker_ipc.client'
```

- [ ] **Step 3: Implement the synchronous client**

`worker_ipc/worker_ipc/client.py`

```python
import socket
import uuid
from pathlib import Path
from typing import Any

from .exceptions import ClientTimeoutError, ProtocolError
from .jsonl import read_json_line, write_json_line
from .messages import Request, Response


class UdsJsonlClient:
    def __init__(self, socket_path: str | Path) -> None:
        self.socket_path = str(socket_path)
        self._socket: socket.socket | None = None
        self._reader = None
        self._writer = None

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        self._socket = sock
        self._reader = sock.makefile("r", encoding="utf-8")
        self._writer = sock.makefile("w", encoding="utf-8")

    def call(
        self,
        command: str,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> Response:
        if self._socket is None or self._reader is None or self._writer is None:
            raise RuntimeError("client is not connected")
        request = Request(
            request_id=request_id or str(uuid.uuid4()),
            command=command,
            payload=payload,
        )
        self._socket.settimeout(timeout)
        try:
            write_json_line(self._writer, request.to_dict())
            raw = read_json_line(self._reader)
        except socket.timeout as exc:
            raise ClientTimeoutError("timed out waiting for worker response") from exc
        finally:
            self._socket.settimeout(None)
        if raw is None:
            raise ProtocolError("worker closed connection without a response")
        return Response.from_dict(raw)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
```

`worker_ipc/worker_ipc/__init__.py`

```python
from .client import UdsJsonlClient
from .messages import Request, Response
from .server import UdsJsonlServer

__all__ = ["Request", "Response", "UdsJsonlClient", "UdsJsonlServer"]
```

- [ ] **Step 4: Add `Response.from_dict()` and run the client tests**

`worker_ipc/worker_ipc/messages.py`

```python
    @classmethod
    def from_dict(cls, value: object) -> "Response":
        if not isinstance(value, dict):
            raise ValueError("message must be a JSON object")
        request_id = value.get("request_id")
        ok = value.get("ok")
        payload = value.get("payload")
        error = value.get("error")
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("request_id must be a string when present")
        if not isinstance(ok, bool):
            raise ValueError("ok must be a boolean")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        if error is not None and not isinstance(error, str):
            raise ValueError("error must be a string when present")
        return cls(request_id=request_id, ok=ok, payload=payload, error=error)
```

Run:

```bash
python3 -m pytest worker_ipc/tests/test_client.py -q
```

Expected:

```text
2 passed
```

### Task 5: Managed Child Process And Demo Worker

**Files:**
- Create: `worker_ipc/worker_ipc/managed_process.py`
- Create: `worker_ipc/examples/echo_worker.py`
- Create: `worker_ipc/examples/echo_parent.py`
- Modify: `worker_ipc/worker_ipc/__init__.py`
- Test: `worker_ipc/tests/test_managed_process.py`

- [ ] **Step 1: Write failing integration tests for managed startup, ping readiness, and trace creation**

```python
from pathlib import Path

from worker_ipc.managed_process import ManagedChildProcess


def test_managed_process_can_start_call_and_stop(tmp_path):
    socket_path = tmp_path / "echo.sock"
    trace_path = tmp_path / "echo-trace.jsonl"

    worker = ManagedChildProcess(
        command=["python3", "-m", "worker_ipc.examples.echo_worker"],
        socket_path=socket_path,
        trace_path=trace_path,
        worker_name="echo",
    )

    worker.start()
    response = worker.call("echo", {"value": 9}, request_id="req-echo")
    assert response.ok is True
    assert response.payload == {"value": 9}
    worker.stop()
    assert trace_path.exists()
```

- [ ] **Step 2: Run the managed-process tests and verify they fail because the wrapper is missing**

Run:

```bash
python3 -m pytest worker_ipc/tests/test_managed_process.py -q
```

Expected:

```text
ERROR ... ModuleNotFoundError: No module named 'worker_ipc.managed_process'
```

- [ ] **Step 3: Implement the demo worker and the managed-process helper**

`worker_ipc/examples/echo_worker.py`

```python
from worker_ipc import Request, Response, UdsJsonlServer


def handle_request(request: Request) -> Response:
    if request.command == "ping":
        return Response.success(request.request_id, {"status": "ready"})
    if request.command == "echo":
        return Response.success(request.request_id, request.payload)
    return Response.error(request.request_id, f"unknown command: {request.command}")


def main() -> int:
    server = UdsJsonlServer.from_env()
    server.serve_forever(handle_request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`worker_ipc/worker_ipc/managed_process.py`

```python
import os
import subprocess
import time
from pathlib import Path

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
    ) -> None:
        self.command = command
        self.socket_path = Path(socket_path)
        self.trace_path = Path(trace_path) if trace_path else None
        self.worker_name = worker_name or "worker"
        self.startup_timeout = startup_timeout
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
            try:
                client = UdsJsonlClient(self.socket_path)
                client.connect()
                response = client.call("ping", {}, request_id="startup-ping", timeout=0.5)
                if not response.ok:
                    raise WorkerStartError("worker ping failed during startup")
                self._client = client
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.05)
        raise WorkerStartError(f"worker did not become ready: {last_error}")

    def call(self, command, payload, *, request_id=None, timeout=None):
        if self._client is None:
            raise RuntimeError("managed worker is not started")
        return self._client.call(command, payload, request_id=request_id, timeout=timeout)

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None
```

`worker_ipc/examples/echo_parent.py`

```python
from pathlib import Path
from tempfile import gettempdir

from worker_ipc import ManagedChildProcess


def main() -> int:
    root = Path(gettempdir())
    worker = ManagedChildProcess(
        command=["python3", "-m", "worker_ipc.examples.echo_worker"],
        socket_path=root / "echo-worker.sock",
        trace_path=root / "echo-worker-trace.jsonl",
        worker_name="echo",
    )
    worker.start()
    try:
        print(worker.call("echo", {"value": 1}).to_dict())
        print(worker.call("echo", {"value": 2}).to_dict())
    finally:
        worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Export the managed wrapper and run the integration tests**

`worker_ipc/worker_ipc/__init__.py`

```python
from .client import UdsJsonlClient
from .managed_process import ManagedChildProcess
from .messages import Request, Response
from .server import UdsJsonlServer

__all__ = [
    "ManagedChildProcess",
    "Request",
    "Response",
    "UdsJsonlClient",
    "UdsJsonlServer",
]
```

Run:

```bash
python3 -m pytest worker_ipc/tests/test_managed_process.py -q
```

Expected:

```text
1 passed
```

### Task 6: README And Full Verification

**Files:**
- Create: `worker_ipc/README.md`
- Modify: `worker_ipc/examples/echo_parent.py`
- Test: `worker_ipc/tests/test_messages.py`
- Test: `worker_ipc/tests/test_jsonl.py`
- Test: `worker_ipc/tests/test_server.py`
- Test: `worker_ipc/tests/test_client.py`
- Test: `worker_ipc/tests/test_managed_process.py`

- [ ] **Step 1: Run the full test suite before writing docs and confirm the implementation is already green**

Run:

```bash
python3 -m pytest worker_ipc/tests -q
```

Expected:

```text
all existing worker_ipc tests pass before the README is finalized
```

- [ ] **Step 2: Write the README with parent and worker usage**

`worker_ipc/README.md`

```markdown
# worker_ipc

Internal package for synchronous parent-child worker communication over Unix Domain Sockets using JSON Lines.

## What It Provides

- `Request` and `Response` message models
- `UdsJsonlServer` for worker-side request handling
- `UdsJsonlClient` for synchronous calls
- `ManagedChildProcess` for parent-side startup, readiness checks, and shutdown
- optional request/response trace files

## Environment Variables

- `WORKER_IPC_SOCKET_PATH`: required
- `WORKER_IPC_TRACE_PATH`: optional
- `WORKER_IPC_WORKER_NAME`: optional

## Minimal Worker

```python
from worker_ipc import Request, Response, UdsJsonlServer


def handle_request(request: Request) -> Response:
    if request.command == "ping":
        return Response.success(request.request_id, {"status": "ready"})
    return Response.error(request.request_id, "unknown command")


server = UdsJsonlServer.from_env()
server.serve_forever(handle_request)
```

## Minimal Parent

```python
from worker_ipc import ManagedChildProcess


worker = ManagedChildProcess(
    command=["python3", "-m", "worker_ipc.examples.echo_worker"],
    socket_path="/tmp/echo.sock",
    trace_path="/tmp/echo-trace.jsonl",
    worker_name="echo",
)

worker.start()
response = worker.call("echo", {"value": 1})
worker.stop()
```
```

- [ ] **Step 3: Run the example parent script as a smoke test**

Run:

```bash
PYTHONPATH=worker_ipc python3 worker_ipc/examples/echo_parent.py
```

Expected:

```text
{'ok': True, 'payload': {'value': 1}, 'request_id': '...'}
{'ok': True, 'payload': {'value': 2}, 'request_id': '...'}
```

- [ ] **Step 4: Run the full verification suite and keep the exact command for implementation handoff**

Run:

```bash
python3 -m pytest worker_ipc/tests -q
```

Expected:

```text
all worker_ipc tests pass
```
