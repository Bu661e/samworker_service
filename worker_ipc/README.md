# worker_ipc

Internal package for synchronous parent-child worker communication over Unix Domain Sockets using JSON Lines.

## What It Provides

- `Request` and `Response` message models
- `UdsJsonlServer` for worker-side request handling
- `UdsJsonlClient` for synchronous request/response calls
- `ManagedChildProcess` for parent-side startup, readiness checks, and shutdown
- optional request/response trace files written as JSON Lines

## Environment Variables

`ManagedChildProcess` injects these into the child worker:

- `WORKER_IPC_SOCKET_PATH`: required
- `WORKER_IPC_TRACE_PATH`: optional
- `WORKER_IPC_WORKER_NAME`: optional

Workers can also construct `UdsJsonlServer` directly if they do not want to use environment-based startup.

## Request And Response Shape

Request:

```json
{"request_id":"req-1","command":"ping","payload":{}}
```

Response:

```json
{"request_id":"req-1","ok":true,"payload":{"status":"ready"}}
```

Error response:

```json
{"request_id":"req-1","ok":false,"payload":{},"error":"unknown command"}
```

## Minimal Worker

```python
from worker_ipc import Request, Response, UdsJsonlServer


def handle_request(request: Request) -> Response:
    if request.command == "ping":
        return Response.success(request.request_id, {"status": "ready"})
    if request.command == "echo":
        return Response.success(request.request_id, request.payload)
    return Response.error(request.request_id, f"unknown command: {request.command}")


server = UdsJsonlServer.from_env()
server.serve_forever(handle_request)
```

## Minimal Parent

```python
import sys
from pathlib import Path

from worker_ipc import ManagedChildProcess


package_root = Path("worker_ipc").resolve()
worker = ManagedChildProcess(
    command=[sys.executable, "-m", "examples.echo_worker"],
    socket_path=Path("/tmp/worker-ipc-demo.sock"),
    trace_path=Path("/tmp/worker-ipc-demo-trace.jsonl"),
    worker_name="echo",
    cwd=package_root,
)

worker.start()
try:
    response = worker.call("echo", {"value": 1})
    print(response.to_dict())
finally:
    worker.stop()
```

## Trace Files

If `trace_path` is provided to `ManagedChildProcess`, or `WORKER_IPC_TRACE_PATH` is set manually, the worker appends request and response records to that file as JSON Lines.

Example trace records:

```json
{"ts":"2026-04-05T12:34:56.123456+00:00","event":"request","request_id":"req-1","command":"echo","message":{"request_id":"req-1","command":"echo","payload":{"value":1}}}
{"ts":"2026-04-05T12:34:56.234567+00:00","event":"response","request_id":"req-1","command":"echo","message":{"request_id":"req-1","ok":true,"payload":{"value":1}}}
```

Trace output is separate from normal process `stdout` and `stderr`.

## Running The Demo From This Repository

From the repository root:

```bash
PYTHONPATH=worker_ipc python3 worker_ipc/examples/echo_parent.py
```

The demo worker itself is launched from the `worker_ipc/` project root with:

```bash
python3 -m examples.echo_worker
```

That launch pattern keeps the nested package layout importable without requiring installation.

## Current Limits

- synchronous request/response only
- one connection at a time per worker
- no streaming responses
- no async job protocol
- no automatic log or trace rotation

## Verification

Current test command:

```bash
python3 -m pytest worker_ipc/tests -q
```
