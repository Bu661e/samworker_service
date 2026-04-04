# Worker IPC Design

## Summary

This document defines a reusable internal Python package for the common case where a parent process starts a long-lived child worker process and sends synchronous requests to it over a local IPC channel.

The package is intended to be general-purpose within this repository. `samworker` will consume it first, but the package must not contain SAM3 or SAM3D business logic.

## Goals

- Provide a simple, reusable local IPC package for parent-child worker communication.
- Keep protocol traffic separate from `stdout` and `stderr`.
- Support long-lived child workers that handle many synchronous requests.
- Reuse the same package for multiple workers with different business handlers.
- Include an optional trace facility that records raw request and response JSON.

## Non-Goals

- Asynchronous job submission or progress tracking.
- Multi-client concurrent request execution within one worker.
- Streaming responses.
- Cross-machine transport.
- Automatic log rotation or trace retention policies.

## Repository Layout

The reusable IPC package lives beside `samworker` at repository root:

```text
repo/
  worker_ipc/
    README.md
    pyproject.toml
    worker_ipc/
      __init__.py
      exceptions.py
      messages.py
      jsonl.py
      client.py
      server.py
      managed_process.py
    examples/
      echo_worker.py
      echo_parent.py
    tests/
      ...
  samworker/
    workers/
      sam3_worker.py
      sam3d_worker.py
```

`worker_ipc/` contains only reusable IPC code. Business commands such as `infer` and `reconstruct` remain in worker-specific modules.

## Transport Choice

The package uses Unix Domain Sockets with JSON Lines framing.

Reasons:

- UDS keeps protocol traffic separate from `stdout` and `stderr`.
- JSON Lines is lighter than length-prefixed framing while remaining easy to debug.
- The parent and child processes run on the same machine, so TCP is unnecessary.

Each message is one JSON object serialized to one line ending with `\n`.

## Protocol Model

### Request

```json
{"request_id":"req-123","command":"infer","payload":{"image_path":"/tmp/a.png"}}
```

Fields:

- `request_id`: required non-empty string
- `command`: required non-empty string
- `payload`: required JSON object

### Response

Success:

```json
{"request_id":"req-123","ok":true,"payload":{"detections":[]}}
```

Failure:

```json
{"request_id":"req-123","ok":false,"payload":{},"error":"image_path is required"}
```

Fields:

- `request_id`: echoed from request when available
- `ok`: required boolean
- `payload`: required JSON object
- `error`: optional string, present for failures

### Rules

- Top-level message must be a JSON object.
- Each message occupies exactly one line.
- The protocol package does not interpret `command` semantics.
- The first version is synchronous request/response only.

## Public API

The package exposes four main concepts:

```python
from worker_ipc import Request, Response
from worker_ipc import UdsJsonlClient, UdsJsonlServer
from worker_ipc import ManagedChildProcess
```

### `Request`

Dataclass with:

- `request_id: str`
- `command: str`
- `payload: dict[str, Any]`

It supports JSON-object validation and conversion.

### `Response`

Dataclass with:

- `request_id: str | None`
- `ok: bool`
- `payload: dict[str, Any]`
- `error: str | None = None`

It supports success and error constructors plus JSON-object conversion.

### `UdsJsonlClient`

Minimal synchronous client:

- `connect()`
- `call(command, payload, request_id=None, timeout=None) -> Response`
- `close()`

### `UdsJsonlServer`

Minimal synchronous server:

- constructor accepts `socket_path` and optional `trace_path` / `worker_name`
- `from_env()`
- `serve_forever(handler, should_stop=None)`
- `close()`

The server accepts one connection at a time and handles requests strictly serially.

### `ManagedChildProcess`

Parent-side lifecycle helper:

- `start()`
- `call(command, payload, request_id=None, timeout=None) -> Response`
- `stop()`

Responsibilities:

- start the child process with `subprocess.Popen`
- inject environment variables required by the worker
- wait for the socket to become reachable
- send `ping` and require a successful response before marking the worker ready
- stop and clean up the child process

The class is optional. Callers that want more control may use `UdsJsonlClient` directly.

## Environment Variable Contract

`ManagedChildProcess` passes worker startup settings through environment variables.

Required:

- `WORKER_IPC_SOCKET_PATH`

Optional:

- `WORKER_IPC_TRACE_PATH`
- `WORKER_IPC_WORKER_NAME`

Workers read these values from `os.environ` at startup. This keeps worker entrypoints thin and avoids repeating argument parsing across multiple workers.

## Lifecycle Rules

### Server Side

- One worker process owns one socket path.
- One connection is accepted at a time.
- Requests are handled serially.
- If the socket path already exists on startup:
  - try connecting to it
  - if connection succeeds, treat it as an already-running worker and raise an error
  - if connection fails, remove the stale socket file and continue
- On shutdown, remove the socket file

### Parent Side

- `start()` launches the child process
- parent waits until it can connect to the socket
- parent sends `ping`
- child is considered ready only after a successful `ping` response

## Trace Recording

Protocol trace is separate from normal process logs.

- Parent and child continue to manage `stdout` and `stderr` as they choose.
- IPC trace is an optional protocol-level feature.
- If `WORKER_IPC_TRACE_PATH` is set, the worker appends request and response events to that file as JSON Lines.

Example trace records:

```json
{"ts":"2026-04-05T12:34:56.123456Z","event":"request","request_id":"req-1","command":"infer","message":{"request_id":"req-1","command":"infer","payload":{"image_path":"/tmp/a.png"}}}
{"ts":"2026-04-05T12:34:56.456789Z","event":"response","request_id":"req-1","command":"infer","message":{"request_id":"req-1","ok":true,"payload":{"detections":[]}}}
```

The parent chooses the full trace file path. A typical filename should include worker name, timestamp, and PID to avoid collisions.

## Error Handling

- Invalid request shape:
  - server returns `ok=false`
  - include `request_id` when it can be recovered
- Handler exception:
  - server logs the exception
  - server returns `ok=false` with a generic internal error
- Client timeout:
  - client raises a timeout exception
  - client does not fabricate a response
- Reply send failure after client disconnect:
  - server logs and closes the connection
  - server does not crash the entire worker loop
- Worker startup failure, premature exit, or socket readiness failure:
  - `ManagedChildProcess.start()` raises a specific startup exception

## Recommended Built-In Worker Command

The protocol layer does not hardcode business commands, but workers are expected to support:

- `ping`

Recommended successful response:

```json
{"request_id":"req-1","ok":true,"payload":{"status":"ready"}}
```

`ManagedChildProcess.start()` uses `ping` for readiness validation.

## First Delivery Scope

The first implementation must include:

- the internal package skeleton
- request and response models
- JSON Lines read/write helpers
- synchronous UDS client
- synchronous UDS server
- managed child-process helper
- README with usage examples
- demo worker and demo parent
- automated tests for protocol, server behavior, and managed startup flow

The first implementation explicitly excludes:

- async job protocol
- cancel/status/progress commands
- streaming payloads
- multi-client concurrent serving
- TCP transport
- trace rotation

## Example Usage

### Worker

```python
from worker_ipc import Request, Response, UdsJsonlServer


def handle_request(request: Request) -> Response:
    if request.command == "ping":
        return Response.success(request.request_id, {"status": "ready"})
    if request.command == "echo":
        return Response.success(request.request_id, request.payload)
    return Response.error(request.request_id, "unknown command")


server = UdsJsonlServer.from_env()
server.serve_forever(handle_request)
```

### Parent

```python
from worker_ipc import ManagedChildProcess


worker = ManagedChildProcess(
    command=["python3", "-m", "worker_ipc.examples.echo_worker"],
    socket_path="/tmp/echo.sock",
    trace_path="/tmp/echo-20260405-120000-12345.jsonl",
    worker_name="echo",
)

worker.start()
response = worker.call("echo", {"value": 1})
worker.stop()
```
