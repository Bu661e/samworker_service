import json
import os
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker_ipc.messages import Response
from worker_ipc.server import UdsJsonlServer


def _short_socket_path(name: str) -> Path:
    return Path("/tmp") / f"worker-ipc-{name}-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"


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
    socket_path = _short_socket_path("server")
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
    socket_path = _short_socket_path("trace")
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
    socket_path = _short_socket_path("stale")
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(socket_path))
    stale.close()

    server = UdsJsonlServer(socket_path=socket_path)
    server.start()
    assert socket_path.exists()
    server.close()
