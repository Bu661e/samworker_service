import os
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker_ipc.client import UdsJsonlClient
from worker_ipc.exceptions import ClientTimeoutError
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


def test_client_call_returns_response():
    socket_path = _short_socket_path("client")
    server = UdsJsonlServer(socket_path=socket_path)

    def handler(request):
        return Response.success(request.request_id, {"echo": request.payload["value"]})

    thread, stop_flag = _start_server(server, handler)
    client = UdsJsonlClient(socket_path)
    client.connect()
    response = client.call("echo", {"value": 7}, request_id="req-7")

    assert response.payload == {"echo": 7}

    client.close()
    stop_flag["value"] = True
    thread.join(timeout=1)
    server.close()


def test_client_call_raises_timeout():
    socket_path = _short_socket_path("timeout")
    server = UdsJsonlServer(socket_path=socket_path)

    def handler(request):
        time.sleep(0.2)
        return Response.success(request.request_id, {"status": "late"})

    thread, stop_flag = _start_server(server, handler)
    client = UdsJsonlClient(socket_path)
    client.connect()

    with pytest.raises(ClientTimeoutError):
        client.call("slow", {}, request_id="req-timeout", timeout=0.05)

    client.close()
    stop_flag["value"] = True
    thread.join(timeout=1)
    server.close()
