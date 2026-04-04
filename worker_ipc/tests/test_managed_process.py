import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker_ipc.managed_process import ManagedChildProcess


def _short_socket_path(name: str) -> Path:
    return Path("/tmp") / f"worker-ipc-{name}-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"


def test_managed_process_can_start_call_and_stop(tmp_path):
    package_root = Path(__file__).resolve().parents[1]
    socket_path = _short_socket_path("echo")
    trace_path = tmp_path / "echo-trace.jsonl"

    worker = ManagedChildProcess(
        command=[sys.executable, "-m", "examples.echo_worker"],
        socket_path=socket_path,
        trace_path=trace_path,
        worker_name="echo",
        cwd=package_root,
    )

    worker.start()
    response = worker.call("echo", {"value": 9}, request_id="req-echo")
    assert response.ok is True
    assert response.payload == {"value": 9}
    worker.stop()
    assert trace_path.exists()
