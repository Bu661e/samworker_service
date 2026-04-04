import sys
from pathlib import Path
from tempfile import gettempdir

from worker_ipc import ManagedChildProcess


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    temp_root = Path(gettempdir())
    worker = ManagedChildProcess(
        command=[sys.executable, "-m", "examples.echo_worker"],
        socket_path=temp_root / "echo-worker.sock",
        trace_path=temp_root / "echo-worker-trace.jsonl",
        worker_name="echo",
        cwd=package_root,
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
