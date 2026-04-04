from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "worker_ipc"))

from worker_ipc import Request, Response, UdsJsonlServer  # noqa: E402

from service import handle_command  # noqa: E402


def _handle_request(request: Request) -> Response:
    try:
        payload = handle_command(request.command, request.payload)
        return Response.success(request.request_id, payload)
    except ValueError as exc:
        return Response.error(request.request_id, str(exc))


def main() -> int:
    server = UdsJsonlServer.from_env()
    server.serve_forever(_handle_request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
