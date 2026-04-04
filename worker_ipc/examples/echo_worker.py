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
