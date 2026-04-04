from __future__ import annotations

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
                raise RuntimeError("socket already in use: %s" % self.socket_path)
            self.socket_path.unlink()

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.socket_path))
        sock.listen(1)
        sock.settimeout(self.accept_timeout)
        self._server_socket = sock

    def serve_forever(
        self,
        handler: Callable[[Request], Response],
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        if self._server_socket is None:
            self.start()

        should_stop = should_stop or (lambda: False)
        assert self._server_socket is not None

        while not should_stop():
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                continue

            with conn:
                reader = conn.makefile("r", encoding="utf-8")
                writer = conn.makefile("w", encoding="utf-8")
                try:
                    while not should_stop():
                        try:
                            raw = read_json_line(reader)
                        except ProtocolError as exc:
                            response = Response.error(None, str(exc))
                            self._write_trace("response", None, None, response.to_dict())
                            try:
                                write_json_line(writer, response.to_dict())
                            except BrokenPipeError:
                                pass
                            break

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
                finally:
                    try:
                        reader.close()
                    except OSError:
                        pass
                    try:
                        writer.close()
                    except OSError:
                        pass

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

    def _write_trace(
        self,
        event: str,
        request_id: str | None,
        command: str | None,
        message: dict[str, object],
    ) -> None:
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
