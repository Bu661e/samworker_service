from __future__ import annotations

import socket
import uuid
from pathlib import Path
from typing import Any, TextIO

from .exceptions import ClientTimeoutError, ProtocolError
from .jsonl import read_json_line, write_json_line
from .messages import Request, Response


class UdsJsonlClient:
    def __init__(self, socket_path: str | Path) -> None:
        self.socket_path = str(socket_path)
        self._socket: socket.socket | None = None
        self._reader: TextIO | None = None
        self._writer: TextIO | None = None

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
        except (socket.timeout, OSError) as exc:
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
