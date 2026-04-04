from __future__ import annotations

import json
from typing import Any, TextIO

from .exceptions import ProtocolError


def write_json_line(stream: TextIO, message: dict[str, Any]) -> None:
    if not isinstance(message, dict):
        raise ProtocolError("top-level JSON value must be an object")

    stream.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


def read_json_line(stream: TextIO) -> dict[str, Any] | None:
    line = stream.readline()
    if line == "":
        return None

    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid JSON line") from exc

    if not isinstance(value, dict):
        raise ProtocolError("top-level JSON value must be an object")

    return value
