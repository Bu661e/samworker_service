from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Request:
    __slots__ = ("request_id", "command", "payload")

    request_id: str
    command: str
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, value: object) -> "Request":
        if not isinstance(value, dict):
            raise ValueError("message must be a JSON object")

        request_id = value.get("request_id")
        command = value.get("command")
        payload = value.get("payload")

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(command, str) or not command:
            raise ValueError("command must be a non-empty string")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")

        return cls(request_id=request_id, command=command, payload=payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "command": self.command,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class Response:
    request_id: Optional[str]
    ok: bool
    payload: dict[str, Any]
    error: Optional[str] = None

    @classmethod
    def success(cls, request_id: Optional[str], payload: dict[str, Any]) -> "Response":
        return cls(request_id=request_id, ok=True, payload=payload)

    @classmethod
    def from_dict(cls, value: object) -> "Response":
        if not isinstance(value, dict):
            raise ValueError("message must be a JSON object")

        request_id = value.get("request_id")
        ok = value.get("ok")
        payload = value.get("payload")
        error = value.get("error")

        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("request_id must be a string when present")
        if not isinstance(ok, bool):
            raise ValueError("ok must be a boolean")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        if error is not None and not isinstance(error, str):
            raise ValueError("error must be a string when present")

        return cls(request_id=request_id, ok=ok, payload=payload, error=error)

    def to_dict(self) -> dict[str, Any]:
        message: dict[str, Any] = {"ok": self.ok, "payload": self.payload}
        if self.request_id is not None:
            message["request_id"] = self.request_id
        if self.error is not None:
            message["error"] = self.error
        return message


def _response_error(
    cls,
    request_id: Optional[str],
    error: str,
    payload: Optional[dict[str, Any]] = None,
) -> Response:
    return cls(request_id=request_id, ok=False, payload=payload or {}, error=error)


Response.error = classmethod(_response_error)
