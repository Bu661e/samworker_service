import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker_ipc.messages import Request, Response


def test_request_from_dict_round_trips_valid_message():
    request = Request.from_dict(
        {
            "request_id": "req-1",
            "command": "ping",
            "payload": {"value": 1},
        }
    )

    assert request.request_id == "req-1"
    assert request.command == "ping"
    assert request.payload == {"value": 1}
    with pytest.raises(AttributeError):
        _ = request.__dict__
    assert request.to_dict() == {
        "request_id": "req-1",
        "command": "ping",
        "payload": {"value": 1},
    }


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "message must be a JSON object"),
        (
            {"request_id": "", "command": "ping", "payload": {}},
            "request_id must be a non-empty string",
        ),
        (
            {"request_id": "req-1", "command": "", "payload": {}},
            "command must be a non-empty string",
        ),
        (
            {"request_id": "req-1", "command": "ping", "payload": []},
            "payload must be a JSON object",
        ),
    ],
)
def test_request_from_dict_rejects_invalid_messages(value, message):
    with pytest.raises(ValueError, match=message):
        Request.from_dict(value)


def test_response_success_error_and_optional_fields():
    success = Response.success("req-1", {"status": "ready"})
    error = Response.error("req-1", "boom")
    notification = Response.success(None, {})

    assert success.to_dict() == {
        "request_id": "req-1",
        "ok": True,
        "payload": {"status": "ready"},
    }
    assert error.to_dict() == {
        "request_id": "req-1",
        "ok": False,
        "payload": {},
        "error": "boom",
    }
    assert notification.to_dict() == {
        "ok": True,
        "payload": {},
    }
