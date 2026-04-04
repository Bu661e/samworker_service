import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker_ipc.exceptions import ProtocolError
from worker_ipc.jsonl import read_json_line, write_json_line


class _FlushTrackingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1
        super().flush()


def test_write_json_line_serializes_compact_json_and_flushes():
    stream = _FlushTrackingStream()

    write_json_line(stream, {"request_id": "req-1", "command": "ping", "payload": {}})

    assert stream.getvalue() == '{"request_id":"req-1","command":"ping","payload":{}}\n'
    assert stream.flush_calls == 1


def test_write_json_line_rejects_non_object_top_level_payload():
    stream = io.StringIO()

    with pytest.raises(ProtocolError, match="top-level JSON value must be an object"):
        write_json_line(stream, [])


def test_read_json_line_returns_none_on_eof():
    stream = io.StringIO("")

    assert read_json_line(stream) is None


def test_read_json_line_returns_dict():
    stream = io.StringIO('{"request_id":"req-1","command":"ping","payload":{}}\n')

    assert read_json_line(stream) == {
        "request_id": "req-1",
        "command": "ping",
        "payload": {},
    }


def test_read_json_line_rejects_malformed_json_and_non_object_top_level_payload():
    malformed_stream = io.StringIO('{"request_id":\n')
    non_object_stream = io.StringIO('["bad"]\n')

    with pytest.raises(ProtocolError, match="invalid JSON line"):
        read_json_line(malformed_stream)

    with pytest.raises(ProtocolError, match="top-level JSON value must be an object"):
        read_json_line(non_object_stream)
