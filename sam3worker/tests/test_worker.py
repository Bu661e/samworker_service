from __future__ import annotations

import sys
from pathlib import Path


WORKER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER_DIR))

import worker as worker_module  # noqa: E402
from worker_ipc import Request  # noqa: E402


def test_handle_request_returns_error_response_for_runtime_error(monkeypatch) -> None:
    def boom(command: str, payload: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("ultralytics is not installed")

    monkeypatch.setattr(worker_module, "handle_command", boom)

    response = worker_module._handle_request(
        Request(request_id="req-1", command="infer", payload={})
    )

    assert response.ok is False
    assert response.request_id == "req-1"
    assert response.payload == {}
    assert response.error == "ultralytics is not installed"


def test_main_initializes_model_before_serving(monkeypatch) -> None:
    order: list[str] = []

    class FakeServer:
        def serve_forever(self, handler) -> None:
            order.append("serve")

    def fake_initialize_model() -> object:
        order.append("init")
        return object()

    monkeypatch.setattr(worker_module, "initialize_model", fake_initialize_model)
    monkeypatch.setattr(
        worker_module.UdsJsonlServer,
        "from_env",
        classmethod(lambda cls: FakeServer()),
    )

    assert worker_module.main() == 0
    assert order == ["init", "serve"]
