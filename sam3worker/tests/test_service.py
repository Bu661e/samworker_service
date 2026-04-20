from __future__ import annotations

from sam3worker import Sam3WorkerClient


def test_sam3_worker_ping_returns_ready(sam3_client: Sam3WorkerClient) -> None:
    assert sam3_client.ping(timeout=30.0) == {"status": "ready"}


def test_sam3_worker_describe_reports_loaded_model(sam3_client: Sam3WorkerClient) -> None:
    description = sam3_client.describe(timeout=30.0)

    assert description["worker"] == "sam3"
    assert description["status"] == "ready"
    assert description["weight_path"] == "/root/sam3.pt"
    assert description["supported_commands"] == ["ping", "describe", "infer"]
    assert description["prompt_modes"] == ["bbox"]
    assert description["model_loaded"] is True


def test_sam3_worker_call_raw_returns_response_envelope(
    sam3_client: Sam3WorkerClient,
) -> None:
    response = sam3_client.call_raw(
        "describe",
        {},
        request_id="sam3-raw-describe",
        timeout=30.0,
    )

    assert response.ok is True
    assert response.request_id == "sam3-raw-describe"
    assert response.error is None
    assert response.payload["worker"] == "sam3"
