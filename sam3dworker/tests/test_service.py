from __future__ import annotations

from sam3dworker import Sam3dWorkerClient


def test_sam3d_worker_ping_returns_ready(sam3d_client: Sam3dWorkerClient) -> None:
    assert sam3d_client.ping(timeout=30.0) == {"status": "ready"}


def test_sam3d_worker_describe_reports_loaded_inference(
    sam3d_client: Sam3dWorkerClient,
) -> None:
    description = sam3d_client.describe(timeout=30.0)

    assert description["worker"] == "sam3d"
    assert description["status"] == "ready"
    assert description["supported_commands"] == ["ping", "describe", "reconstruct"]
    assert description["reconstruct_stage"] == "full_inference"
    assert description["inference_loaded"] is True
    assert str(description["config_path"]).endswith("third_party/SAM3D-object/checkpoints/hf/pipeline.yaml")


def test_sam3d_worker_call_raw_returns_response_envelope(
    sam3d_client: Sam3dWorkerClient,
) -> None:
    response = sam3d_client.call_raw(
        "describe",
        {},
        request_id="sam3d-raw-describe",
        timeout=30.0,
    )

    assert response.ok is True
    assert response.request_id == "sam3d-raw-describe"
    assert response.error is None
    assert response.payload["worker"] == "sam3d"
