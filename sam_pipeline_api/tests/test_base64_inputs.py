from __future__ import annotations

import base64

import pytest

from sam_pipeline_api.models import ReconstructObjectsRequest
from sam_pipeline_api.pipeline import PipelineInputError, _decode_base64_bytes


def test_reconstruct_request_accepts_base64_artifacts() -> None:
    request = ReconstructObjectsRequest.model_validate(
        {
            "task": "pick object",
            "bboxes": [
                {
                    "label": "cube",
                    "bbox_2d": [10, 20, 30, 40],
                }
            ],
            "camera": {
                "id": "camera-0",
                "intrinsics": {
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 50.0,
                    "cy": 50.0,
                },
                "rgb_image": {
                    "ref": {
                        "id": "rgb-1",
                        "kind": "artifact_file",
                        "content_type": "image/png",
                        "content_base64": base64.b64encode(b"png-bytes").decode("ascii"),
                    }
                },
                "depth_image": {
                    "unit": "meter",
                    "ref": {
                        "id": "depth-1",
                        "kind": "artifact_file",
                        "content_type": "application/x-npy",
                        "content_base64": base64.b64encode(b"npy-bytes").decode("ascii"),
                    },
                },
            },
        }
    )

    assert request.camera.rgb_image.ref.content_type == "image/png"
    assert request.camera.depth_image.ref.content_type == "application/x-npy"


def test_decode_base64_bytes_accepts_data_url() -> None:
    payload = base64.b64encode(b"hello-world").decode("ascii")

    assert _decode_base64_bytes(f"data:image/png;base64,{payload}") == b"hello-world"


def test_decode_base64_bytes_rejects_invalid_payload() -> None:
    with pytest.raises(PipelineInputError, match="invalid base64 artifact payload"):
        _decode_base64_bytes("not-valid-base64")
