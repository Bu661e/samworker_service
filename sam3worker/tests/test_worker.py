from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def test_sam3_worker_infer_example_inputs_returns_masks_and_bboxes(
    sam3_infer_response: dict[str, object],
    sam3_test_inputs: dict[str, object],
) -> None:
    image_path = Path(str(sam3_test_inputs["image_path"]))
    bboxes = sam3_test_inputs["bboxes"]
    results = sam3_infer_response["results"]

    assert sam3_infer_response["worker"] == "sam3"
    assert sam3_infer_response["prompt_mode"] == "bbox"
    assert sam3_infer_response["image_path"] == str(image_path)
    assert sam3_infer_response["batch_model_inference_ms"] > 0.0
    assert len(results) == len(bboxes)
    assert [item["label"] for item in results] == [item["label"] for item in bboxes]
    assert all(item["found"] is True for item in results)

    for expected, result in zip(bboxes, results):
        assert result["label"] == expected["label"]
        assert result["prompt_bbox_2d"] == expected["bbox_2d"]
        assert result["avg_inference_ms"] > 0.0
        assert result["bbox_2d"] is not None
        assert result["mask_path"] is not None

        x1, y1, x2, y2 = result["bbox_2d"]
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 640

        mask_path = Path(result["mask_path"])
        assert mask_path.is_file()
        assert mask_path.parent == Path(sam3_infer_response["output_dir"])
        assert mask_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

        with Image.open(mask_path) as mask:
            mask_array = np.asarray(mask.convert("L"))
        assert mask_array.shape == (640, 640)
        assert int(mask_array.max()) == 255
        assert int((mask_array > 0).sum()) > 0
