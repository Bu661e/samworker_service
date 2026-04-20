from __future__ import annotations

from pathlib import Path

import numpy as np


def test_sam3d_worker_reconstruct_payloads_return_pose_and_artifacts(
    sam3d_reconstruct_responses: list[dict[str, object]],
    sam3d_test_payloads: list[dict[str, object]],
) -> None:
    assert len(sam3d_reconstruct_responses) == len(sam3d_test_payloads)

    expected_labels = [str(payload["label"]) for payload in sam3d_test_payloads]
    response_labels = [str(response["label"]) for response in sam3d_reconstruct_responses]
    assert response_labels == expected_labels

    for payload, response in zip(sam3d_test_payloads, sam3d_reconstruct_responses):
        output_dir = Path(str(payload["output_dir"]))

        assert response["worker"] == "sam3d"
        assert response["label"] == payload["label"]
        assert response["image_path"] == str(payload["image_path"])
        assert response["depth_path"] == str(payload["depth_path"])
        assert response["mask_path"] == str(payload["mask_path"])
        assert response["output_dir"] == str(output_dir)
        assert response["model_inference_ms"] > 0.0

        pose = response["pose"]
        assert len(pose["rotation"]) == 4
        assert len(pose["translation"]) == 3
        assert len(pose["scale"]) == 3
        assert all(isinstance(value, float) for value in pose["rotation"])
        assert all(isinstance(value, float) for value in pose["translation"])
        assert all(isinstance(value, float) for value in pose["scale"])

        pointmap_path = Path(str(response["pointmap_path"]))
        assert pointmap_path.is_file()
        assert pointmap_path.parent == output_dir

        artifacts = response["artifacts"]
        assert set(artifacts.keys()) == {"gaussian_ply_path", "mesh_glb_path"}
        gaussian_path = Path(str(artifacts["gaussian_ply_path"]))
        mesh_path = Path(str(artifacts["mesh_glb_path"]))
        assert gaussian_path.is_file()
        assert mesh_path.is_file()
        assert gaussian_path.parent == output_dir
        assert mesh_path.parent == output_dir


def test_sam3d_worker_outputs_pointmap_file_under_runs_directory(
    sam3d_reconstruct_responses: list[dict[str, object]],
    sam3d_run_root: Path,
) -> None:
    for response in sam3d_reconstruct_responses:
        pointmap_path = Path(str(response["pointmap_path"]))
        assert pointmap_path.is_relative_to(sam3d_run_root)

        pointmap = np.load(pointmap_path)
        assert pointmap.shape == (640, 640, 3)
        assert pointmap.dtype == np.float32
        assert np.isfinite(pointmap[..., 2]).any()
