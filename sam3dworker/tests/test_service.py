from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


WORKER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER_DIR))

import service as service_module  # noqa: E402
from service import describe_service, handle_command  # noqa: E402


def test_describe_service_reports_pointmap_stage() -> None:
    assert describe_service() == {
        "worker": "sam3d",
        "third_party_dir": str(service_module.THIRD_PARTY_SAM3D_DIR),
        "config_path": str(service_module.SAM3D_CONFIG_PATH),
        "status": "ready",
        "supported_commands": ["ping", "describe", "reconstruct"],
        "reconstruct_stage": "full_inference",
        "inference_loaded": False,
    }


def test_reconstruct_requires_existing_depth_path(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(image_path)
    Image.fromarray(np.array([[255, 255], [255, 255]], dtype=np.uint8)).save(mask_path)

    payload = {
        "image_path": str(image_path.resolve()),
        "depth_path": str((tmp_path / "missing.npy").resolve()),
        "mask_path": str(mask_path.resolve()),
        "output_dir": str((tmp_path / "outputs").resolve()),
        "fx": 100.0,
        "fy": 100.0,
        "cx": 10.0,
        "cy": 10.0,
        "label": "cube_0",
    }

    with pytest.raises(ValueError, match="depth_path does not exist"):
        handle_command("reconstruct", payload)


def test_reconstruct_requires_float32_depth(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    depth_path = tmp_path / "depth.npy"

    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(image_path)
    Image.fromarray(np.array([[255, 255], [255, 255]], dtype=np.uint8)).save(mask_path)
    np.save(depth_path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))

    payload = {
        "image_path": str(image_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "output_dir": str((tmp_path / "outputs").resolve()),
        "fx": 100.0,
        "fy": 100.0,
        "cx": 10.0,
        "cy": 10.0,
        "label": "cube_0",
    }

    with pytest.raises(ValueError, match="depth array must have dtype float32"):
        handle_command("reconstruct", payload)


def test_reconstruct_requires_matching_image_depth_and_mask_sizes(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    depth_path = tmp_path / "depth.npy"

    Image.new("RGB", (3, 2), color=(255, 0, 0)).save(image_path)
    Image.fromarray(np.array([[255, 255], [255, 255]], dtype=np.uint8)).save(mask_path)
    np.save(depth_path, np.ones((2, 2), dtype=np.float32))

    payload = {
        "image_path": str(image_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "output_dir": str((tmp_path / "outputs").resolve()),
        "fx": 100.0,
        "fy": 100.0,
        "cx": 10.0,
        "cy": 10.0,
        "label": "cube_0",
    }

    with pytest.raises(ValueError, match="image, depth, and mask dimensions must match exactly"):
        handle_command("reconstruct", payload)


def test_reconstruct_writes_pointmap_and_returns_pose_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    depth_path = tmp_path / "depth.npy"
    output_dir = tmp_path / "outputs"

    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(image_path)
    Image.fromarray(np.array([[255, 0], [255, 255]], dtype=np.uint8)).save(mask_path)
    np.save(
        depth_path,
        np.array(
            [
                [1.0, 2.0],
                [0.0, np.nan],
            ],
            dtype=np.float32,
        ),
    )

    class FakeTensor:
        def __init__(self, value):
            self._value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self._value

    class FakeGaussian:
        def save_ply(self, path: str) -> None:
            Path(path).write_text("ply")

    class FakeGlb:
        def export(self, path: str) -> None:
            Path(path).write_text("glb")

    def fake_run_sam3d_inference(*, request, image, mask, pointmap):
        assert request.label == "cube_0"
        assert image.shape == (2, 2, 3)
        assert mask.shape == (2, 2)
        assert set(np.unique(mask)) == {0, 1}
        assert pointmap.shape == (2, 2, 3)
        return {
            "rotation": FakeTensor([[1.0, 0.0, 0.0, 0.0]]),
            "translation": FakeTensor([[0.1, 0.2, 0.3]]),
            "scale": FakeTensor([[1.23, 1.23, 1.23]]),
            "gs": FakeGaussian(),
            "glb": FakeGlb(),
        }

    monkeypatch.setattr(service_module, "_run_sam3d_inference", fake_run_sam3d_inference)

    response = handle_command(
        "reconstruct",
        {
            "image_path": str(image_path.resolve()),
            "depth_path": str(depth_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "fx": 2.0,
            "fy": 2.0,
            "cx": 0.5,
            "cy": 0.5,
            "label": "cube_0",
            "artifact_types": ["gaussian", "mesh"],
        },
    )

    pointmap = np.load(output_dir / "pointmap.npy")
    assert pointmap.shape == (2, 2, 3)
    assert pointmap.dtype == np.float32
    np.testing.assert_allclose(
        pointmap[0, 0],
        np.array([0.25, 0.25, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        pointmap[0, 1],
        np.array([-0.5, 0.5, 2.0], dtype=np.float32),
    )
    assert np.isnan(pointmap[1, 0]).all()
    assert np.isnan(pointmap[1, 1]).all()

    assert response == {
        "worker": "sam3d",
        "label": "cube_0",
        "image_path": str(image_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "pointmap_path": str((output_dir / "pointmap.npy").resolve()),
        "pose": {
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "translation": [0.1, 0.2, 0.3],
            "scale": [1.23, 1.23, 1.23],
        },
        "artifacts": {
            "gaussian_ply_path": str((output_dir / "splat.ply").resolve()),
            "mesh_glb_path": str((output_dir / "mesh.glb").resolve()),
        },
    }

    assert (output_dir / "splat.ply").read_text() == "ply"
    assert (output_dir / "mesh.glb").read_text() == "glb"
