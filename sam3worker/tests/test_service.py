from __future__ import annotations

import sys
from pathlib import Path

import pytest


WORKER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER_DIR))

import service as service_module  # noqa: E402
from service import describe_service, handle_command, initialize_model  # noqa: E402


def test_describe_service_reports_current_worker_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL", object())
    assert describe_service() == {
        "worker": "sam3",
        "status": "ready",
        "weight_path": "/root/sam3.pt",
        "supported_commands": ["ping", "describe", "infer"],
        "prompt_modes": ["bbox"],
        "model_loaded": True,
    }


def test_handle_command_describe_matches_describe_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "_MODEL", object())
    assert handle_command("describe", {}) == describe_service()


def test_initialize_model_loads_once_and_reuses_cached_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    weight_path = tmp_path / "sam3.pt"
    weight_path.write_text("stub weight")
    calls: list[Path] = []
    sentinel = object()

    def fake_construct_model(path: Path) -> object:
        calls.append(path)
        return sentinel

    monkeypatch.setattr(service_module, "SAM3_WEIGHT_PATH", str(weight_path))
    monkeypatch.setattr(service_module, "_MODEL", None)
    monkeypatch.setattr(service_module, "_construct_model", fake_construct_model)

    assert initialize_model() is sentinel
    assert initialize_model() is sentinel
    assert calls == [weight_path]


def test_infer_requires_existing_absolute_image_path(tmp_path: Path) -> None:
    payload = {
        "image_path": "relative/image.jpg",
        "output_dir": str((tmp_path / "outputs").resolve()),
        "bboxes": [{"label": "cube_0", "bbox_2d": [1, 2, 3, 4]}],
    }

    with pytest.raises(ValueError, match="image_path must be an absolute path"):
        handle_command("infer", payload)


def test_infer_requires_absolute_output_dir(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.jpg"
    image_path.write_text("stub image")
    payload = {
        "image_path": str(image_path.resolve()),
        "output_dir": "relative/output",
        "bboxes": [{"label": "cube_0", "bbox_2d": [1, 2, 3, 4]}],
    }

    with pytest.raises(ValueError, match="output_dir must be an absolute path"):
        handle_command("infer", payload)


def test_infer_requires_unique_labels(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.jpg"
    image_path.write_text("stub image")
    payload = {
        "image_path": str(image_path.resolve()),
        "output_dir": str((tmp_path / "outputs").resolve()),
        "bboxes": [
            {"label": "cube_0", "bbox_2d": [1, 2, 3, 4]},
            {"label": "cube_0", "bbox_2d": [5, 6, 7, 8]},
        ],
    }

    with pytest.raises(ValueError, match="duplicate bbox label: cube_0"):
        handle_command("infer", payload)


def test_infer_requires_valid_bbox_geometry(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.jpg"
    image_path.write_text("stub image")
    payload = {
        "image_path": str(image_path.resolve()),
        "output_dir": str((tmp_path / "outputs").resolve()),
        "bboxes": [{"label": "cube_0", "bbox_2d": [5, 2, 3, 4]}],
    }

    with pytest.raises(ValueError, match="must satisfy x1 < x2 and y1 < y2"):
        handle_command("infer", payload)


def test_handle_command_infer_creates_output_dir_and_writes_mask_png(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "rgb.jpg"
    image_path.write_text("stub image")
    output_dir = tmp_path / "sam3_outputs" / "req-1"
    calls: dict[str, object] = {}

    class FakeModel:
        def predict(self, *, source, bboxes, verbose, save, multimask_output=False):
            calls["source"] = source
            calls["bboxes"] = bboxes
            calls["verbose"] = verbose
            calls["save"] = save
            calls["multimask_output"] = multimask_output
            return [
                type(
                    "FakeResult",
                    (),
                    {
                        "masks": type("FakeMasks", (), {"data": [[[0, 1], [1, 1]]]}),
                        "boxes": type(
                            "FakeBoxes",
                            (),
                            {"xyxy": [[380, 459, 430, 521]]},
                        ),
                    },
                )()
            ]

    monkeypatch.setattr(service_module, "_MODEL", FakeModel())

    response = handle_command(
        "infer",
        {
            "image_path": str(image_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "bboxes": [
                {"label": "red_cube_0", "bbox_2d": [379, 458, 431, 522]},
                {"label": "blue_cube_0", "bbox_2d": [301, 365, 353, 427]},
            ],
        },
    )

    assert calls == {
        "source": str(image_path.resolve()),
        "bboxes": [
            [379.0, 458.0, 431.0, 522.0],
            [301.0, 365.0, 353.0, 427.0],
        ],
        "verbose": False,
        "save": False,
        "multimask_output": False,
    }
    assert output_dir.is_dir()
    assert response["worker"] == "sam3"
    assert response["prompt_mode"] == "bbox"
    assert response["image_path"] == str(image_path.resolve())
    assert response["output_dir"] == str(output_dir.resolve())
    assert len(response["results"]) == 2

    first, second = response["results"]
    assert first["label"] == "red_cube_0"
    assert first["prompt_bbox_2d"] == [379.0, 458.0, 431.0, 522.0]
    assert first["found"] is True
    assert first["bbox_2d"] == [380.0, 459.0, 430.0, 521.0]
    assert first["mask_path"] is not None

    mask_path = Path(first["mask_path"])
    assert mask_path.is_file()
    assert mask_path.parent == output_dir.resolve()
    assert mask_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    assert second == {
        "label": "blue_cube_0",
        "prompt_bbox_2d": [301.0, 365.0, 353.0, 427.0],
        "found": False,
        "bbox_2d": None,
        "mask_path": None,
    }
