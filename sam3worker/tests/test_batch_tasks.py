from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from sam3worker.batch_tasks import save_combined_mask_png


def _write_mask(path: Path, rows: list[list[int]]) -> None:
    Image.fromarray(np.asarray(rows, dtype=np.uint8), mode="L").save(path, format="PNG")


def test_save_combined_mask_png_writes_white_union_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8), mode="RGB").save(image_path, format="PNG")

    mask_a = tmp_path / "mask-a.png"
    mask_b = tmp_path / "mask-b.png"
    _write_mask(mask_a, [[255, 0, 0], [0, 0, 0], [0, 0, 0]])
    _write_mask(mask_b, [[0, 0, 0], [0, 255, 0], [0, 0, 0]])

    output_path = tmp_path / "combined_masks.png"
    save_combined_mask_png([mask_a, mask_b], output_path, image_path=image_path)

    with Image.open(output_path) as combined_image:
        combined_array = np.asarray(combined_image.convert("L"))

    assert combined_array.shape == (3, 3)
    assert int(combined_array[0, 0]) == 255
    assert int(combined_array[1, 1]) == 255
    assert int(combined_array[2, 2]) == 0
