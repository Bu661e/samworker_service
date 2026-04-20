from __future__ import annotations

import re
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SAM3_WEIGHT_PATH = "/root/sam3.pt"
SUPPORTED_COMMANDS = ["ping", "describe", "infer"]
PROMPT_MODES = ["bbox"]
_MODEL: object | None = None


@dataclass(frozen=True)
class BBoxPrompt:
    label: str
    bbox_2d: tuple[float, float, float, float]


@dataclass(frozen=True)
class InferRequest:
    image_path: Path
    output_dir: Path
    bboxes: list[BBoxPrompt]


@dataclass(frozen=True)
class ModelPrediction:
    found: bool
    bbox_2d: tuple[float, float, float, float] | None
    mask_rows: tuple[tuple[int, ...], ...] | None


def describe_service() -> dict[str, Any]:
    return {
        "worker": "sam3",
        "status": "ready" if _MODEL is not None else "not_loaded",
        "weight_path": SAM3_WEIGHT_PATH,
        "supported_commands": SUPPORTED_COMMANDS,
        "prompt_modes": PROMPT_MODES,
        "model_loaded": _MODEL is not None,
    }


def handle_command(command: str, payload: dict[str, Any]) -> dict[str, Any]:
    if command == "ping":
        return {"status": "ready"}
    if command == "describe":
        return describe_service()
    if command == "infer":
        return infer(payload)
    raise ValueError(f"unknown command: {command}")


def initialize_model() -> object:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    weight_path = Path(SAM3_WEIGHT_PATH)
    if not weight_path.exists():
        raise ValueError(f"SAM3 weight file does not exist: {weight_path}")

    _MODEL = _construct_model(weight_path)
    return _MODEL


def infer(payload: dict[str, Any]) -> dict[str, Any]:
    request = _parse_infer_request(payload)

    try:
        request.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"failed to create output_dir: {request.output_dir}") from exc

    predictions, model_inference_ms = _run_sam_inference(request)
    if len(predictions) != len(request.bboxes):
        raise ValueError(
            f"sam3 returned {len(predictions)} results for {len(request.bboxes)} bbox prompts"
        )
    avg_inference_ms = model_inference_ms / len(request.bboxes)

    results: list[dict[str, Any]] = []
    for index, (prompt, prediction) in enumerate(zip(request.bboxes, predictions)):
        item: dict[str, Any] = {
            "label": prompt.label,
            "prompt_bbox_2d": list(prompt.bbox_2d),
            "found": prediction.found,
            "bbox_2d": None,
            "mask_path": None,
            "avg_inference_ms": avg_inference_ms,
        }

        if prediction.found:
            if prediction.mask_rows is None:
                raise ValueError("sam3 returned a found result without mask_rows")

            bbox_2d = prediction.bbox_2d or _bbox_from_mask_rows(prediction.mask_rows)
            mask_path = request.output_dir / _build_mask_filename(index, prompt.label)

            try:
                _write_grayscale_png(mask_path, prediction.mask_rows)
            except OSError as exc:
                raise ValueError(f"failed to write mask file: {mask_path}") from exc

            item["bbox_2d"] = list(bbox_2d)
            item["mask_path"] = str(mask_path.resolve())

        results.append(item)

    return {
        "worker": "sam3",
        "prompt_mode": "bbox",
        "image_path": str(request.image_path),
        "output_dir": str(request.output_dir),
        "batch_model_inference_ms": model_inference_ms,
        "results": results,
    }


def _parse_infer_request(payload: dict[str, Any]) -> InferRequest:
    image_path_value = payload.get("image_path")
    if not isinstance(image_path_value, str) or not image_path_value.strip():
        raise ValueError("image_path must be a non-empty string")

    image_path = Path(image_path_value)
    if not image_path.is_absolute():
        raise ValueError("image_path must be an absolute path")
    if not image_path.exists() or not image_path.is_file():
        raise ValueError(f"image_path does not exist: {image_path}")

    output_dir_value = payload.get("output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value.strip():
        raise ValueError("output_dir must be a non-empty string")

    output_dir = Path(output_dir_value)
    if not output_dir.is_absolute():
        raise ValueError("output_dir must be an absolute path")

    bboxes_value = payload.get("bboxes")
    if not isinstance(bboxes_value, list) or not bboxes_value:
        raise ValueError("bboxes must be a non-empty list")

    labels: set[str] = set()
    prompts: list[BBoxPrompt] = []
    for index, entry in enumerate(bboxes_value):
        if not isinstance(entry, dict):
            raise ValueError(f"bboxes[{index}] must be a JSON object")

        label = entry.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"bboxes[{index}].label must be a non-empty string")
        if label in labels:
            raise ValueError(f"duplicate bbox label: {label}")
        labels.add(label)

        bbox_value = entry.get("bbox_2d")
        if not isinstance(bbox_value, list) or len(bbox_value) != 4:
            raise ValueError(f"bboxes[{index}].bbox_2d must be a list of 4 numbers")

        if not all(_is_number(coord) for coord in bbox_value):
            raise ValueError(f"bboxes[{index}].bbox_2d must contain only numbers")

        x1, y1, x2, y2 = (float(coord) for coord in bbox_value)
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"bboxes[{index}].bbox_2d must satisfy x1 < x2 and y1 < y2")

        prompts.append(BBoxPrompt(label=label, bbox_2d=(x1, y1, x2, y2)))

    return InferRequest(
        image_path=image_path,
        output_dir=output_dir,
        bboxes=prompts,
    )


def _run_sam_inference(request: InferRequest) -> tuple[list[ModelPrediction], float]:
    model = _require_loaded_model()
    prompts = [list(prompt.bbox_2d) for prompt in request.bboxes]

    raw_results: object
    start = time.perf_counter()
    try:
        try:
            raw_results = model.predict(
                source=str(request.image_path),
                bboxes=prompts,
                verbose=False,
                save=False,
                multimask_output=False,
            )
        except Exception as exc:
            if not _should_retry_without_multimask_output(exc):
                raise
            raw_results = model.predict(
                source=str(request.image_path),
                bboxes=prompts,
                verbose=False,
                save=False,
            )
    finally:
        model_inference_ms = (time.perf_counter() - start) * 1000.0

    return _normalize_sam_results(raw_results, len(request.bboxes)), model_inference_ms


def _should_retry_without_multimask_output(exc: Exception) -> bool:
    if isinstance(exc, TypeError):
        return True
    return "multimask_output" in str(exc)


def _construct_model(weight_path: Path) -> object:
    try:
        from ultralytics import SAM
    except Exception as exc:
        raise ValueError("ultralytics is not installed in the active Python environment") from exc

    try:
        return SAM(str(weight_path))
    except Exception as exc:
        raise ValueError(f"failed to load SAM3 model from {weight_path}") from exc


def _require_loaded_model() -> object:
    if _MODEL is None:
        raise ValueError("sam3 model is not loaded")
    return _MODEL


def _normalize_sam_results(raw_results: object, expected_count: int) -> list[ModelPrediction]:
    if not isinstance(raw_results, Sequence) or len(raw_results) == 0:
        return [_not_found_prediction() for _ in range(expected_count)]

    result = raw_results[0]
    masks_data = _extract_tensor_data(getattr(getattr(result, "masks", None), "data", None))
    boxes_data = _extract_tensor_data(getattr(getattr(result, "boxes", None), "xyxy", None))

    masks = _normalize_mask_batch(masks_data)
    boxes = _normalize_box_batch(boxes_data)

    predictions: list[ModelPrediction] = []
    for index in range(expected_count):
        if index >= len(masks):
            predictions.append(_not_found_prediction())
            continue

        mask_rows = _normalize_mask_rows(masks[index])
        bbox_2d = boxes[index] if index < len(boxes) else _bbox_from_mask_rows(mask_rows)
        predictions.append(
            ModelPrediction(
                found=True,
                bbox_2d=bbox_2d,
                mask_rows=mask_rows,
            )
        )

    return predictions


def _extract_tensor_data(value: object) -> object:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _normalize_mask_batch(value: object) -> list[list[list[object]]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("sam3 mask output has unsupported type")
    if not value:
        return []
    first = value[0]
    if not isinstance(first, list):
        raise ValueError("sam3 mask output has unsupported shape")
    if first and _is_number(first[0]):
        return [value]
    return value


def _normalize_box_batch(value: object) -> list[tuple[float, float, float, float]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("sam3 bbox output has unsupported type")
    if not value:
        return []
    first = value[0]
    if isinstance(first, list):
        rows = value
    else:
        rows = [value]

    boxes: list[tuple[float, float, float, float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 4 or not all(_is_number(item) for item in row):
            raise ValueError("sam3 bbox output has unsupported shape")
        boxes.append(tuple(float(item) for item in row))
    return boxes


def _normalize_mask_rows(value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("sam3 returned an empty mask")

    width: int | None = None
    rows: list[tuple[int, ...]] = []
    for row in value:
        if not isinstance(row, list) or not row:
            raise ValueError("sam3 returned a malformed mask row")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("sam3 returned a non-rectangular mask")
        rows.append(tuple(255 if float(pixel) > 0 else 0 for pixel in row))

    return tuple(rows)


def _bbox_from_mask_rows(mask_rows: Iterable[Iterable[int]]) -> tuple[float, float, float, float]:
    min_x: int | None = None
    min_y: int | None = None
    max_x: int | None = None
    max_y: int | None = None

    for y, row in enumerate(mask_rows):
        for x, pixel in enumerate(row):
            if pixel <= 0:
                continue
            min_x = x if min_x is None else min(min_x, x)
            min_y = y if min_y is None else min(min_y, y)
            max_x = x if max_x is None else max(max_x, x)
            max_y = y if max_y is None else max(max_y, y)

    if min_x is None or min_y is None or max_x is None or max_y is None:
        raise ValueError("sam3 returned a positive result with an empty mask")

    return (float(min_x), float(min_y), float(max_x + 1), float(max_y + 1))


def _write_grayscale_png(path: Path, rows: Sequence[Sequence[int]]) -> None:
    if not rows or not rows[0]:
        raise ValueError("mask_rows must be a non-empty 2D array")

    width = len(rows[0])
    height = len(rows)
    raw = bytearray()
    for row in rows:
        if len(row) != width:
            raise ValueError("mask_rows must be rectangular")
        raw.append(0)
        raw.extend(int(pixel) & 0xFF for pixel in row)

    payload = [
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0),
        ),
        _png_chunk(b"IDAT", zlib.compress(bytes(raw))),
        _png_chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(payload))


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _build_mask_filename(index: int, label: str) -> str:
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")
    if not safe_label:
        safe_label = f"bbox_{index}"
    return f"{index:03d}_{safe_label}.png"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _not_found_prediction() -> ModelPrediction:
    return ModelPrediction(found=False, bbox_2d=None, mask_rows=None)
