from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_SAM3D_DIR = REPO_ROOT / "third_party" / "SAM3D-object"
SAM3D_CONFIG_PATH = THIRD_PARTY_SAM3D_DIR / "checkpoints" / "hf" / "pipeline.yaml"
SUPPORTED_COMMANDS = ["ping", "describe", "reconstruct"]
SUPPORTED_ARTIFACT_TYPES = {"gaussian", "mesh"}
WARMUP_IMAGE_SIZE = 256
_INFERENCE: object | None = None
_WARMUP_DONE = False


@dataclass(frozen=True)
class ReconstructRequest:
    image_path: Path
    depth_path: Path
    mask_path: Path
    output_dir: Path
    fx: float
    fy: float
    cx: float
    cy: float
    label: str
    artifact_types: list[str]


def describe_service() -> dict[str, Any]:
    return {
        "worker": "sam3d",
        "third_party_dir": str(THIRD_PARTY_SAM3D_DIR),
        "config_path": str(SAM3D_CONFIG_PATH),
        "status": "ready",
        "supported_commands": SUPPORTED_COMMANDS,
        "reconstruct_stage": "full_inference",
        "inference_loaded": _INFERENCE is not None,
    }


def handle_command(command: str, payload: dict[str, Any]) -> dict[str, Any]:
    if command == "ping":
        return {"status": "ready"}
    if command == "describe":
        return describe_service()
    if command == "reconstruct":
        return reconstruct(payload)
    raise ValueError(f"unknown command: {command}")


def initialize_inference() -> object:
    return _get_inference()


def warmup_inference() -> None:
    global _WARMUP_DONE

    if _WARMUP_DONE:
        return

    inference = _get_inference()
    image, mask, pointmap = _build_warmup_inputs(WARMUP_IMAGE_SIZE)

    try:
        import torch
    except Exception as exc:
        raise ValueError("torch is not installed in the active Python environment") from exc

    pointmap_tensor = torch.from_numpy(pointmap.astype(np.float32, copy=False))

    try:
        inference(image, mask, seed=0, pointmap=pointmap_tensor)
    except Exception as exc:
        raise ValueError("sam3d warmup inference failed") from exc

    _WARMUP_DONE = True


def reconstruct(payload: dict[str, Any]) -> dict[str, Any]:
    request = _parse_reconstruct_request(payload)

    try:
        request.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"failed to create output_dir: {request.output_dir}") from exc

    image = _load_rgb_image(request.image_path)
    depth = _load_depth_array(request.depth_path)
    mask = _load_mask_image(request.mask_path)
    _validate_matching_shapes(image=image, depth=depth, mask=mask)

    pointmap = _build_pointmap(
        depth,
        fx=request.fx,
        fy=request.fy,
        cx=request.cx,
        cy=request.cy,
    )
    pointmap_path = request.output_dir / "pointmap.npy"
    np.save(pointmap_path, pointmap.astype(np.float32, copy=False))

    output, model_inference_ms = _run_sam3d_inference(
        request=request,
        image=image,
        mask=mask,
        pointmap=pointmap,
    )

    return _build_reconstruct_response(
        request=request,
        pointmap_path=pointmap_path,
        model_inference_ms=model_inference_ms,
        output=output,
    )


def _parse_reconstruct_request(payload: dict[str, Any]) -> ReconstructRequest:
    image_path = _parse_existing_absolute_path(
        payload.get("image_path"),
        field_name="image_path",
        expected_suffix=None,
    )
    depth_path = _parse_existing_absolute_path(
        payload.get("depth_path"),
        field_name="depth_path",
        expected_suffix=".npy",
    )
    mask_path = _parse_existing_absolute_path(
        payload.get("mask_path"),
        field_name="mask_path",
        expected_suffix=None,
    )
    output_dir = _parse_absolute_output_dir(payload.get("output_dir"))
    fx = _parse_positive_finite_number(payload.get("fx"), "fx")
    fy = _parse_positive_finite_number(payload.get("fy"), "fy")
    cx = _parse_finite_number(payload.get("cx"), "cx")
    cy = _parse_finite_number(payload.get("cy"), "cy")

    label = payload.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")

    artifact_types = _parse_artifact_types(payload.get("artifact_types"))

    return ReconstructRequest(
        image_path=image_path,
        depth_path=depth_path,
        mask_path=mask_path,
        output_dir=output_dir,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        label=label.strip(),
        artifact_types=artifact_types,
    )


def _parse_existing_absolute_path(
    value: object,
    *,
    field_name: str,
    expected_suffix: str | None,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")
    if expected_suffix is not None and path.suffix.lower() != expected_suffix:
        raise ValueError(f"{field_name} must end with {expected_suffix}")
    if not path.exists() or not path.is_file():
        raise ValueError(f"{field_name} does not exist: {path}")
    return path


def _parse_absolute_output_dir(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("output_dir must be a non-empty string")

    output_dir = Path(value)
    if not output_dir.is_absolute():
        raise ValueError("output_dir must be an absolute path")
    return output_dir


def _parse_positive_finite_number(value: object, field_name: str) -> float:
    parsed = _parse_finite_number(value, field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return parsed


def _parse_finite_number(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite number")
    return parsed


def _parse_artifact_types(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("artifact_types must be a list when provided")

    parsed: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"artifact_types[{index}] must be a non-empty string")
        normalized = item.strip()
        if normalized not in SUPPORTED_ARTIFACT_TYPES:
            raise ValueError(
                f"artifact_types[{index}] must be one of {sorted(SUPPORTED_ARTIFACT_TYPES)}"
            )
        if normalized in seen:
            raise ValueError(f"duplicate artifact type: {normalized}")
        seen.add(normalized)
        parsed.append(normalized)
    return parsed


def _load_rgb_image(image_path: Path) -> np.ndarray:
    try:
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"))
    except OSError as exc:
        raise ValueError(f"failed to read image_path: {image_path}") from exc


def _load_mask_image(mask_path: Path) -> np.ndarray:
    try:
        with Image.open(mask_path) as mask:
            mask_array = np.asarray(mask.convert("L"))
    except OSError as exc:
        raise ValueError(f"failed to read mask_path: {mask_path}") from exc
    return (mask_array > 0).astype(np.uint8)


def _load_depth_array(depth_path: Path) -> np.ndarray:
    try:
        depth = np.load(depth_path)
    except OSError as exc:
        raise ValueError(f"failed to read depth_path: {depth_path}") from exc
    except ValueError as exc:
        raise ValueError(f"failed to parse depth_path as numpy array: {depth_path}") from exc

    if depth.dtype != np.float32:
        raise ValueError("depth array must have dtype float32")
    if depth.ndim != 2:
        raise ValueError("depth array must have shape H x W")
    return depth


def _validate_matching_shapes(
    *,
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
) -> None:
    image_hw = image.shape[:2]
    depth_hw = depth.shape[:2]
    mask_hw = mask.shape[:2]
    if image_hw != depth_hw or image_hw != mask_hw:
        raise ValueError("image, depth, and mask dimensions must match exactly")


def _build_pointmap(
    depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    height, width = depth.shape
    u_coords, v_coords = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    z = depth.astype(np.float32, copy=False)
    valid = np.isfinite(z) & (z > 0)

    x = (u_coords - np.float32(cx)) * z / np.float32(fx)
    y = (v_coords - np.float32(cy)) * z / np.float32(fy)

    # SAM3D's external pointmap path expects PyTorch3D-style camera coordinates.
    pointmap = np.stack((-x, -y, z), axis=-1).astype(np.float32, copy=False)
    pointmap[~valid] = np.nan
    return pointmap


def _build_warmup_inputs(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_gradient = np.linspace(0, 255, size, dtype=np.uint8)
    y_gradient = np.linspace(0, 255, size, dtype=np.uint8)[:, None]

    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = x_gradient
    image[..., 1] = y_gradient
    image[..., 2] = 128

    mask = np.zeros((size, size), dtype=np.uint8)
    inset = max(size // 4, 1)
    mask[inset : size - inset, inset : size - inset] = 1

    depth = np.ones((size, size), dtype=np.float32)
    pointmap = _build_pointmap(
        depth,
        fx=float(size),
        fy=float(size),
        cx=(size - 1) / 2.0,
        cy=(size - 1) / 2.0,
    )
    return image, mask, pointmap


def _run_sam3d_inference(
    *,
    request: ReconstructRequest,
    image: np.ndarray,
    mask: np.ndarray,
    pointmap: np.ndarray,
) -> tuple[dict[str, Any], float]:
    inference = _get_inference()

    try:
        import torch
        import time
    except Exception as exc:
        raise ValueError("torch is not installed in the active Python environment") from exc

    pointmap_tensor = torch.from_numpy(pointmap.astype(np.float32, copy=False))

    try:
        start = time.perf_counter()
        output = inference(image, mask, seed=42, pointmap=pointmap_tensor)
    except Exception as exc:
        raise ValueError(f"sam3d inference failed for label {request.label}") from exc
    model_inference_ms = (time.perf_counter() - start) * 1000.0
    return output, model_inference_ms


def _get_inference() -> object:
    global _INFERENCE

    if _INFERENCE is not None:
        return _INFERENCE

    _INFERENCE = _construct_inference(SAM3D_CONFIG_PATH)
    return _INFERENCE


def _construct_inference(config_path: Path) -> object:
    if not config_path.exists():
        raise ValueError(f"SAM3D config file does not exist: {config_path}")

    third_party_path = str(THIRD_PARTY_SAM3D_DIR)
    if third_party_path not in sys.path:
        sys.path.insert(0, third_party_path)

    module_name = "_sam3d_notebook_inference"
    module = sys.modules.get(module_name)
    if module is None:
        inference_path = THIRD_PARTY_SAM3D_DIR / "notebook" / "inference.py"
        spec = importlib.util.spec_from_file_location(module_name, inference_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"failed to load SAM3D inference module from {inference_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    inference_cls = getattr(module, "Inference", None)
    if inference_cls is None:
        raise ValueError("SAM3D inference module does not expose Inference")

    try:
        return inference_cls(str(config_path), compile=False)
    except Exception as exc:
        raise ValueError(f"failed to initialize SAM3D inference from {config_path}") from exc


def _build_reconstruct_response(
    *,
    request: ReconstructRequest,
    pointmap_path: Path,
    model_inference_ms: float,
    output: dict[str, Any],
) -> dict[str, Any]:
    pose = {
        "rotation": _serialize_numeric_vector(output.get("rotation"), length=4, field_name="rotation"),
        "translation": _serialize_numeric_vector(
            output.get("translation"), length=3, field_name="translation"
        ),
        "scale": _serialize_numeric_vector(output.get("scale"), length=3, field_name="scale"),
    }

    artifacts: dict[str, str] = {}
    if "gaussian" in request.artifact_types:
        artifacts["gaussian_ply_path"] = str(
            _export_gaussian(output, request.output_dir / "splat.ply")
        )
    if "mesh" in request.artifact_types:
        artifacts["mesh_glb_path"] = str(_export_mesh_glb(output, request.output_dir / "mesh.glb"))

    return {
        "worker": "sam3d",
        "label": request.label,
        "image_path": str(request.image_path),
        "depth_path": str(request.depth_path),
        "mask_path": str(request.mask_path),
        "output_dir": str(request.output_dir),
        "pointmap_path": str(pointmap_path.resolve()),
        "model_inference_ms": model_inference_ms,
        "pose": pose,
        "artifacts": artifacts,
    }


def _serialize_numeric_vector(value: object, *, length: int, field_name: str) -> list[float]:
    flattened = _flatten_singleton_leading_dims(_to_serializable(value))
    if not isinstance(flattened, list) or len(flattened) != length:
        raise ValueError(f"sam3d output field {field_name} must be a list of {length} numbers")

    result: list[float] = []
    for item in flattened:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"sam3d output field {field_name} must contain only numbers")
        result.append(float(item))
    return result


def _to_serializable(value: object) -> object:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _flatten_singleton_leading_dims(value: object) -> object:
    while (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    return value


def _export_gaussian(output: dict[str, Any], output_path: Path) -> Path:
    gs = output.get("gs")
    if gs is None or not hasattr(gs, "save_ply"):
        raise ValueError("sam3d output missing gs gaussian object")

    try:
        gs.save_ply(str(output_path))
    except Exception as exc:
        raise ValueError(f"failed to export gaussian ply: {output_path}") from exc
    return output_path.resolve()


def _export_mesh_glb(output: dict[str, Any], output_path: Path) -> Path:
    glb = output.get("glb")
    if glb is None or not hasattr(glb, "export"):
        raise ValueError("sam3d output missing glb object")

    try:
        glb.export(str(output_path))
    except Exception as exc:
        raise ValueError(f"failed to export mesh glb: {output_path}") from exc
    return output_path.resolve()
