#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Request:
    label: str
    depth_path: Path
    mask_path: Path
    fx: float
    fy: float
    cx: float
    cy: float
    camera_world_position_xyz_m: np.ndarray
    camera_world_quaternion_wxyz: np.ndarray
    expected_world_position_xyz_m: np.ndarray | None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate an IsaacSim world-space object position directly from "
            "mask + depth + intrinsics + camera world pose."
        )
    )
    parser.add_argument("request_json", type=Path, help="Path to a request JSON file")
    parser.add_argument(
        "--estimator",
        choices=("centroid", "median", "bbox_center"),
        default="median",
        help="How to summarize the masked 3D points into a single position",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the result JSON",
    )
    args = parser.parse_args()

    request = _load_request(args.request_json)
    masked_points_worker = _load_masked_points_worker(request)
    position_worker_xyz_m = _estimate_position(masked_points_worker, estimator=args.estimator)
    position_camera_forward_left_up_m = _worker_to_isaac_camera_local(position_worker_xyz_m)
    position_world_xyz_m = _camera_local_to_world(
        position_camera_forward_left_up_m,
        camera_world_position_xyz_m=request.camera_world_position_xyz_m,
        camera_world_quaternion_wxyz=request.camera_world_quaternion_wxyz,
    )

    result: dict[str, Any] = {
        "label": request.label,
        "estimator": args.estimator,
        "position_world_xyz_m": _vector_to_list(position_world_xyz_m),
        "position_camera_forward_left_up_m": _vector_to_list(position_camera_forward_left_up_m),
        "position_camera_worker_xyz_m": _vector_to_list(position_worker_xyz_m),
        "mask_valid_point_count": int(masked_points_worker.shape[0]),
        "camera_world_position_xyz_m": _vector_to_list(request.camera_world_position_xyz_m),
        "camera_world_quaternion_wxyz": _vector_to_list(request.camera_world_quaternion_wxyz),
        "notes": [
            "position_world_xyz_m is computed from masked depth geometry, not from SAM3D pose.",
            "position_camera_worker_xyz_m uses the current repo convention: [x_left, y_up, z_forward].",
            "position_camera_forward_left_up_m is the Isaac camera-local convention used here before world transform.",
        ],
    }
    if request.expected_world_position_xyz_m is not None:
        world_error_xyz_m = position_world_xyz_m - request.expected_world_position_xyz_m
        result["expected_world_position_xyz_m"] = _vector_to_list(request.expected_world_position_xyz_m)
        result["world_position_error_xyz_m"] = _vector_to_list(world_error_xyz_m)
        result["world_position_error_norm_m"] = float(np.linalg.norm(world_error_xyz_m))

    output_text = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    else:
        print(output_text, end="")
    return 0


def _load_request(request_path: Path) -> Request:
    resolved_request_path = request_path.resolve()
    data = json.loads(resolved_request_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"request file must contain a JSON object: {resolved_request_path}")

    label = data.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")

    depth_path = _parse_existing_path(
        data.get("depth_path"),
        field_name="depth_path",
        base_dir=resolved_request_path.parent,
        expected_suffix=".npy",
    )
    mask_path = _parse_existing_path(
        data.get("mask_path"),
        field_name="mask_path",
        base_dir=resolved_request_path.parent,
        expected_suffix=None,
    )

    intrinsics = data.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError("intrinsics must be an object")

    expected_world_position = data.get("expected_world_position_xyz_m")
    expected_world_position_xyz_m = None
    if expected_world_position is not None:
        expected_world_position_xyz_m = _parse_numeric_vector(
            expected_world_position,
            field_name="expected_world_position_xyz_m",
            length=3,
        )

    return Request(
        label=label.strip(),
        depth_path=depth_path,
        mask_path=mask_path,
        fx=_parse_positive_finite_number(intrinsics.get("fx"), "intrinsics.fx"),
        fy=_parse_positive_finite_number(intrinsics.get("fy"), "intrinsics.fy"),
        cx=_parse_finite_number(intrinsics.get("cx"), "intrinsics.cx"),
        cy=_parse_finite_number(intrinsics.get("cy"), "intrinsics.cy"),
        camera_world_position_xyz_m=_parse_numeric_vector(
            data.get("camera_world_position_xyz_m"),
            field_name="camera_world_position_xyz_m",
            length=3,
        ),
        camera_world_quaternion_wxyz=_parse_numeric_vector(
            data.get("camera_world_quaternion_wxyz"),
            field_name="camera_world_quaternion_wxyz",
            length=4,
        ),
        expected_world_position_xyz_m=expected_world_position_xyz_m,
    )


def _parse_existing_path(
    value: object,
    *,
    field_name: str,
    base_dir: Path,
    expected_suffix: str | None,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if expected_suffix is not None and path.suffix.lower() != expected_suffix:
        raise ValueError(f"{field_name} must end with {expected_suffix}")
    if not path.is_file():
        raise ValueError(f"{field_name} does not exist: {path}")
    return path


def _parse_positive_finite_number(value: object, field_name: str) -> float:
    parsed = _parse_finite_number(value, field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be greater than 0")
    return parsed


def _parse_finite_number(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite number")
    return parsed


def _parse_numeric_vector(value: object, *, field_name: str, length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field_name} must be a list of {length} numbers")
    result = np.empty(length, dtype=np.float64)
    for index, item in enumerate(value):
        result[index] = _parse_finite_number(item, f"{field_name}[{index}]")
    return result


def _load_masked_points_worker(request: Request) -> np.ndarray:
    depth = np.load(request.depth_path)
    if depth.dtype != np.float32:
        raise ValueError("depth array must have dtype float32")
    if depth.ndim != 2:
        raise ValueError("depth array must have shape H x W")

    with Image.open(request.mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L")) > 0

    if mask.shape != depth.shape:
        raise ValueError("mask and depth must have identical H x W dimensions")

    valid = mask & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("mask does not contain any valid depth pixels")

    ys, xs = np.where(valid)
    z = depth[ys, xs].astype(np.float64, copy=False)
    x_right = (xs.astype(np.float64) - request.cx) * z / request.fx
    y_down = (ys.astype(np.float64) - request.cy) * z / request.fy

    # Match the existing repo convention used by sam3dworker: [x_left, y_up, z_forward].
    return np.stack((-x_right, -y_down, z), axis=1)


def _estimate_position(masked_points_worker: np.ndarray, *, estimator: str) -> np.ndarray:
    if estimator == "centroid":
        return masked_points_worker.mean(axis=0)
    if estimator == "median":
        return np.median(masked_points_worker, axis=0)
    if estimator == "bbox_center":
        return (masked_points_worker.min(axis=0) + masked_points_worker.max(axis=0)) / 2.0
    raise ValueError(f"unsupported estimator: {estimator}")


def _worker_to_isaac_camera_local(position_worker_xyz_m: np.ndarray) -> np.ndarray:
    # In this repo we treat worker camera coordinates [x_left, y_up, z_forward]
    # as Isaac camera-local [forward, left, up] for the world transform step.
    return np.array(
        [
            position_worker_xyz_m[2],
            position_worker_xyz_m[0],
            position_worker_xyz_m[1],
        ],
        dtype=np.float64,
    )


def _camera_local_to_world(
    position_camera_forward_left_up_m: np.ndarray,
    *,
    camera_world_position_xyz_m: np.ndarray,
    camera_world_quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    rotation = _quaternion_wxyz_to_matrix(camera_world_quaternion_wxyz)
    return camera_world_position_xyz_m + rotation @ position_camera_forward_left_up_m


def _quaternion_wxyz_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion_wxyz))
    if norm <= 0.0:
        raise ValueError("camera_world_quaternion_wxyz must not be the zero quaternion")
    w, x, y, z = quaternion_wxyz / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _vector_to_list(value: np.ndarray) -> list[float]:
    return [float(item) for item in value.tolist()]


if __name__ == "__main__":
    raise SystemExit(main())
