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
    world_up_axis_xyz: np.ndarray
    expected_world_position_xyz_m: np.ndarray | None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a tabletop object OBB from mask + depth + intrinsics + camera world pose. "
            "Outputs position, rotation, size_xyz, and OBB corners in IsaacSim world coordinates."
        )
    )
    parser.add_argument("request_json", type=Path, help="Path to a request JSON file")
    parser.add_argument(
        "--mode",
        choices=("tabletop", "pca3d"),
        default="tabletop",
        help="OBB estimation mode. 'tabletop' keeps the box upright using world_up_axis_xyz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    request = _load_request(args.request_json)
    points_world_xyz = _load_masked_points_world(request)
    obb = _estimate_obb(points_world_xyz, mode=args.mode, world_up_axis_xyz=request.world_up_axis_xyz)

    result: dict[str, Any] = {
        "label": request.label,
        "mode": args.mode,
        "position_world_xyz_m": _vector_to_list(obb.center_world_xyz_m),
        "rotation_quaternion_wxyz": _vector_to_list(obb.rotation_quaternion_wxyz),
        "rotation_matrix_world_from_obb": _matrix_to_lists(obb.rotation_matrix_world_from_obb),
        "size_xyz_m": _vector_to_list(obb.size_xyz_m),
        "obb_axes_world_xyz": _matrix_to_lists(obb.rotation_matrix_world_from_obb),
        "obb_corners_world_xyz_m": _matrix_to_lists(obb.corners_world_xyz_m),
        "visible_point_centroid_world_xyz_m": _vector_to_list(points_world_xyz.mean(axis=0)),
        "visible_point_count": int(points_world_xyz.shape[0]),
        "camera_world_position_xyz_m": _vector_to_list(request.camera_world_position_xyz_m),
        "camera_world_quaternion_wxyz": _vector_to_list(request.camera_world_quaternion_wxyz),
        "world_up_axis_xyz": _vector_to_list(request.world_up_axis_xyz),
        "notes": [
            "This OBB is estimated from masked visible depth points, not from sam3d pose.",
            "In tabletop mode the box is constrained to stay upright with world_up_axis_xyz.",
            "For symmetric objects such as cubes, yaw can be ambiguous even if size_xyz is stable.",
            "Single-view depth only sees the visible surface; hidden-side thickness can still be biased.",
        ],
    }
    if request.expected_world_position_xyz_m is not None:
        world_position_error_xyz_m = obb.center_world_xyz_m - request.expected_world_position_xyz_m
        result["expected_world_position_xyz_m"] = _vector_to_list(request.expected_world_position_xyz_m)
        result["world_position_error_xyz_m"] = _vector_to_list(world_position_error_xyz_m)
        result["world_position_error_norm_m"] = float(np.linalg.norm(world_position_error_xyz_m))

    output_text = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    else:
        print(output_text, end="")

    return 0


@dataclass(frozen=True)
class OBBResult:
    center_world_xyz_m: np.ndarray
    rotation_matrix_world_from_obb: np.ndarray
    rotation_quaternion_wxyz: np.ndarray
    size_xyz_m: np.ndarray
    corners_world_xyz_m: np.ndarray


def _load_request(request_path: Path) -> Request:
    resolved_request_path = request_path.resolve()
    data = json.loads(resolved_request_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"request file must contain a JSON object: {resolved_request_path}")

    intrinsics = data.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError("intrinsics must be an object")

    expected_world_position_xyz_m = None
    expected_world_position = data.get("expected_world_position_xyz_m")
    if expected_world_position is not None:
        expected_world_position_xyz_m = _parse_numeric_vector(
            expected_world_position,
            field_name="expected_world_position_xyz_m",
            length=3,
        )

    world_up_axis = data.get("world_up_axis_xyz", [0.0, 0.0, 1.0])
    world_up_axis_xyz = _parse_numeric_vector(world_up_axis, field_name="world_up_axis_xyz", length=3)
    up_norm = float(np.linalg.norm(world_up_axis_xyz))
    if up_norm <= 0.0:
        raise ValueError("world_up_axis_xyz must not be the zero vector")
    world_up_axis_xyz = world_up_axis_xyz / up_norm

    return Request(
        label=_parse_non_empty_string(data.get("label"), "label"),
        depth_path=_parse_existing_path(
            data.get("depth_path"),
            field_name="depth_path",
            base_dir=resolved_request_path.parent,
            expected_suffix=".npy",
        ),
        mask_path=_parse_existing_path(
            data.get("mask_path"),
            field_name="mask_path",
            base_dir=resolved_request_path.parent,
            expected_suffix=None,
        ),
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
        world_up_axis_xyz=world_up_axis_xyz,
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


def _parse_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


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
    parsed = np.empty(length, dtype=np.float64)
    for index, item in enumerate(value):
        parsed[index] = _parse_finite_number(item, f"{field_name}[{index}]")
    return parsed


def _load_masked_points_world(request: Request) -> np.ndarray:
    depth = np.load(request.depth_path)
    if depth.dtype != np.float32:
        raise ValueError("depth array must have dtype float32")
    if depth.ndim != 2:
        raise ValueError("depth array must have shape H x W")

    with Image.open(request.mask_path) as image:
        mask = np.asarray(image.convert("L")) > 0

    if mask.shape != depth.shape:
        raise ValueError("mask and depth must have identical H x W dimensions")

    valid = mask & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("mask does not contain any valid depth pixels")

    ys, xs = np.where(valid)
    forward = depth[ys, xs].astype(np.float64, copy=False)
    left = (request.cx - xs.astype(np.float64)) * forward / request.fx
    up = (request.cy - ys.astype(np.float64)) * forward / request.fy
    points_camera_world_axes = np.stack((forward, left, up), axis=1)

    rotation_world_from_camera = _quaternion_wxyz_to_matrix(request.camera_world_quaternion_wxyz)
    return request.camera_world_position_xyz_m + points_camera_world_axes @ rotation_world_from_camera.T


def _estimate_obb(
    points_world_xyz: np.ndarray,
    *,
    mode: str,
    world_up_axis_xyz: np.ndarray,
) -> OBBResult:
    if mode == "tabletop":
        rotation_world_from_obb = _estimate_tabletop_rotation(points_world_xyz, world_up_axis_xyz)
    elif mode == "pca3d":
        rotation_world_from_obb = _estimate_pca3d_rotation(points_world_xyz)
    else:
        raise ValueError(f"unsupported OBB mode: {mode}")

    local_coords = points_world_xyz @ rotation_world_from_obb
    local_min = local_coords.min(axis=0)
    local_max = local_coords.max(axis=0)
    center_local = (local_min + local_max) / 2.0
    size_xyz_m = local_max - local_min

    center_world_xyz_m = center_local @ rotation_world_from_obb.T
    corners_local = _build_obb_corners_local(center_local, size_xyz_m)
    corners_world_xyz_m = corners_local @ rotation_world_from_obb.T
    rotation_quaternion_wxyz = _rotation_matrix_to_quaternion_wxyz(rotation_world_from_obb)

    return OBBResult(
        center_world_xyz_m=center_world_xyz_m,
        rotation_matrix_world_from_obb=rotation_world_from_obb,
        rotation_quaternion_wxyz=rotation_quaternion_wxyz,
        size_xyz_m=size_xyz_m,
        corners_world_xyz_m=corners_world_xyz_m,
    )


def _estimate_tabletop_rotation(points_world_xyz: np.ndarray, world_up_axis_xyz: np.ndarray) -> np.ndarray:
    up_axis = world_up_axis_xyz / np.linalg.norm(world_up_axis_xyz)

    centroid = points_world_xyz.mean(axis=0)
    centered = points_world_xyz - centroid
    planar = centered - np.outer(centered @ up_axis, up_axis)

    covariance = planar.T @ planar / max(points_world_xyz.shape[0], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    primary_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    primary_axis = primary_axis - up_axis * float(primary_axis @ up_axis)

    norm_primary = float(np.linalg.norm(primary_axis))
    if norm_primary <= 1e-8:
        # 点云在桌面平面上的投影退化时，退回到一个固定的水平轴。
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(fallback @ up_axis)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        primary_axis = fallback - up_axis * float(fallback @ up_axis)
        norm_primary = float(np.linalg.norm(primary_axis))

    axis_x = primary_axis / norm_primary
    axis_y = np.cross(up_axis, axis_x)
    axis_y /= np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    axis_z /= np.linalg.norm(axis_z)

    rotation_world_from_obb = np.column_stack((axis_x, axis_y, axis_z))
    if np.linalg.det(rotation_world_from_obb) < 0.0:
        rotation_world_from_obb[:, 1] *= -1.0
    return rotation_world_from_obb


def _estimate_pca3d_rotation(points_world_xyz: np.ndarray) -> np.ndarray:
    centered = points_world_xyz - points_world_xyz.mean(axis=0)
    covariance = centered.T @ centered / max(points_world_xyz.shape[0], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    rotation_world_from_obb = eigenvectors[:, order]
    if np.linalg.det(rotation_world_from_obb) < 0.0:
        rotation_world_from_obb[:, 2] *= -1.0
    return rotation_world_from_obb


def _build_obb_corners_local(center_local: np.ndarray, size_xyz_m: np.ndarray) -> np.ndarray:
    half = size_xyz_m / 2.0
    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return center_local + signs * half


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


def _rotation_matrix_to_quaternion_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    m = rotation_matrix
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    quaternion = np.array([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _vector_to_list(vector: np.ndarray) -> list[float]:
    return [float(item) for item in vector.tolist()]


def _matrix_to_lists(matrix: np.ndarray) -> list[list[float]]:
    return [[float(item) for item in row] for row in matrix.tolist()]


if __name__ == "__main__":
    raise SystemExit(main())
