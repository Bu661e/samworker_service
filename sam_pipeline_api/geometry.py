from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CameraObbResult:
    mode: str
    center_xyz_m: np.ndarray
    rotation_quaternion_wxyz: np.ndarray
    rotation_matrix_camera_from_obb: np.ndarray
    size_xyz_m: np.ndarray
    corners_xyz_m: np.ndarray
    visible_point_centroid_xyz_m: np.ndarray
    visible_point_count: int


def estimate_masked_camera_obb(
    *,
    depth_path: Path,
    mask_path: Path,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> CameraObbResult:
    points_camera_xyz = load_masked_camera_points(
        depth_path=depth_path,
        mask_path=mask_path,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    rotation_matrix_camera_from_obb = _estimate_pca3d_rotation(points_camera_xyz)

    centroid = points_camera_xyz.mean(axis=0)
    centered_points = points_camera_xyz - centroid
    local_coords = centered_points @ rotation_matrix_camera_from_obb
    local_min = local_coords.min(axis=0)
    local_max = local_coords.max(axis=0)
    center_local = (local_min + local_max) / 2.0
    size_xyz_m = local_max - local_min

    center_xyz_m = centroid + center_local @ rotation_matrix_camera_from_obb.T
    corners_local = _build_obb_corners_local(center_local, size_xyz_m)
    corners_xyz_m = centroid + corners_local @ rotation_matrix_camera_from_obb.T

    return CameraObbResult(
        mode="pca3d",
        center_xyz_m=center_xyz_m,
        rotation_quaternion_wxyz=_rotation_matrix_to_quaternion_wxyz(rotation_matrix_camera_from_obb),
        rotation_matrix_camera_from_obb=rotation_matrix_camera_from_obb,
        size_xyz_m=size_xyz_m,
        corners_xyz_m=corners_xyz_m,
        visible_point_centroid_xyz_m=centroid,
        visible_point_count=int(points_camera_xyz.shape[0]),
    )


def load_masked_camera_points(
    *,
    depth_path: Path,
    mask_path: Path,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    depth = np.load(depth_path)
    if depth.ndim != 2:
        raise ValueError("depth array must have shape H x W")

    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) > 0

    if mask.shape != depth.shape:
        raise ValueError("mask and depth must have identical H x W dimensions")

    valid = mask & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("mask does not contain any valid depth pixels")

    ys, xs = np.where(valid)
    z = depth[ys, xs].astype(np.float64, copy=False)
    x_right = (xs.astype(np.float64) - cx) * z / fx
    y_down = (ys.astype(np.float64) - cy) * z / fy

    # Match sam3dworker's external pointmap convention: [x_left, y_up, z_forward].
    return np.stack((-x_right, -y_down, z), axis=1)


def _estimate_pca3d_rotation(points_xyz: np.ndarray) -> np.ndarray:
    if points_xyz.shape[0] < 3:
        return np.eye(3, dtype=np.float64)

    centered = points_xyz - points_xyz.mean(axis=0)
    covariance = centered.T @ centered / max(points_xyz.shape[0], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    rotation_matrix = eigenvectors[:, order]
    if np.linalg.det(rotation_matrix) < 0.0:
        rotation_matrix[:, 2] *= -1.0
    return rotation_matrix


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
