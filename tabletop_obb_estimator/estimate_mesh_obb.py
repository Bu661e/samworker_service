#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate an oriented bounding box directly from a mesh file such as mesh.glb."
    )
    parser.add_argument("mesh_path", type=Path, help="Path to a mesh file, e.g. mesh.glb")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path",
    )
    args = parser.parse_args()

    result = _estimate_mesh_obb(args.mesh_path.resolve())
    output_text = json.dumps(result, ensure_ascii=True, indent=2) + "\n"

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    else:
        print(output_text, end="")
    return 0


def _estimate_mesh_obb(mesh_path: Path) -> dict[str, Any]:
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError(
            "trimesh is required. Use the sam3d environment or install trimesh in the current Python."
        ) from exc

    scene_or_mesh = trimesh.load(mesh_path, force="scene")
    if isinstance(scene_or_mesh, trimesh.Scene):
        if hasattr(scene_or_mesh, "to_geometry"):
            mesh = scene_or_mesh.to_geometry()
        else:
            mesh = scene_or_mesh.dump(concatenate=True)
    else:
        mesh = scene_or_mesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"failed to resolve a mesh geometry from {mesh_path}")

    obb = mesh.bounding_box_oriented
    transform = np.asarray(obb.primitive.transform, dtype=np.float64)
    extents = np.asarray(obb.primitive.extents, dtype=np.float64)
    center = transform[:3, 3]
    rotation = transform[:3, :3]
    corners = np.asarray(obb.vertices, dtype=np.float64)
    quaternion_wxyz = _rotation_matrix_to_quaternion_wxyz(rotation)

    return {
        "mesh_path": str(mesh_path),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "obb_center_xyz": _vector_to_list(center),
        "obb_extents_xyz": _vector_to_list(extents),
        "obb_rotation_matrix": _matrix_to_lists(rotation),
        "obb_rotation_quaternion_wxyz": _vector_to_list(quaternion_wxyz),
        "obb_transform": _matrix_to_lists(transform),
        "obb_corners_xyz": _matrix_to_lists(corners),
        "notes": [
            "obb_extents_xyz are the three box edge lengths.",
            "If the three extents are almost equal, the box rotation may be numerically unstable or not meaningful.",
        ],
    }


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
