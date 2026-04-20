from __future__ import annotations

import logging
import os
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from sam3dworker import Sam3dWorkerClient
from sam3worker import Sam3WorkerClient, Sam3WorkerCommandError

from .geometry import CameraObbResult, estimate_masked_camera_obb
from .models import (
    ArtifactPathsModel,
    AxisConventionModel,
    Object3DModel,
    ObjectResultModel,
    ObjectTimingModel,
    PoseCameraModel,
    ReconstructObjectsRequest,
    ReconstructObjectsResponse,
    RequestTimingModel,
    ResponseCameraModel,
    SegmentationResultModel,
)


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON_EXECUTABLE = Path("/root/autodl-tmp/conda/envs/sam3d-objects/bin/python")
DEFAULT_RUN_ROOT = REPO_ROOT / "sam_pipeline_api" / "runs"
DEFAULT_SOCKET_DIR = REPO_ROOT / "sam_pipeline_api" / "sockets"
DEFAULT_TRACE_DIR = REPO_ROOT / "sam_pipeline_api" / "traces"


class PipelineInputError(ValueError):
    pass


@dataclass(frozen=True)
class PipelinePaths:
    run_root: Path
    socket_dir: Path
    trace_dir: Path


class SamPipelineService:
    def __init__(
        self,
        *,
        python_executable: Path,
        paths: PipelinePaths,
        startup_timeout_s: float,
        default_request_timeout_s: float,
    ) -> None:
        self._python_executable = python_executable
        self._paths = paths
        self._startup_timeout_s = startup_timeout_s
        self._default_request_timeout_s = default_request_timeout_s
        self._lock = threading.Lock()
        self._sam3_client: Sam3WorkerClient | None = None
        self._sam3d_client: Sam3dWorkerClient | None = None

    @classmethod
    def from_env(cls) -> "SamPipelineService":
        python_executable = Path(
            os.environ.get("SAM_PIPELINE_PYTHON", str(DEFAULT_PYTHON_EXECUTABLE))
        ).expanduser()
        startup_timeout_s = float(os.environ.get("SAM_PIPELINE_STARTUP_TIMEOUT_S", "180.0"))
        default_request_timeout_s = float(
            os.environ.get("SAM_PIPELINE_DEFAULT_REQUEST_TIMEOUT_S", "600.0")
        )
        run_root = Path(os.environ.get("SAM_PIPELINE_RUN_ROOT", str(DEFAULT_RUN_ROOT))).expanduser()
        socket_dir = Path(
            os.environ.get("SAM_PIPELINE_SOCKET_DIR", str(DEFAULT_SOCKET_DIR))
        ).expanduser()
        trace_dir = Path(os.environ.get("SAM_PIPELINE_TRACE_DIR", str(DEFAULT_TRACE_DIR))).expanduser()
        return cls(
            python_executable=python_executable,
            paths=PipelinePaths(
                run_root=run_root.resolve(),
                socket_dir=socket_dir.resolve(),
                trace_dir=trace_dir.resolve(),
            ),
            startup_timeout_s=startup_timeout_s,
            default_request_timeout_s=default_request_timeout_s,
        )

    def start(self) -> None:
        self._paths.run_root.mkdir(parents=True, exist_ok=True)
        self._paths.socket_dir.mkdir(parents=True, exist_ok=True)
        self._paths.trace_dir.mkdir(parents=True, exist_ok=True)

        sam3_socket_path = self._paths.socket_dir / "sam3.sock"
        sam3d_socket_path = self._paths.socket_dir / "sam3d.sock"
        _unlink_if_exists(sam3_socket_path)
        _unlink_if_exists(sam3d_socket_path)

        self._sam3_client = Sam3WorkerClient(
            socket_path=sam3_socket_path,
            python_executable=self._python_executable,
            startup_timeout=self._startup_timeout_s,
            cwd=REPO_ROOT,
            trace_path=self._paths.trace_dir / "sam3-client-trace.jsonl",
        )
        self._sam3d_client = Sam3dWorkerClient(
            socket_path=sam3d_socket_path,
            python_executable=self._python_executable,
            startup_timeout=self._startup_timeout_s,
            cwd=REPO_ROOT,
            trace_path=self._paths.trace_dir / "sam3d-client-trace.jsonl",
        )

        try:
            self._sam3_client.start()
            self._sam3d_client.start()
            LOGGER.info("sam3 worker metadata: %s", self._sam3_client.describe(timeout=30.0))
            LOGGER.info("sam3d worker metadata: %s", self._sam3d_client.describe(timeout=30.0))
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._sam3d_client is not None:
            self._sam3d_client.stop()
            self._sam3d_client = None
        if self._sam3_client is not None:
            self._sam3_client.stop()
            self._sam3_client = None

    def reconstruct_objects(self, request: ReconstructObjectsRequest) -> ReconstructObjectsResponse:
        with self._lock:
            return self._reconstruct_objects_locked(request)

    def _reconstruct_objects_locked(
        self,
        request: ReconstructObjectsRequest,
    ) -> ReconstructObjectsResponse:
        sam3_client = self._require_sam3_client()
        sam3d_client = self._require_sam3d_client()

        if request.camera.depth_image.unit.lower() not in {"meter", "meters", "m"}:
            raise PipelineInputError(
                f"depth_image.unit must be meter; got {request.camera.depth_image.unit!r}"
            )

        request_id = request.request_id or _build_request_id()
        request_root = _resolve_request_root(
            request_id=request_id,
            explicit_output_root=request.output_root,
            default_run_root=self._paths.run_root,
        )

        inputs_dir = request_root / "inputs"
        sam3_output_dir = request_root / "sam3"
        objects_root = request_root / "objects"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        sam3_output_dir.mkdir(parents=True, exist_ok=True)
        objects_root.mkdir(parents=True, exist_ok=True)

        total_start = time.perf_counter()
        download_started_at = time.perf_counter()
        image_path = _download_artifact(
            request.camera.rgb_image.ref.download_url,
            destination_path=inputs_dir / _build_artifact_filename("rgb", request.camera.rgb_image.ref.content_type),
        )
        depth_path = _download_artifact(
            request.camera.depth_image.ref.download_url,
            destination_path=inputs_dir
            / _build_artifact_filename("depth", request.camera.depth_image.ref.content_type),
        )
        download_inputs_ms = (time.perf_counter() - download_started_at) * 1000.0

        sam3_response = sam3_client.infer(
            image_path=image_path,
            output_dir=sam3_output_dir,
            bboxes=[
                {
                    "label": target.label,
                    "bbox_2d": target.bbox_2d,
                }
                for target in request.bboxes
            ],
            request_id=f"{request_id}-sam3",
            timeout=request.sam3_timeout_s,
        )

        sam3_results = sam3_response.get("results", [])
        if len(sam3_results) != len(request.bboxes):
            raise Sam3WorkerCommandError(
                f"sam3 returned {len(sam3_results)} results for {len(request.bboxes)} bboxes"
            )

        objects: list[ObjectResultModel] = []
        errors: list[str] = []
        sam3d_total_inference_ms = 0.0
        obb_total_estimation_ms = 0.0

        for target, sam3_result in zip(request.bboxes, sam3_results, strict=True):
            segmentation = SegmentationResultModel(
                prompt_bbox_2d=list(target.bbox_2d),
                found=bool(sam3_result.get("found", False)),
                bbox_2d=_optional_int_list(sam3_result.get("bbox_2d")),
                mask_path=_optional_string(sam3_result.get("mask_path")),
            )
            timing = ObjectTimingModel(
                sam3_avg_inference_ms=_optional_float(sam3_result.get("avg_inference_ms")),
            )

            if not segmentation.found or segmentation.mask_path is None:
                error_message = (
                    f"SAM3 did not produce a mask for label={target.label} from bbox={target.bbox_2d}"
                )
                objects.append(
                    ObjectResultModel(
                        label=target.label,
                        status="not_found",
                        segmentation=segmentation,
                        artifacts=ArtifactPathsModel(),
                        timing=timing,
                        error=error_message,
                    )
                )
                errors.append(error_message)
                continue

            object_output_dir = objects_root / _safe_fragment(target.label)
            object_output_dir.mkdir(parents=True, exist_ok=True)

            pose_camera: PoseCameraModel | None = None
            object_3d: Object3DModel | None = None
            artifacts = ArtifactPathsModel()
            object_status = "success"
            object_error: str | None = None

            try:
                sam3d_response = sam3d_client.reconstruct(
                    image_path=image_path,
                    depth_path=depth_path,
                    mask_path=segmentation.mask_path,
                    output_dir=object_output_dir,
                    fx=request.camera.intrinsics.fx,
                    fy=request.camera.intrinsics.fy,
                    cx=request.camera.intrinsics.cx,
                    cy=request.camera.intrinsics.cy,
                    label=target.label,
                    artifact_types=list(request.artifact_types),
                    request_id=f"{request_id}-sam3d-{_safe_fragment(target.label)}",
                    timeout=request.sam3d_timeout_s or self._default_request_timeout_s,
                )
                pose = sam3d_response["pose"]
                pose_camera = PoseCameraModel(
                    rotation_quaternion_wxyz=[float(value) for value in pose["rotation"]],
                    translation_xyz_m=[float(value) for value in pose["translation"]],
                    scale_xyz_m=[float(value) for value in pose["scale"]],
                )
                timing.sam3d_inference_ms = _optional_float(sam3d_response.get("model_inference_ms"))
                if timing.sam3d_inference_ms is not None:
                    sam3d_total_inference_ms += timing.sam3d_inference_ms
                artifacts = ArtifactPathsModel(
                    pointmap_path=_optional_string(sam3d_response.get("pointmap_path")),
                    mesh_glb_path=_optional_string(
                        sam3d_response.get("artifacts", {}).get("mesh_glb_path")
                    ),
                    gaussian_ply_path=_optional_string(
                        sam3d_response.get("artifacts", {}).get("gaussian_ply_path")
                    ),
                )
            except Exception as exc:
                object_status = "error"
                object_error = str(exc)
                errors.append(f"{target.label}: {object_error}")
                objects.append(
                    ObjectResultModel(
                        label=target.label,
                        status=object_status,
                        segmentation=segmentation,
                        object_3d=object_3d,
                        pose_camera=pose_camera,
                        artifacts=artifacts,
                        timing=timing,
                        error=object_error,
                    )
                )
                continue

            try:
                obb_started_at = time.perf_counter()
                obb_result = estimate_masked_camera_obb(
                    depth_path=depth_path,
                    mask_path=Path(segmentation.mask_path),
                    fx=request.camera.intrinsics.fx,
                    fy=request.camera.intrinsics.fy,
                    cx=request.camera.intrinsics.cx,
                    cy=request.camera.intrinsics.cy,
                )
                timing.obb_estimation_ms = (time.perf_counter() - obb_started_at) * 1000.0
                obb_total_estimation_ms += timing.obb_estimation_ms
                object_3d = _build_object_3d_model(obb_result)
            except Exception as exc:
                object_status = "partial_success"
                object_error = f"sam3d succeeded but object_3d estimation failed: {exc}"
                errors.append(f"{target.label}: {object_error}")

            objects.append(
                ObjectResultModel(
                    label=target.label,
                    status=object_status,
                    segmentation=segmentation,
                    object_3d=object_3d,
                    pose_camera=pose_camera,
                    artifacts=artifacts,
                    timing=timing,
                    error=object_error,
                )
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return ReconstructObjectsResponse(
            request_id=request_id,
            task=request.task,
            status=_summarize_request_status(objects),
            output_root=str(request_root),
            camera=ResponseCameraModel(
                id=request.camera.id,
                coordinate_frame="camera",
                axis_convention=AxisConventionModel(x="left", y="up", z="forward"),
            ),
            timing=RequestTimingModel(
                total_ms=total_ms,
                download_inputs_ms=download_inputs_ms,
                sam3_batch_inference_ms=_optional_float(sam3_response.get("batch_model_inference_ms")),
                sam3d_total_inference_ms=sam3d_total_inference_ms,
                obb_total_estimation_ms=obb_total_estimation_ms,
            ),
            objects=objects,
            errors=errors,
        )

    def _require_sam3_client(self) -> Sam3WorkerClient:
        if self._sam3_client is None:
            raise RuntimeError("sam3 worker client is not started")
        return self._sam3_client

    def _require_sam3d_client(self) -> Sam3dWorkerClient:
        if self._sam3d_client is None:
            raise RuntimeError("sam3d worker client is not started")
        return self._sam3d_client


def _build_object_3d_model(obb_result: CameraObbResult) -> Object3DModel:
    return Object3DModel(
        source="visible_depth_pca3d_obb",
        position_xyz_m=_vector_to_float_list(obb_result.center_xyz_m),
        rotation_quaternion_wxyz=_vector_to_float_list(obb_result.rotation_quaternion_wxyz),
        rotation_matrix_camera_from_obb=_matrix_to_float_lists(
            obb_result.rotation_matrix_camera_from_obb
        ),
        size_xyz_m=_vector_to_float_list(obb_result.size_xyz_m),
        obb_corners_xyz_m=_matrix_to_float_lists(obb_result.corners_xyz_m),
        visible_point_centroid_xyz_m=_vector_to_float_list(obb_result.visible_point_centroid_xyz_m),
        visible_point_count=obb_result.visible_point_count,
    )


def _build_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:12]}"


def _resolve_request_root(
    *,
    request_id: str,
    explicit_output_root: str | None,
    default_run_root: Path,
) -> Path:
    if explicit_output_root is not None:
        output_root = Path(explicit_output_root).expanduser()
        if not output_root.is_absolute():
            output_root = (Path.cwd() / output_root).resolve()
        else:
            output_root = output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root

    safe_request_id = _safe_fragment(request_id)
    candidate = default_run_root / safe_request_id
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = default_run_root / f"{safe_request_id}-{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _download_artifact(download_url: str, *, destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(download_url)
    if parsed.scheme in {"", None}:
        source_path = Path(download_url).expanduser()
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve()
        else:
            source_path = source_path.resolve()
        if not source_path.is_file():
            raise PipelineInputError(f"artifact source does not exist: {source_path}")
        destination_path.write_bytes(source_path.read_bytes())
        return destination_path

    try:
        with urllib.request.urlopen(download_url) as response:
            destination_path.write_bytes(response.read())
    except Exception as exc:
        raise PipelineInputError(f"failed to download artifact from {download_url}: {exc}") from exc
    return destination_path


def _build_artifact_filename(prefix: str, content_type: str) -> str:
    normalized = content_type.lower().strip()
    suffix = ".bin"
    if normalized == "image/png":
        suffix = ".png"
    elif normalized in {"image/jpeg", "image/jpg"}:
        suffix = ".jpg"
    elif normalized == "application/x-npy":
        suffix = ".npy"
    return f"{prefix}{suffix}"


def _summarize_request_status(objects: list[ObjectResultModel]) -> str:
    statuses = {item.status for item in objects}
    if statuses == {"success"}:
        return "success"
    if "success" in statuses or "partial_success" in statuses:
        return "partial_success"
    return "error"


def _safe_fragment(value: str) -> str:
    safe_chars: list[str] = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe_value = "".join(safe_chars).strip("._")
    return safe_value or "item"


def _unlink_if_exists(path: Path) -> None:
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
    except FileNotFoundError:
        return


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int_list(value: object) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("expected a list of integers")
    return [int(item) for item in value]


def _vector_to_float_list(vector: object) -> list[float]:
    return [float(item) for item in list(vector)]


def _matrix_to_float_lists(matrix: object) -> list[list[float]]:
    return [[float(item) for item in list(row)] for row in list(matrix)]
