from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ArtifactRefModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    content_type: str
    download_url: str

    @field_validator("id", "kind", "content_type", "download_url")
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("artifact ref fields must be non-empty strings")
        return text


class ImageRefModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: ArtifactRefModel


class DepthImageRefModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: str
    ref: ArtifactRefModel

    @field_validator("unit")
    @classmethod
    def _validate_unit(cls, value: str) -> str:
        unit = value.strip()
        if not unit:
            raise ValueError("unit must be a non-empty string")
        return unit


class CameraIntrinsicsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fx: float = Field(gt=0.0)
    fy: float = Field(gt=0.0)
    cx: float
    cy: float
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class CameraPoseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_xyz_m: list[float] = Field(min_length=3, max_length=3)
    quaternion_wxyz: list[float] = Field(min_length=4, max_length=4)


class CameraRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: str | None = None
    prim_path: str | None = None
    mount_mode: str | None = None
    resolution: list[int] | None = Field(default=None, min_length=2, max_length=2)
    intrinsics: CameraIntrinsicsModel
    pose: CameraPoseModel | None = None
    rgb_image: ImageRefModel
    depth_image: DepthImageRefModel

    @field_validator("id", "status", "prim_path", "mount_mode")
    @classmethod
    def _validate_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("camera string fields must not be empty")
        return text


class BboxPromptModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    bbox_2d: list[int] = Field(min_length=4, max_length=4)

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        label = value.strip()
        if not label:
            raise ValueError("label must be a non-empty string")
        return label

    @field_validator("bbox_2d")
    @classmethod
    def _validate_bbox(cls, value: list[int]) -> list[int]:
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_2d must satisfy x2 > x1 and y2 > y1")
        return value


class ReconstructObjectsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    task: str
    bboxes: list[BboxPromptModel] = Field(min_length=1)
    camera: CameraRequestModel
    output_root: str | None = None
    sam3_timeout_s: float = Field(default=300.0, gt=0.0)

    @field_validator("request_id", "task", "output_root")
    @classmethod
    def _validate_top_level_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("string fields must not be empty")
        return text

    @model_validator(mode="after")
    def _validate_unique_labels(self) -> "ReconstructObjectsRequest":
        labels = [item.label for item in self.bboxes]
        if len(labels) != len(set(labels)):
            raise ValueError("bboxes[].label must be unique within one request")
        return self


class AxisConventionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: Literal["left"]
    y: Literal["up"]
    z: Literal["forward"]


class ResponseCameraModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    coordinate_frame: Literal["camera"]
    axis_convention: AxisConventionModel


class SegmentationResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_bbox_2d: list[int]
    found: bool
    bbox_2d: list[int] | None = None
    mask_path: str | None = None


class Object3DModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["visible_depth_pca3d_obb"]
    position_xyz_m: list[float]
    rotation_quaternion_wxyz: list[float]
    rotation_matrix_camera_from_obb: list[list[float]]
    size_xyz_m: list[float]
    obb_corners_xyz_m: list[list[float]]
    visible_point_centroid_xyz_m: list[float]
    visible_point_count: int


class ObjectTimingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sam3_avg_inference_ms: float | None = None
    obb_estimation_ms: float | None = None


class ObjectResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    status: Literal["success", "partial_success", "not_found", "error"]
    segmentation: SegmentationResultModel
    object_3d: Object3DModel | None = None
    timing: ObjectTimingModel
    error: str | None = None


class RequestTimingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_ms: float
    download_inputs_ms: float = 0.0
    sam3_batch_inference_ms: float | None = None
    obb_total_estimation_ms: float = 0.0


class ReconstructObjectsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    task: str
    status: Literal["success", "partial_success", "error"]
    output_root: str
    camera: ResponseCameraModel
    timing: RequestTimingModel
    objects: list[ObjectResultModel]
    errors: list[str]
