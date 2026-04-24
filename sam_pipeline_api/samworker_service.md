# sam_pipeline_api

这个分支提供一个不依赖 `sam3dworker` 的 FastAPI 服务。服务只常驻拉起 `sam3worker`，然后用 `object_geometry/` 根据 `SAM3 mask + depth + camera intrinsics` 计算相机坐标系下的 3D OBB。

当前只暴露一个业务接口：

```text
POST /v1/objects/reconstruct
```

## 服务行为

服务启动后会常驻拉起：

- `sam3worker`

单次请求流程：

1. 根据 `camera.rgb_image.ref.download_url` 下载 RGB
2. 根据 `camera.depth_image.ref.download_url` 下载 depth
3. `SAM3` 根据 `bboxes[]` 生成每个目标的 `mask`
4. `object_geometry` 根据 `mask + depth + intrinsics` 估计相机坐标系下的 `OBB`
5. 返回 `objects[]`

## 启动方式

请使用 `base` 环境里的 Python：

```bash
cd /root/samworker_service
/opt/conda/bin/python -m sam_pipeline_api.serve
```

或者：

```bash
cd /root/samworker_service
/opt/conda/bin/python -m uvicorn sam_pipeline_api.app:app --host 0.0.0.0 --port 6006
```

## 请求格式

```json
{
  "request_id": "demo-request",
  "task": "把红色方块放到蓝色方块上面",
  "bboxes": [
    {
      "bbox_2d": [379, 458, 431, 522],
      "label": "red_cube_0"
    }
  ],
  "camera": {
    "id": "table_top",
    "intrinsics": {
      "fx": 533.33,
      "fy": 533.33,
      "cx": 320.0,
      "cy": 320.0,
      "width": 640,
      "height": 640
    },
    "rgb_image": {
      "ref": {
        "id": "artifact-rgb-001",
        "kind": "artifact_file",
        "content_type": "image/png",
        "download_url": "http://api-host/.../artifacts/rgb"
      }
    },
    "depth_image": {
      "unit": "meter",
      "ref": {
        "id": "artifact-depth-001",
        "kind": "artifact_file",
        "content_type": "application/x-npy",
        "download_url": "http://api-host/.../artifacts/depth"
      }
    }
  }
}
```

必填字段：

- `task`: 上游任务描述，服务原样回传
- `bboxes[].label`: 单次请求内必须唯一
- `bboxes[].bbox_2d`: 提供给 `SAM3` 的提示框，格式 `[x1, y1, x2, y2]`
- `camera.intrinsics`: 相机内参
- `camera.rgb_image.ref.download_url`: RGB 下载地址
- `camera.depth_image.unit`: 当前要求是 `meter`
- `camera.depth_image.ref.download_url`: depth `.npy` 下载地址

可选字段：

- `request_id`: 不传则服务自动生成
- `output_root`: 自定义本次请求的输出目录
- `sam3_timeout_s`: `SAM3` 超时
- `camera.pose`: 当前接收但不参与世界坐标变换

## 响应格式

```json
{
  "request_id": "req-123",
  "task": "把红色方块放到蓝色方块上面",
  "status": "success",
  "output_root": "/abs/path/sam_pipeline_api/runs/req-123",
  "camera": {
    "id": "table_top",
    "coordinate_frame": "camera",
    "axis_convention": {
      "x": "left",
      "y": "up",
      "z": "forward"
    }
  },
  "timing": {
    "total_ms": 120.5,
    "download_inputs_ms": 43.2,
    "sam3_batch_inference_ms": 21.4,
    "obb_total_estimation_ms": 7.8
  },
  "objects": [
    {
      "label": "red_cube_0",
      "status": "success",
      "segmentation": {
        "prompt_bbox_2d": [379, 458, 431, 522],
        "found": true,
        "bbox_2d": [380, 459, 430, 521],
        "mask_path": "/abs/path/.../sam3/000_red_cube_0.png"
      },
      "object_3d": {
        "source": "visible_depth_pca3d_obb",
        "position_xyz_m": [0.401, 0.053, 3.628],
        "rotation_quaternion_wxyz": [0.0, 0.0, 0.0, 1.0],
        "rotation_matrix_camera_from_obb": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "size_xyz_m": [0.081, 0.080, 0.079],
        "obb_corners_xyz_m": [],
        "visible_point_centroid_xyz_m": [0.401, 0.053, 3.628],
        "visible_point_count": 1321
      },
      "timing": {
        "sam3_avg_inference_ms": 10.7,
        "obb_estimation_ms": 3.2
      },
      "error": null
    }
  ],
  "errors": []
}
```

## 坐标约定

当前所有返回的 3D 数值都在相机坐标系下：

- `x`: left
- `y`: up
- `z`: forward

外部系统建议主用：

- `objects[].object_3d.position_xyz_m`
- `objects[].object_3d.rotation_quaternion_wxyz`
- `objects[].object_3d.size_xyz_m`
- `objects[].object_3d.obb_corners_xyz_m`

## 输出目录

默认会在 `sam_pipeline_api/runs/<request_id>/` 下生成：

- `inputs/`: 下载下来的 RGB 和 depth
- `sam3/`: `SAM3` 输出的 mask

## 当前限制

- 服务层当前用一个全局锁串行处理请求
- `SAM3` 如果某个框没有分出目标，该物体会返回 `status=not_found`
- OBB 估计失败时，该物体会返回 `status=error`
- 当前 `object_3d` 来自可见深度点云的 `PCA3D OBB`，具体几何实现位于 `object_geometry/`
- 当前不会把结果转换到世界坐标系
