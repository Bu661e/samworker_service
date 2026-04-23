# sam_pipeline_api

这个目录提供一个对外的 FastAPI 服务，用来把：

- `SAM3`
- `SAM3D`

串成一个单接口。

当前只暴露一个业务接口：

- `POST /v1/objects/reconstruct`

## 服务行为

服务启动后会先常驻拉起两个 worker：

- `sam3worker`
- `sam3dworker`

也就是说，FastAPI 进程起来以后，这两个 worker 就已经启动并完成模型加载，后续每次请求直接复用，不会在每个请求里重新起 worker。

单次请求的固定执行顺序是：

1. 根据 `camera.rgb_image.ref.download_url` 下载 RGB
2. 根据 `camera.depth_image.ref.download_url` 下载 depth
3. `SAM3` 根据 `bboxes[]` 生成每个目标的 `mask`
4. `SAM3D` 根据 `RGB + depth + mask + intrinsics` 做单目标重建
5. 服务层根据 `mask + depth + intrinsics` 再估计一个相机坐标系下的 `OBB`
6. 返回 `objects[]`

## 启动方式

请使用 `base` 环境里的 Python。

推荐命令：

```bash
cd /root/samworker_service
/opt/conda/bin/python -m sam_pipeline_api.serve
```

或者：

```bash
cd /root/samworker_service
/opt/conda/bin/python -m uvicorn sam_pipeline_api.app:app --host 0.0.0.0 --port 6006
```

说明：

- `sam3dworker` 子进程内部已经会在启动前尝试 `source /etc/network_turbo`
- 这里不需要再额外包一层 shell

## 接口地址

```text
POST /v1/objects/reconstruct
```

## 请求格式

当前接口按下面这个结构收参：

```json
{
  "request_id": "demo-request",
  "task": "把红色方块放到蓝色方块上面",
  "bboxes": [
    {
      "bbox_2d": [379, 458, 431, 522],
      "label": "red_cube_0"
    },
    {
      "bbox_2d": [301, 365, 353, 427],
      "label": "blue_cube_0"
    }
  ],
  "camera": {
    "id": "table_top",
    "status": "ready",
    "prim_path": "/World/Cameras/TableTopCamera",
    "mount_mode": "world",
    "resolution": [640, 640],
    "intrinsics": {
      "fx": 533.33,
      "fy": 533.33,
      "cx": 320.0,
      "cy": 320.0,
      "width": 640,
      "height": 640
    },
    "pose": {
      "position_xyz_m": [0.0, 0.0, 6.0],
      "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]
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

完整示例文件：

- [reconstruct.request.json](/root/samworker_service/sam_pipeline_api/examples/reconstruct.request.json)
- [新接口定义.md](/root/samworker_service/sam_pipeline_api/新接口定义.md)

### 请求字段说明

必填字段：

- `task`
  - 上游任务描述，服务会原样回传
- `bboxes`
  - 需要处理的目标数组
- `bboxes[].label`
  - 单次请求内必须唯一
- `bboxes[].bbox_2d`
  - 提供给 `SAM3` 的提示框，格式 `[x1, y1, x2, y2]`
- `camera.id`
  - 相机标识
- `camera.intrinsics`
  - 相机内参
- `camera.rgb_image.ref.download_url`
  - RGB 下载地址
- `camera.depth_image.unit`
  - 当前要求是 `meter`
- `camera.depth_image.ref.download_url`
  - depth 下载地址

可选字段：

- `request_id`
  - 不传则服务自动生成
- `artifact_types`
  - 可选值：`mesh`、`gaussian`
  - 默认：`["mesh"]`
  - 当前对外请求如果没有特殊需要，直接不传即可
- `output_root`
  - 自定义本次请求的输出目录
- `sam3_timeout_s`
  - `SAM3` 超时
- `sam3d_timeout_s`
  - `SAM3D` 超时
- `camera.pose`
  - 当前会接收，但这版服务不会拿它做世界坐标变换

### 输入 artifact 约定

- `rgb_image.ref.download_url` 需要返回一张 RGB 图
- `depth_image.ref.download_url` 需要返回 `.npy` 深度图
- 当前深度单位要求是米
- `download_url` 可以是 `http(s)`，也可以是 `file://`

## 响应格式

响应主结构如下：

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
    "total_ms": 7240.5,
    "download_inputs_ms": 43.2,
    "sam3_batch_inference_ms": 21.4,
    "sam3d_total_inference_ms": 6884.9,
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
        "rotation_matrix_camera_from_obb": [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]
        ],
        "size_xyz_m": [0.081, 0.080, 0.079],
        "obb_corners_xyz_m": [
          [0.36, 0.01, 3.59],
          [0.36, 0.01, 3.67],
          [0.36, 0.09, 3.59],
          [0.36, 0.09, 3.67],
          [0.44, 0.01, 3.59],
          [0.44, 0.01, 3.67],
          [0.44, 0.09, 3.59],
          [0.44, 0.09, 3.67]
        ],
        "visible_point_centroid_xyz_m": [0.401, 0.053, 3.628],
        "visible_point_count": 1321
      },
      "pose_camera": {
        "rotation_quaternion_wxyz": [0.0087, 0.0101, 0.8875, 0.4605],
        "translation_xyz_m": [0.3959, 0.0513, 3.6316],
        "scale_xyz_m": [0.4388, 0.4388, 0.4388]
      },
      "artifacts": {
        "pointmap_path": "/abs/path/.../pointmap.npy",
        "mesh_glb_path": "/abs/path/.../mesh.glb",
        "gaussian_ply_path": null
      },
      "timing": {
        "sam3_avg_inference_ms": 10.7,
        "sam3d_inference_ms": 6884.9,
        "obb_estimation_ms": 3.2
      },
      "error": null
    }
  ],
  "errors": []
}
```

### 响应字段说明

顶层字段：

- `request_id`
  - 请求标识
- `task`
  - 原样回传任务描述
- `status`
  - `success` / `partial_success` / `error`
- `output_root`
  - 本次请求在服务端的落盘目录
- `camera`
  - 当前返回结果所使用的坐标系说明
- `timing`
  - 请求级耗时
- `objects`
  - 每个 bbox 对应一个物体结果
- `errors`
  - 请求级错误摘要

`objects[]` 内每个元素：

- `label`
  - 和输入 `bboxes[].label` 对齐
- `status`
  - `success` / `partial_success` / `not_found` / `error`
- `segmentation`
  - `SAM3` 的 2D 分割结果
- `object_3d`
  - 最终给外部消费的 3D 结果
- `pose_camera`
  - `SAM3D` 原始 pose，主要用于调试和对比
- `artifacts`
  - `SAM3D` 产物路径
  - 默认请求下通常只会有 `mesh_glb_path`，`gaussian_ply_path` 会是 `null`
- `timing`
  - 单物体耗时
- `error`
  - 单物体错误信息

## 推荐消费方式

如果外部系统只关心“这个物体最终的 3D 信息”，建议直接使用：

- `objects[i].object_3d.position_xyz_m`
- `objects[i].object_3d.rotation_quaternion_wxyz`
- `objects[i].object_3d.size_xyz_m`
- `objects[i].object_3d.obb_corners_xyz_m`

`pose_camera` 不建议直接当最终结果用，它更适合拿来做调试比对。

## 坐标约定

当前所有返回的 3D 数值都在相机坐标系下：

- `x`: left
- `y`: up
- `z`: forward

主要包括：

- `objects[].object_3d.position_xyz_m`
- `objects[].object_3d.obb_corners_xyz_m`
- `objects[].pose_camera.translation_xyz_m`

## 请求示例

```bash
curl -X POST http://127.0.0.1:6006/v1/objects/reconstruct \
  -H 'Content-Type: application/json' \
  --data @sam_pipeline_api/examples/reconstruct.request.json
```

## 输出目录

默认会在 `sam_pipeline_api/runs/<request_id>/` 下生成：

- `inputs/`
  - 下载下来的 RGB 和 depth
- `sam3/`
  - `SAM3` 输出的 mask
- `objects/<label>/`
  - `SAM3D` 输出的 `pointmap.npy`
  - `mesh.glb`
  - 如果请求了 `gaussian`，还会有 `splat.ply`

## 当前限制

- 服务层当前用一个全局锁串行处理请求，不会并发复用同一组 worker
- `SAM3` 如果某个框没有分出目标，该物体会返回 `status=not_found`
- `SAM3D` 如果某个物体重建失败，该物体会返回 `status=error`
- 如果 `SAM3D` 成功但 `object_3d` 估计失败，该物体会返回 `status=partial_success`
- 当前 `object_3d` 来自可见深度点云的 `PCA3D OBB`，具体几何实现位于 `object_geometry/`
- 当前不会把结果转换到世界坐标系
- 当前默认只请求 `mesh`，不会请求 `gaussian`
