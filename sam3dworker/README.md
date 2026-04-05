# sam3dworker

`sam3dworker` 负责当前仓库中面向 SAM3D 的 worker 代码。

## 目录与职责

- worker 目录：`sam3dworker/`
- 共享 IPC 包：`../worker_ipc/`
- 上游 SAM3D 代码快照：`../third_party/SAM3D-object/`

这个 worker 当前负责：

- SAM3D 请求的 worker 入口
- 通过 `worker_ipc` 接收同步请求
- 分发与 SAM3D-object 相关的任务
- 计划消费 RGB、depth、mask 和相机内参，执行 3D 重建
- pointmap 由 `sam3dworker` 内部统一生成，不要求外部直接传 pointmap
- 当前实现已经接入真实 `SAM3D-object` 推理
- pointmap 仍然由 worker 内部生成并落盘
- `gaussian` 和 `mesh` 导出受 `artifact_types` 控制

## 运行环境

- 推荐 conda 环境：`sam3d-objects`
- 推荐解释器：`/root/autodl-tmp/conda/envs/sam3d-objects/bin/python`
- 如果直接使用 conda：`conda run -n sam3d-objects python ...`

## 权重与路径

- 实际 SAM3D-object 权重目录：`/root/hf`
- 上游入口路径应当指向该目录：
  - `../third_party/SAM3D-object/checkpoints/hf -> /root/hf`

## Pointmap 职责边界

当前集成约定：

- 父进程上传 `RGB + depth + mask + intrinsics`
- `sam3dworker` 内部统一把 `depth + intrinsics` 转成 `pointmap`
- 然后再把该 `pointmap` 传给 `SAM3D-object`
- 不建议外部直接构造 `pointmap`，避免单位、坐标系和对齐方式不一致

## 文档

- 上游使用说明：`SAM3D-object_使用指南.md`

## 业务指令汇总

- `ping`
  - 用于探活，返回 `{"status": "ready"}`
- `describe`
  - 返回当前 worker 的静态描述信息
  - 当前代码包含 `worker`、`third_party_dir`、`config_path`、`status`、`supported_commands`、`reconstruct_stage`、`inference_loaded`

### 通用外层格式

所有请求和响应都遵循 `worker_ipc` 的统一信封格式。

请求：

```json
{
  "request_id": "req-1",
  "command": "ping",
  "payload": {}
}
```

成功响应：

```json
{
  "request_id": "req-1",
  "ok": true,
  "payload": {
    "status": "ready"
  }
}
```

失败响应：

```json
{
  "request_id": "req-1",
  "ok": false,
  "payload": {},
  "error": "unknown command: reconstruct"
}
```

### `ping`

请求：

```json
{
  "request_id": "req-1",
  "command": "ping",
  "payload": {}
}
```

响应：

```json
{
  "request_id": "req-1",
  "ok": true,
  "payload": {
    "status": "ready"
  }
}
```

### `describe`

请求：

```json
{
  "request_id": "req-2",
  "command": "describe",
  "payload": {}
}
```

当前代码响应：

```json
{
  "request_id": "req-2",
  "ok": true,
  "payload": {
    "worker": "sam3d",
    "third_party_dir": "/abs/path/to/third_party/SAM3D-object",
    "config_path": "/abs/path/to/third_party/SAM3D-object/checkpoints/hf/pipeline.yaml",
    "status": "ready",
    "supported_commands": ["ping", "describe", "reconstruct"],
    "reconstruct_stage": "full_inference",
    "inference_loaded": false
  }
}
```

### `reconstruct`

下面是 `sam3dworker` 当前的业务协议。当前实现会先内部生成 `pointmap.npy`，再调用上游 `SAM3D-object` 做真实推理。

请求：

```json
{
  "request_id": "req-3",
  "command": "reconstruct",
  "payload": {
    "image_path": "/abs/path/image.png",
    "depth_path": "/abs/path/depth.npy",
    "mask_path": "/abs/path/mask.png",
    "output_dir": "/abs/path/sam3d_outputs/req-3",
    "fx": 912.3,
    "fy": 910.8,
    "cx": 640.0,
    "cy": 360.0,
    "label": "red_cube_0",
    "artifact_types": ["gaussian", "mesh"]
  }
}
```

请求字段约定：

- `image_path`：必填，RGB 图绝对路径
- `depth_path`：必填，`.npy` 绝对路径，内容固定为 `float32` 深度图，单位固定为米
- `mask_path`：必填，单目标 mask 绝对路径
- `output_dir`：必填，输出目录绝对路径
- `fx`：必填，像素坐标系焦距 `x`
- `fy`：必填，像素坐标系焦距 `y`
- `cx`：必填，像素坐标系主点 `x`
- `cy`：必填，像素坐标系主点 `y`
- `label`：必填，目标标识，用于和上游任务对齐
- `artifact_types`：可选，字符串数组，可取值为 `gaussian`、`mesh`
- `artifact_types` 省略时默认 `[]`

输入数据约定：

- `image`、`depth`、`mask` 必须来自同一视角
- `depth.npy` 形状应为 `H x W`
- `mask` 应为单目标 mask
- `depth <= 0`、`NaN`、`inf` 视为无效深度
- 如果 `image`、`depth`、`mask` 尺寸不一致，worker 可按实现策略报错或先做对齐，但第一版建议直接报错

pointmap 内部生成约定：

- `sam3dworker` 内部使用 `depth + intrinsics` 反投影生成 `pointmap`
- 反投影公式固定为：
  - `X = (u - cx) * Z / fx`
  - `Y = (v - cy) * Z / fy`
  - `Z = depth_in_meter`
- 当前实现会在反投影后把 `X` 和 `Y` 翻转，转换到上游 `SAM3D-object` 外部 pointmap 路径所使用的坐标约定
- 生成后的 `pointmap` 建议保存为 `output_dir/pointmap.npy`，便于调试和复现

成功响应：

```json
{
  "request_id": "req-3",
  "ok": true,
  "payload": {
    "worker": "sam3d",
    "label": "red_cube_0",
    "image_path": "/abs/path/image.png",
    "depth_path": "/abs/path/depth.npy",
    "mask_path": "/abs/path/mask.png",
    "output_dir": "/abs/path/sam3d_outputs/req-3",
    "pointmap_path": "/abs/path/sam3d_outputs/req-3/pointmap.npy",
    "pose": {
      "rotation": [1.0, 0.0, 0.0, 0.0],
      "translation": [0.1, 0.2, 0.3],
      "scale": [1.23, 1.23, 1.23]
    },
    "artifacts": {
      "gaussian_ply_path": "/abs/path/sam3d_outputs/req-3/splat.ply",
      "mesh_glb_path": "/abs/path/sam3d_outputs/req-3/mesh.glb"
    }
  }
}
```

响应字段约定：

- `pose` 为必返字段
- `pose.rotation`：长度固定为 4 的数组，表示四元数，顺序固定为 `wxyz`
- `pose.translation`：长度固定为 3 的数组，表示 `[tx, ty, tz]`
- `pose.scale`：长度固定为 3 的数组，表示 `[sx, sy, sz]`
- `pointmap_path` 为必返字段
- `artifacts` 只返回请求中显式要求导出的文件
- 如果 `artifact_types` 为空或省略，`artifacts` 可为空对象

文件输出约定：

- `pointmap.npy` 由 worker 内部生成
- `gaussian_ply_path` 对应高斯结果导出文件
- `mesh_glb_path` 对应 mesh 结果导出文件
- `output_dir` 不存在时由 worker 创建
- 子进程只负责生成，父进程负责清理

错误处理约定：

- 外层 JSON 非法，或缺少必填字段：整条响应 `ok=false`
- `depth_path` 不是 `.npy`，或内容不是米单位 `float32` 深度图：整条响应 `ok=false`
- `image`、`depth`、`mask` 尺寸不匹配：整条响应 `ok=false`
- 重建失败：整条响应 `ok=false`

当前实现的实际行为：
- `reconstruct` 会先校验请求字段
- 然后读取 `image`、`depth`、`mask`
- 在 `output_dir` 下生成 `pointmap.npy`
- 再把 `image + mask + pointmap` 传给上游 `SAM3D-object`
- 返回标准化后的 `pose`
- 按 `artifact_types` 决定是否导出 `splat.ply` 和 `mesh.glb`

## 当前状态

这个目录已经接入了真实 `SAM3D-object` 推理路径，但仍然依赖正确的上游 conda 环境、Hydra 补丁和权重目录。当前本地代码尚未在目标环境里执行验证。
