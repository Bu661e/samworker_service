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
- 上游推理入口当前默认开启 `with_layout_postprocess=True`
- worker 启动时会先 preload 模型，并执行一次内部 warmup 推理，再进入服务循环

## 运行环境

- 推荐 conda 环境：`base`
- 推荐解释器：`/opt/conda/bin/python`
- 如果直接使用 conda：`conda run -n base python ...`

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

## 父进程使用方式

外部模块如果要以子进程方式使用 `sam3dworker`，优先建议直接使用仓库内提供的父进程封装：

- `from sam3dworker import Sam3dWorkerClient`

这层封装内部已经基于 `worker_ipc.ManagedChildProcess` 处理了：

- worker 子进程拉起
- 启动前自动尝试 `source /etc/network_turbo`
- `ping` 探活
- Unix Domain Socket 路径传递
- `start()` / `stop()`
- `describe`、`ping`、`reconstruct(...)` 的业务方法封装

推荐示例：

```python
from pathlib import Path

from sam3dworker import Sam3dWorkerClient


python_bin = "/opt/conda/bin/python"

with Sam3dWorkerClient(
    socket_path=Path("/tmp/sam3d-worker.sock"),
    python_executable=python_bin,
) as client:
    meta = client.describe()
    result = client.reconstruct(
        image_path="/abs/path/image.png",
        depth_path="/abs/path/depth.npy",
        mask_path="/abs/path/mask.png",
        output_dir="/abs/path/sam3d_outputs/req-1",
        fx=912.3,
        fy=910.8,
        cx=640.0,
        cy=360.0,
        label="red_cube_0",
        artifact_types=["gaussian", "mesh"],
        timeout=300.0,
    )
```

接口约定：

- `Sam3dWorkerClient.start()`：启动 worker 并等待 `ping` 就绪
- 默认会通过 `bash -lc` 先尝试 `source /etc/network_turbo`，再 `exec` worker 进程；用于让 Hugging Face 等外网依赖在当前环境下可达
- `Sam3dWorkerClient.stop()`：关闭 worker 子进程
- `Sam3dWorkerClient.call_raw(...)`：返回 `worker_ipc.Response`
- `Sam3dWorkerClient.call(...)`：返回业务 `payload`，如果 worker 返回 `ok=false` 则抛 `Sam3dWorkerCommandError`
- `Sam3dWorkerClient.describe()` / `ping()` / `reconstruct(...)`：对应业务命令的便捷方法

如果外部模块已经自己管理了 worker 生命周期，也可以直接使用 `worker_ipc.UdsJsonlClient` 或 `ManagedChildProcess`；但在当前仓库里，优先建议使用 `Sam3dWorkerClient`，避免重复拼装请求格式。

## GPU 集成测试

`sam3dworker/tests/` 当前已经切成真实 worker 集成测试，不再使用 fake IPC 或 mock 推理结果。

测试特点：

- 通过 `Sam3dWorkerClient` 真正拉起 `sam3dworker/worker.py`
- 使用 `sam3dworker/tests/inputs/emp_default_tableoverview/` 下的 `example.png`、`example.npy`、`red_cube_0.png`、`blue_cube_0.png`
- 从 `payloads.json` 读取请求模板
- `payloads.json` 不保存 `output_dir`
- 测试运行时会自动把每个请求的 `output_dir` 注入到 `sam3dworker/tests/runs/<时间戳>/...`
- 真实执行 GPU 重建，并检查 `pointmap.npy`、`splat.ply`、`mesh.glb`、`pose` 和耗时字段

运行前提：

- 机器可见 GPU
- `torch.cuda.is_available()` 为 `True`
- `/root/hf` 权重目录存在
- `third_party/SAM3D-object/checkpoints/hf -> /root/hf` 有效
- conda base 路径 `/opt/conda` 可用，解释器为 `/opt/conda/bin/python`
- 如果当前环境访问 Hugging Face 需要网络切换，worker 启动前需要先执行：

```bash
source /etc/network_turbo
```

- 当前 `Sam3dWorkerClient` 已经默认通过 `bash -lc` 在子进程启动前自动执行这条命令；如果你是手工启动 `sam3dworker/worker.py`，也应当手工先执行这条命令

推荐测试命令：

```bash
/opt/conda/bin/python -m pytest sam3dworker/tests
```

基线实测时间（2026-04-07，输入为 `emp_default_tableoverview`，当时 `with_layout_postprocess=False`）：

- worker 启动到 ready，总耗时约 `48.67s`
  - 这个时间包含 preload + warmup
- `red_cube_0` 单次 `reconstruct`
  - 父进程 wall time：约 `6.714s`
  - worker 返回的 `model_inference_ms`：约 `6084.888ms`
- `blue_cube_0` 单次 `reconstruct`
  - 父进程 wall time：约 `6.582s`
  - worker 返回的 `model_inference_ms`：约 `5931.002ms`

当前默认配置已经开启 `with_layout_postprocess=True`。这会在初始 pose 之后，再做一轮基于 `mask + pointmap + intrinsics` 的姿态后处理优化；如果走 Gaussian 路径，还会额外使用 `RGB` 图像做监督。当前对外仍然只返回 `pose.rotation`、`pose.translation`、`pose.scale`，但这些值已经是后处理修正后的结果。

开启 `with_layout_postprocess=True` 之后，单次 `reconstruct` 的额外时间成本在当前测试输入上约为 `1s`：

- `red_cube_0`
  - 关闭后处理时：`5905.296ms`
  - 开启后处理时：`6884.893ms`
  - 增加：`979.597ms`，约 `+16.6%`
- `blue_cube_0`
  - 关闭后处理时：`5930.211ms`
  - 开启后处理时：`7058.488ms`
  - 增加：`1128.277ms`，约 `+19.0%`
- 两个样本平均增加：`1053.937ms`，约 `+17.8%`

这部分开销主要发生在单请求 `reconstruct` 阶段。由于 warmup 也会走同一条推理入口，worker 启动阶段理论上也会受影响；但当前文档里没有单独拆出开启后处理后的最新启动耗时，因此上面的 `48.67s` 应视为关闭后处理时的启动基线。

对应产物和时间记录文件：

- timing probe 输出目录：`sam3dworker/tests/runs/2026-04-07-04-36-28-timing-probe/`
- 时间汇总文件：`sam3dworker/tests/runs/2026-04-07-04-36-28-timing-probe/timing-summary.json`
- 开启 `with_layout_postprocess=True` 后的最新测试输出目录：`sam3dworker/tests/runs/2026-04-07-05-11-36/`
- 对应 trace：`sam3dworker/tests/runs/2026-04-07-05-11-36/worker-runtime/sam3d-trace.jsonl`

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
    "model_inference_ms": 842.1,
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
- `model_inference_ms`：当前这次单 object `reconstruct` 请求的真实模型推理耗时，单位毫秒
- `artifacts` 只返回请求中显式要求导出的文件
- 如果 `artifact_types` 为空或省略，`artifacts` 可为空对象
- 当前默认开启 `with_layout_postprocess=True`，因此这里的 `pose` 是后处理修正后的结果，不是纯模型初始输出
- 当前 worker 还没有把上游后处理内部的 `iou`、`iou_before_optim`、`optim_accepted` 暴露给外部调用方

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
- 上游当前默认启用 `with_layout_postprocess=True`
- 返回标准化后的 `pose`
- 按 `artifact_types` 决定是否导出 `splat.ply` 和 `mesh.glb`

## 当前状态

这个目录已经接入真实 `SAM3D-object` 推理路径，并且已经在当前目标环境完成真实 GPU 集成测试验证。当前仍然依赖正确的上游 conda 环境、权重目录和网络环境切换配置。
