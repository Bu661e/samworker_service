# sam3worker

`sam3worker` 负责当前仓库中面向 SAM3 的 worker 代码。

## 目录与职责

- worker 目录：`sam3worker/`
- 共享 IPC 包：`../worker_ipc/`

这个 worker 当前负责：

- SAM3 请求的 worker 入口
- 通过 `worker_ipc` 接收同步请求
- 分发与 SAM3 相关的任务
- 进程启动时加载 `SAM("sam3.pt")` 到当前 Python 进程并常驻复用
- 已接入基于 `SAM("sam3.pt") + bboxes` 的单图推理代码
- 当前仓库内尚未验证真实环境运行结果

## 运行环境

- 推荐 conda 环境：`base`
- 推荐解释器：`/opt/conda/bin/python`
- 如果直接使用 conda：`conda run -n base python ...`

## 权重与路径

- 实际 SAM3 权重文件：`/root/sam3.pt`
- worker 启动时会先检查该文件是否存在
- 如果模型加载失败，worker 不进入服务循环，直接启动失败

## 文档

- 使用说明：`ultralytics_SAM3_使用指南.md`

## 父进程使用方式

外部模块如果要以子进程方式使用 `sam3worker`，优先建议直接使用仓库内提供的父进程封装：

- `from sam3worker import Sam3WorkerClient`

这层封装内部已经基于 `worker_ipc.ManagedChildProcess` 处理了：

- worker 子进程拉起
- `ping` 探活
- Unix Domain Socket 路径传递
- `start()` / `stop()`
- `describe`、`ping`、`infer(...)` 的业务方法封装

推荐示例：

```python
from pathlib import Path

from sam3worker import Sam3WorkerClient


python_bin = "/opt/conda/bin/python"

with Sam3WorkerClient(
    socket_path=Path("/tmp/sam3-worker.sock"),
    python_executable=python_bin,
) as client:
    meta = client.describe()
    result = client.infer(
        image_path="/abs/path/image.jpg",
        output_dir="/abs/path/sam3_outputs/req-1",
        bboxes=[
            {"label": "red_cube_0", "bbox_2d": [379, 458, 431, 522]},
            {"label": "blue_cube_0", "bbox_2d": [301, 365, 353, 427]},
        ],
        timeout=60.0,
    )
```

接口约定：

- `Sam3WorkerClient.start()`：启动 worker 并等待 `ping` 就绪
- `Sam3WorkerClient.stop()`：关闭 worker 子进程
- `Sam3WorkerClient.call_raw(...)`：返回 `worker_ipc.Response`
- `Sam3WorkerClient.call(...)`：返回业务 `payload`，如果 worker 返回 `ok=false` 则抛 `Sam3WorkerCommandError`
- `Sam3WorkerClient.describe()` / `ping()` / `infer(...)`：对应业务命令的便捷方法

如果外部模块已经自己管理了 worker 生命周期，也可以直接使用 `worker_ipc.UdsJsonlClient` 或 `ManagedChildProcess`；但在当前仓库里，优先建议使用 `Sam3WorkerClient`，避免重复拼装请求格式。

## GPU 集成测试

`sam3worker/tests/` 当前已经切成真实 worker 集成测试，不再使用 mock 模型或 fake IPC。

测试特点：

- 通过 `Sam3WorkerClient` 真正拉起 `sam3worker/worker.py`
- 使用 `sam3worker/tests/inputs/example.png`
- 使用 `sam3worker/tests/inputs/bboxes.json`
- 真实执行 GPU 推理，并检查输出 mask、bbox 和耗时字段

运行前提：

- 机器可见 GPU
- `torch.cuda.is_available()` 为 `True`
- `/root/sam3.pt` 存在
- conda base 路径 `/opt/conda` 可用，解释器为 `/opt/conda/bin/python`

推荐测试命令：

```bash
/opt/conda/bin/python -m pytest sam3worker/tests
```

如果当前机器没有 GPU，测试会自动跳过，不会伪造 CPU 路径结果。

## 业务指令汇总

- `ping`
  - 用于探活，返回 `{"status": "ready"}`
- `describe`
  - 返回当前 worker 的静态描述信息
  - 当前包含 `worker`、`status`、`weight_path`、`supported_commands`、`prompt_modes`、`model_loaded`

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
  "error": "unknown command: infer"
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
    "worker": "sam3",
    "status": "ready",
    "weight_path": "/root/sam3.pt",
    "supported_commands": ["ping", "describe", "infer"],
    "prompt_modes": ["bbox"],
    "model_loaded": true
  }
}
```

### `infer`

下面是当前代码实现对应的第一版 `bbox` 推理协议。实际运行仍需在装好 `ultralytics` 和 `sam3.pt` 的目标环境中验证。

请求：

```json
{
  "request_id": "req-3",
  "command": "infer",
  "payload": {
    "image_path": "/abs/path/image.jpg",
    "output_dir": "/abs/path/sam3_outputs/req-3",
    "bboxes": [
      {
        "bbox_2d": [379, 458, 431, 522],
        "label": "red_cube_0"
      },
      {
        "bbox_2d": [301, 365, 353, 427],
        "label": "blue_cube_0"
      }
    ]
  }
}
```

请求字段约定：

- `image_path`：必填，原图绝对路径
- `output_dir`：必填，mask 输出目录，由父进程指定
- `bboxes`：必填，至少 1 个元素
- `bbox_2d`：格式固定为 `[x1, y1, x2, y2]`，原图像素坐标，`xyxy`
- `label`：必填，单次请求内必须唯一，仅用于结果对齐和回传

成功响应：

```json
{
  "request_id": "req-3",
  "ok": true,
  "payload": {
    "worker": "sam3",
    "prompt_mode": "bbox",
    "image_path": "/abs/path/image.jpg",
    "output_dir": "/abs/path/sam3_outputs/req-3",
    "batch_model_inference_ms": 18.4,
    "results": [
      {
        "label": "red_cube_0",
        "prompt_bbox_2d": [379, 458, 431, 522],
        "found": true,
        "bbox_2d": [380, 459, 430, 521],
        "mask_path": "/abs/path/sam3_outputs/req-3/000_red_cube_0.png",
        "avg_inference_ms": 9.2
      },
      {
        "label": "blue_cube_0",
        "prompt_bbox_2d": [301, 365, 353, 427],
        "found": false,
        "bbox_2d": null,
        "mask_path": null,
        "avg_inference_ms": 9.2
      }
    ]
  }
}
```

响应字段约定：

- `results` 与请求中的 `bboxes` 顺序保持一致
- `batch_model_inference_ms`：当前这次请求里，SAM3 对整组 `bboxes` 做单次批量推理的耗时，单位毫秒
- `prompt_bbox_2d`：原样回传输入提示框
- `avg_inference_ms`：`batch_model_inference_ms / len(bboxes)` 的分摊值，便于按 object 统计；它不是每个 object 的真实独立 GPU 推理时间
- `found=true` 时：
  - `bbox_2d` 为实际分割结果对应的 `xyxy`
  - `mask_path` 为生成的 mask 文件绝对路径
- `found=false` 时：
  - `bbox_2d` 为 `null`
  - `mask_path` 为 `null`

mask 文件约定：

- 文件格式：单通道 PNG
- 前景像素值：`255`
- 背景像素值：`0`
- `output_dir` 不存在时由 worker 创建
- 子进程只负责生成，父进程负责清理

错误处理约定：

- 外层 JSON 非法，或缺少必填字段：整条响应 `ok=false`
- 某个 bbox 未分出目标：整条响应仍为 `ok=true`，由该结果项的 `found=false` 表示
