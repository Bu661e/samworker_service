# samworker_service

`samworker_service` 是一个面向单帧 RGB-D 输入的 FastAPI 服务。服务常驻拉起 `sam3worker`，接收 base64 编码的 RGB 图像、depth `.npy` 和一组 2D bbox，返回每个目标的 `SAM3` 分割结果，以及基于可见深度点云估计出的相机坐标系 3D OBB。

当前主流程不依赖 `sam3dworker`；`sam3dworker/` 仍保留在仓库中，但不是这个服务的在线推理链路。

## 主要能力

- `GET /`：健康检查，返回 `{"status": "ok"}`
- `POST /v1/objects/reconstruct`：对请求中的每个 bbox 做分割并估计 `object_3d`
- 输入图像和深度通过 `content_base64` 直接上传，服务端不访问外部 artifact URL
- 所有 3D 输出都在相机坐标系下，坐标轴约定为 `x=left`、`y=up`、`z=forward`

## 仓库结构

- [`sam_pipeline_api/`](sam_pipeline_api/)：FastAPI 入口、请求模型、推理编排、示例请求
- [`sam3worker/`](sam3worker/)：常驻 `SAM3` worker，负责 2D mask 推理，也包含批量任务脚本
- [`object_geometry/`](object_geometry/)：根据 `mask + depth + intrinsics` 估计可见点云的 PCA 3D OBB
- [`worker_ipc/`](worker_ipc/)：基于 Unix domain socket JSONL 的父子进程 IPC 组件
- [`sam3dworker/`](sam3dworker/)：仓库保留的 3D worker 代码，不在当前主流程中
- [`docs/`](docs/)：设计说明和重构文档

## 运行要求

- 推荐使用现有 `conda base` 环境，默认 worker Python 路径是 `/opt/conda/bin/python`
- 当前代码要求模型权重文件存在于 `/root/sam3.pt`
- 仓库根目录没有统一的顶层 `requirements.txt`、`setup.py` 或 `pyproject.toml`

运行服务至少需要这些 Python 依赖：

- `fastapi`
- `pydantic`
- `uvicorn`
- `ultralytics`
- `numpy`
- `pillow`
- `pytest`

## 快速启动

```bash
cd /root/samworker_service
source /opt/conda/etc/profile.d/conda.sh
conda activate base
python -m pip install fastapi pydantic uvicorn ultralytics numpy pillow pytest
ls -l /root/sam3.pt
/opt/conda/bin/python -m sam_pipeline_api.serve
```

也可以直接用 `uvicorn`：

```bash
cd /root/samworker_service
/opt/conda/bin/python -m uvicorn sam_pipeline_api.app:app --host 0.0.0.0 --port 6006
```

启动成功后默认监听 `0.0.0.0:6006`。

## 环境变量

- `SAM_PIPELINE_HOST`：服务监听地址，默认 `0.0.0.0`
- `SAM_PIPELINE_PORT`：服务端口，默认 `6006`
- `SAM_PIPELINE_PYTHON`：拉起 `sam3worker` 时使用的 Python，默认 `/opt/conda/bin/python`
- `SAM_PIPELINE_STARTUP_TIMEOUT_S`：worker 启动超时，默认 `180.0`
- `SAM_PIPELINE_RUN_ROOT`：请求输出目录根路径，默认 `sam_pipeline_api/runs`
- `SAM_PIPELINE_SOCKET_DIR`：worker socket 目录，默认 `sam_pipeline_api/sockets`
- `SAM_PIPELINE_TRACE_DIR`：IPC trace 目录，默认 `sam_pipeline_api/traces`

## 请求与响应

完整示例见 [`sam_pipeline_api/examples/reconstruct.request.json`](sam_pipeline_api/examples/reconstruct.request.json)。

最小请求结构如下：

```json
{
  "task": "把红色方块放到蓝色方块上面",
  "bboxes": [
    {
      "label": "red_cube_0",
      "bbox_2d": [244, 294, 274, 333]
    }
  ],
  "camera": {
    "id": "table_overview",
    "intrinsics": {
      "fx": 533.3333740234375,
      "fy": 533.3333740234375,
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
        "content_base64": "<base64-encoded-rgb-png>"
      }
    },
    "depth_image": {
      "unit": "meter",
      "ref": {
        "id": "artifact-depth-001",
        "kind": "artifact_file",
        "content_type": "application/x-npy",
        "content_base64": "<base64-encoded-depth-npy>"
      }
    }
  }
}
```

接口约束：

- `bboxes[].label` 在单次请求内必须唯一
- `bboxes[].bbox_2d` 必须满足 `[x1, y1, x2, y2]` 且 `x2 > x1`、`y2 > y1`
- `camera.depth_image.unit` 当前只接受 `meter`、`meters` 或 `m`
- `content_base64` 既支持纯 base64 字符串，也支持 `data:<mime>;base64,<payload>` 形式
- `request_id`、`output_root`、`sam3_timeout_s`、`camera.pose` 是可选字段

响应中会返回：

- `request_id`、`task`、`status`
- `output_root`：本次请求的输出目录
- `camera`：返回结果使用的坐标系约定
- `timing`：请求级耗时统计
- `objects[]`：每个 bbox 的分割结果、3D 几何结果和错误信息
- `errors[]`：请求级错误汇总

## 运行产物

默认情况下，每次请求会在 `sam_pipeline_api/runs/<request_id>/` 下生成：

- `inputs/`：服务端从 base64 解码落盘的 RGB 和 depth 文件
- `sam3/`：`SAM3` 生成的 mask PNG

服务运行时还会额外生成：

- `sam_pipeline_api/sockets/sam3.sock`：worker IPC socket
- `sam_pipeline_api/traces/sam3-client-trace.jsonl`：IPC trace

这些目录已经被 `.gitignore` 排除，不应该进入最终提交。

## 测试

轻量、CPU 可运行的测试：

```bash
python3 -m pytest sam_pipeline_api/tests -q
python3 -m pytest sam3worker/tests/test_batch_tasks.py -q
(cd worker_ipc && python3 -m pytest tests -q)
```

依赖 GPU、`ultralytics` 和 `/root/sam3.pt` 的测试：

```bash
python3 -m pytest sam3worker/tests -q
```

`sam3worker/tests` 在环境不满足时会自动 `skip`。`sam3dworker/tests` 仍可单独执行，但不属于当前主流程验收范围。

## 当前限制

- 服务层当前通过一个全局锁串行处理请求
- 在线主流程只使用 bbox prompt 模式
- `object_3d` 来源于可见深度点云的 `PCA3D OBB`，不是完整重建结果
- 当前不会把结果转换到世界坐标系
- `worker_ipc` 通过仓库内路径直接导入，不要求单独安装成 site-package
