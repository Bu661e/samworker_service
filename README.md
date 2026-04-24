# samworker_service

当前分支：`no-sam3d-pipeline`

这个分支提供一个不依赖 `sam3dworker` 的 FastAPI 服务。服务常驻拉起 `sam3worker`，再结合 `object_geometry/` 用 `SAM3 mask + depth + camera intrinsics` 计算相机坐标系下的 3D OBB。

## 目录结构

- `sam_pipeline_api/`：FastAPI 服务入口，请求编排逻辑，请求示例也在这里
- `sam3worker/`：常驻 SAM3 worker，负责 2D mask 推理
- `sam3dworker/`：仓库中仍保留，但这个分支的主流程不依赖它
- `object_geometry/`：基于 depth + mask 估计相机坐标系 OBB
- `worker_ipc/`：仓库内的 Unix domain socket JSONL IPC 组件，供 worker 代码直接导入
- `docs/`：补充设计说明

## 环境和依赖

- 推荐使用 `conda` 的 `base` 环境
- 仓库根目录没有顶层 `requirements.txt`、`setup.py` 或 `pyproject.toml`
- `worker_ipc/` 是仓库内子目录，按当前代码路径不需要单独安装

从一个未安装本项目依赖的 `conda base` 开始，至少需要这些包：

- `fastapi`
- `pydantic`
- `uvicorn`
- `ultralytics`
- `pytest`

服务启动前还需要准备模型权重：

- `/root/sam3.pt`

如果没有这个文件，`sam3worker` 启动会失败。

## 安装

拉取指定分支：

```bash
git clone --depth 1 --single-branch --branch no-sam3d-pipeline https://github.com/Bu661e/samworker_service.git
cd samworker_service
```

进入 `conda base`：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate base
```

安装运行依赖：

```bash
pip install fastapi pydantic uvicorn ultralytics pytest
```

## 启动服务

在仓库根目录执行：

```bash
cd /root/samworker_service
/opt/conda/bin/python -m sam_pipeline_api.serve
```

或者直接用 `uvicorn`：

```bash
cd /root/samworker_service
/opt/conda/bin/python -m uvicorn sam_pipeline_api.app:app --host 0.0.0.0 --port 6006
```

## 启动前检查

确认模型权重存在：

```bash
ls -l /root/sam3.pt
```

如果缺失，典型报错是：

```text
SAM3 weight file does not exist: /root/sam3.pt
```

## 测试

轻量 IPC 测试：

```bash
python3 -m pytest worker_ipc/tests -q
```

SAM3 worker 测试：

```bash
/opt/conda/bin/python -m pytest sam3worker/tests
```

`sam3dworker` 测试命令仍然存在，但它不属于这个分支的主流程：

```bash
/opt/conda/bin/python -m pytest sam3dworker/tests
```

## 说明

- `sam3worker/` 和 `sam3dworker/` 会在运行时把仓库里的 `worker_ipc/` 路径插入 `sys.path`
- 所以按仓库约定方式启动时，`worker_ipc` 不需要单独 `pip install`
- 不要提交模型权重、运行输出目录或其他机器相关文件
