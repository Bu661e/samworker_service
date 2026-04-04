# sam3worker

`sam3worker` 负责当前仓库中面向 SAM3 的 worker 代码。

## 作用范围

- SAM3 请求的 worker 入口
- 指向 `third_party/` 下上游代码的轻量 service 层

## 目录与职责

- worker 目录：`sam3worker/`
- 共享 IPC 包：`../worker_ipc/`
- 上游 SAM3 代码快照：`../third_party/sam3-ultralytics/`

这个 worker 当前负责：

- 通过 `worker_ipc` 接收同步请求
- 分发与 SAM3 相关的任务
- 返回 2D mask 和 bbox 结果

## 运行环境

- 推荐 conda 环境：`sam3d-objects`
- 推荐解释器：`/root/autodl-tmp/conda/envs/sam3d-objects/bin/python`
- 如果直接使用 conda：`conda run -n sam3d-objects python ...`

## 权重与路径

- 实际 SAM3 权重文件：`/root/sam3.pt`
- 上游入口路径应当指向该文件：
  - `../third_party/sam3-ultralytics/sam3.pt -> /root/sam3.pt`
- 当前已知桥接脚本：
  - `../third_party/sam3-ultralytics/run_sam3_inference.py`

## 文档

- 上游使用说明：`ultralytics_SAM3_使用指南.md`

## 当前状态

这个目录目前只包含 worker 骨架。真正的 SAM3 实现仍然位于上游目录，后续会在这里继续完成封装。
