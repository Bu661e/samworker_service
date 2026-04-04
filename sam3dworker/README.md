# sam3dworker

`sam3dworker` 负责当前仓库中面向 SAM3D 的 worker 代码。

## 作用范围

- SAM3D 请求的 worker 入口
- 指向 `third_party/` 下上游代码的轻量 service 层

## 目录与职责

- worker 目录：`sam3dworker/`
- 共享 IPC 包：`../worker_ipc/`
- 上游 SAM3D 代码快照：`../third_party/SAM3D-object/`

这个 worker 当前负责：

- 通过 `worker_ipc` 接收同步请求
- 分发与 SAM3D-object 相关的任务
- 消费 RGB、mask 和 pointmap 输入，执行 3D 重建

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

- 上游感知逻辑先把 `depth + intrinsics` 转成 `pointmap`
- `sam3dworker` 消费该 `pointmap`，然后调用 `../third_party/SAM3D-object/`

## 文档

- 上游使用说明：`SAM3D-object_使用指南.md`

## 当前状态

这个目录目前只包含 worker 骨架。真正的 SAM3D 实现仍然位于上游目录，后续会在这里继续完成封装。
