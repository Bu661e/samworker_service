# SAM3D-object 使用指南

本文档面向当前仓库里的 `sam3dworker` 开发，整理 `SAM3D-object` 仓库的作用、安装方式、入口代码、输入输出约束，以及接入我们感知服务时应该怎么用。

## 1. 本地落位信息

- 上游仓库：`https://github.com/facebookresearch/sam-3d-objects`
- 本地路径：`third_party/SAM3D-object/`
- 当前仓库策略：这个目录只作为本地外部代码 checkout 使用，默认由 Git 忽略，不提交到主仓库
- 当前本地快照提交：`81a8237`
- 当前目录体积：约 `495MB`

这个仓库目前是一个“研究/推理仓库”，不是现成的 HTTP 服务项目。它公开的是 Python 推理入口和 notebook 示例，不是 REST API。

## 2. 这个仓库到底做什么

按官方 README 和代码实现，`SAM3D-object` 的核心能力是：

- 输入一张单图
- 再给它一个目标物体的 2D mask
- 输出该物体的 3D 重建结果

输出内容不只是一个网格，还包括：

- 物体位姿
  - `rotation`
  - `translation`
  - `scale`
- 3D 形状表示
  - `gaussian` / `gs`
  - `mesh`
  - `glb`
- 推理过程中使用的点图
  - `pointmap`
  - `pointmap_colors`

要注意两件事：

1. 它不是 2D 分割模型，不负责从自然语言里直接找“瓶子”“杯子”。
2. 它也不是多物体端到端系统。多物体示例本质上是“每个 mask 单独跑一次，再把多个结果合并”。

也就是说，在我们的 `sam3dworker` 体系里，更合理的职责分工是：

- `ultralytics` 的 SAM3 负责产生 2D mask
- `SAM3D-object` 负责把单个 mask 对应的目标重建成 3D

## 3. 仓库结构怎么读

建议优先看下面这些文件：

- `README.md`
  - 上游项目简介和最小示例
- `doc/setup.md`
  - 官方安装步骤
- `demo.py`
  - 最小可运行脚本
- `notebook/inference.py`
  - notebook 对外暴露的推理包装层
- `sam3d_objects/pipeline/inference_pipeline_pointmap.py`
  - 当前单图 + point map 推理主流程
- `sam3d_objects/pipeline/inference_pipeline.py`
  - 通用推理管线基类，负责模型装载、解码和部分后处理
- `requirements.txt`
  - 主依赖
- `requirements.p3d.txt`
  - `pytorch3d` 和 `flash_attn`
- `requirements.inference.txt`
  - 推理时还要补的依赖
- `environments/default.yml`
  - 官方推荐的 Conda/Mamba 环境

把这几个文件串起来以后，主链路其实很清楚：

1. 加载 `checkpoints/hf/pipeline.yaml`
2. 通过 Hydra 实例化 `InferencePipelinePointMap`
3. 输入 RGB 图 + 单物体 mask
4. 在当前项目里由上游感知层基于 `depth + intrinsics` 先生成 `pointmap`
5. 做 sparse structure 采样和 SLAT 解码
6. 输出 Gaussian / mesh / GLB / pose 等结果

## 4. 运行前提

官方 `doc/setup.md` 写得很明确，这个仓库的推理前提比较重：

- Linux 64-bit
- NVIDIA GPU
- 至少 `32GB` 显存
- Python `3.11`
- CUDA `12.1`

从环境文件和 requirements 也能看出它对 CUDA 版本非常敏感，核心组合大致是：

- Python `3.11.0`
- CUDA `12.1`
- `pytorch3d` 来自 GitHub 指定提交
- `flash_attn==2.8.3`
- `kaolin==0.17.0`
- `gsplat`
- `MoGe`

因此不建议在当前主仓库的通用虚拟环境里直接“顺手 pip install 一把”。更现实的做法是把它当成单独的感知推理环境。

## 5. 安装步骤

### 5.1 创建环境

官方推荐的是 `mamba`：

```bash
cd samworker_service/third_party/SAM3D-object

mamba env create -f environments/default.yml
mamba activate sam3d-objects
```

如果你只能用 `conda`，上游文档说可以直接把命令里的 `mamba` 换成 `conda`。

### 5.2 安装主依赖和 PyTorch3D

```bash
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

pip install -e '.[dev]'
pip install -e '.[p3d]'
```

这里分成两步不是随便写的。上游文档明确说明：`pytorch3d` 对 `pytorch` 的依赖链有问题，所以他们才要求单独跑 `.[p3d]`。

### 5.3 安装推理依赖

```bash
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
```

这一层主要补：

- `kaolin`
- `gsplat`
- `seaborn`
- `gradio`

### 5.4 打官方要求的 Hydra 补丁

```bash
./patching/hydra
```

这个脚本会：

- 检查你安装的 `hydra` 版本是否是 `1.3.2`
- 直接覆盖站点包里的 `hydra/core/utils.py`

所以这是一个“必须知道自己在做什么”的补丁步骤。没有这一步，后面的 Hydra 配置实例化可能会出问题。

### 5.5 下载权重

`SAM3D-object` 权重不在仓库里，官方要求先去 Hugging Face 申请访问权限：

- 模型地址：`https://huggingface.co/facebook/sam-3d-objects`

拿到权限后：

```bash
pip install 'huggingface-hub[cli]<1.0'

hf auth login

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects

mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

下载完成后，最关键的是这个路径必须存在：

```bash
checkpoints/hf/pipeline.yaml
```

因为 `demo.py` 和 notebook 都是按这个固定路径加载配置的。

在当前机器上的固定约定是：

- 权重真实目录：`/root/hf`
- 仓库内使用的入口路径：`/root/robot_task/samworker_service/third_party/SAM3D-object/checkpoints/hf`
- 这个入口路径应是一个软链接，指向 `/root/hf`

如果软链接丢失，可执行：

```bash
ln -s /root/hf /root/robot_task/samworker_service/third_party/SAM3D-object/checkpoints/hf
```

## 6. 最小可运行方式

### 6.1 直接跑 demo

一定要在仓库根目录下执行，因为 `demo.py` 用的是相对路径：

```bash
cd samworker_service/third_party/SAM3D-object
python demo.py
```

成功时它会：

- 加载 `checkpoints/hf/pipeline.yaml`
- 读取 `notebook/images/.../image.png`
- 读取对应索引的单个 mask
- 跑一次 3D 重建
- 导出 `splat.ply`

### 6.2 用 Python 调

官方 README 里最小示例的核心逻辑就是下面这样：

```python
import sys

sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

config_path = "checkpoints/hf/pipeline.yaml"
inference = Inference(config_path, compile=False)

image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

output = inference(image, mask, seed=42)
output["gs"].save_ply("splat.ply")
```

## 7. 对外推理接口怎么理解

当前最值得复用的公开入口不是 `demo.py`，而是：

- `notebook/inference.py` 里的 `Inference` 类

它的初始化方式：

```python
inference = Inference(config_path, compile=False)
```

这里做了几件重要的事情：

- 读取 `pipeline.yaml`
- 强制把 `rendering_engine` 改成 `pytorch3d`
  - 这样就绕开了 `nvdiffrast`
- 可以通过 `compile=True` 打开 `torch.compile`
- 用 Hydra 实例化 `InferencePipelinePointMap`

### 7.1 `Inference.__call__` 的真实输入

表面签名是：

```python
output = inference(image, mask, seed=42, pointmap=None)
```

其中：

- `image`
  - `PIL.Image` 或 `numpy.ndarray`
  - 最终会被转成 `np.uint8`
- `mask`
  - 建议传单通道 2D mask
  - `bool`、0/1、0/255 都可以，但最终最好保证语义是“目标区域为真”
- `seed`
  - 控制采样随机性
- `pointmap`
  - 可选
  - 对上游仓库本身来说可以直接传
  - 但在当前项目里，这个参数应由上游感知层内部准备，不对决策协议暴露

有一个很容易踩坑的点：

- `Inference.__call__` 虽然把 `mask` 标成了 Optional
- 但它内部会直接执行 `merge_mask_to_rgba(image, mask)`
- 这一步默认会对 `mask` 做 `astype`

所以按当前代码，`mask` 在实践里是“必填”，不能当成真的可选。

### 7.2 它对图像和 mask 做了什么

`notebook/inference.py` 会把你的 `mask` 写进 alpha channel：

- 取 `image[..., :3]`
- 把 mask 扩成单通道
- 拼成 RGBA

所以可以把它理解成：

- 你传入的是“RGB + 单目标 mask”
- 模型真正接收的是“mask 嵌进 alpha 通道的 RGBA”

### 7.3 没有外部深度时会发生什么

`InferencePipelinePointMap.compute_pointmap()` 里可以看到：

- 如果 `pointmap is None`
  - 会调用内部 `depth_model`
  - 当前依赖链里这个角色主要由 `MoGe` 承担
- 如果你传了 `pointmap`
  - 它会直接使用你提供的点图
  - 尺寸不一致时会做 resize
  - 如果没有相机内参，会尝试从点图反推内参

对我们的 `sam3dworker` 集成来说，这意味着：

- 当前项目里更合理的做法是：
  - 让 `llm_decision_making` 只上传 RGB、深度图和相机内参
  - 由上游感知层内部把深度图转换成 `pointmap`
  - 再把这个内部 `pointmap` 传给 `SAM3D-object`
- 不要默认让 `SAM3D-object` 自己从 RGB 估计
  - 这样对机器人坐标系和实际尺度更友好

## 8. 实际输出字典里有什么

这里要先说明一个容易误解的点：

- 上游公开接口 `Inference.__call__()` 没有再包一层 `results`
- 它是直接 `return self._pipeline.run(...)`
- 所以你最终拿到的是一个“顶层输出字典”，不是 `{"results": {...}}`

也就是说，代码里应该这样取值：

```python
output = inference(image, mask, seed=42, pointmap=pointmap)
rotation = output["rotation"]
translation = output["translation"]
```

而不是：

```python
output["results"]["rotation"]
```

### 8.1 输出字典是怎么拼出来的

`InferencePipelinePointMap.run()` 的默认返回形式是：

```python
return {
    **ss_return_dict,
    **outputs,
    "pointmap": ...,
    "pointmap_colors": ...,
}
```

所以最终输出本质上由 3 部分组成：

1. `ss_return_dict`
   - 稀疏结构采样和 pose 解码阶段产出的字段
2. `outputs`
   - mesh / gaussian 解码和后处理产出的字段
3. `pointmap` 与 `pointmap_colors`
   - 供可视化和后续处理使用的点图信息

### 8.2 默认 public API 下最常见的键

按 `notebook/inference.py` 当前默认调用参数：

- `stage1_only=False`
- `with_mesh_postprocess=False`
- `with_texture_baking=False`
- `with_layout_postprocess=False`
- `use_vertex_color=True`

最常见的顶层输出字典可以理解成：

```python
{
    "shape": ...,
    "coords_original": ...,
    "coords": ...,
    "downsample_factor": ...,
    "translation": ...,
    "rotation": ...,
    "scale": ...,
    "mesh": ...,
    "gaussian": ...,
    "glb": ...,
    "gs": ...,
    "pointmap": ...,
    "pointmap_colors": ...,
}
```

其中各字段含义如下：

- `shape`
  - 稀疏结构阶段的 latent 输出
  - 更偏模型内部中间结果
- `coords_original`
  - 稀疏体素坐标，未下采样版本
- `coords`
  - 稀疏体素坐标，下采样后版本
- `downsample_factor`
  - 稀疏结构下采样因子
- `translation`
  - 物体平移
  - 原始类型通常是 `torch.Tensor`
  - 形状通常接近 `(1, 3)`
- `rotation`
  - 物体旋转四元数
  - 原始类型通常是 `torch.Tensor`
  - 顺序是 `wxyz`
  - 形状通常接近 `(1, 4)`
- `scale`
  - 物体缩放
  - 原始类型通常是 `torch.Tensor`
  - 当前上游返回形状通常接近 `(1, 3)`
- `mesh`
  - mesh decoder 的原始输出
  - 不是文件路径
- `gaussian`
  - Gaussian decoder 的原始输出
  - 通常是列表或容器对象
  - 代码里通常按 `gaussian[0]` 取第一个对象
- `glb`
  - 由 mesh 和 gaussian 后处理得到的 3D 对象
  - 当前实现里实际是 `trimesh.Trimesh`
  - 它也不是磁盘文件路径
- `gs`
  - `gaussian[0]` 的便捷别名
- `pointmap`
  - 下采样后的点图
  - 形状是 `H x W x 3`
- `pointmap_colors`
  - 与下采样点图对齐的颜色图
  - 形状也是 `H x W x 3`

### 8.3 哪些字段是条件返回

不是所有场景都会返回完全相同的键。

#### `stage1_only=True`

如果只跑第一阶段，不解码 mesh / gaussian，那么返回会变成：

```python
{
    ...ss_return_dict,
    "voxel": ...,
    "pointmap": ...,
    "pointmap_colors": ...,
}
```

这时通常不会有：

- `mesh`
- `gaussian`
- `glb`
- `gs`

#### `with_layout_postprocess=True`

如果开启布局后处理，`ss_return_dict` 还可能被补充或覆盖这些字段：

- `iou`
- `iou_before_optim`
- `optim_accepted`
- 更新后的 `rotation`
- 更新后的 `translation`
- 更新后的 `scale`

也就是说，后处理打开以后，最终 `pose` 相关字段不一定还是第一次解码出来的原始值。

#### `estimate_plane=True`

如果走地面平面估计的特殊分支，返回会缩成：

```python
{
    "glb": ...,
    "translation": ...,
    "scale": ...,
    "rotation": ...,
}
```

这个分支更像一个特例，不是常规单物体重建主路径。

### 8.4 为什么不能把这个输出字典直接透传给 `sam3dworker`

上游这个输出字典很适合 notebook 和 Python 内部调试，但不适合直接作为 IPC JSON 返回，原因很具体：

- `mesh` 不是 JSON
- `gaussian` / `gs` 不是 JSON
- `glb` 也不是文件路径，而是内存里的 3D 对象
- `translation` / `rotation` / `scale` 默认还是 tensor
- `pointmap` 和 `pointmap_colors` 体积可能很大

所以接进 `sam3dworker` 时，不应该原样透传整个 `output`，而应该做一次标准化，只保留适合协议层暴露的内容，例如：

- `rotation`
- `translation`
- `scale`
- `pointmap_path`
- 请求要求导出的 `gaussian` 或 `mesh` 文件路径
- `label`
- 必要的调试字段

### 8.5 `translation` / `rotation` / `scale` 分别是什么意思

这 3 个字段最好放在同一个公式里理解：

```text
T(x) = s * R(q) * x + t
```

也就是把“物体局部坐标系里的点”变换到“当前场景 / 相机坐标系里”。

- `translation`
  - 3 元向量
  - 表示物体局部坐标原点在场景 / 相机坐标系里的位置
  - 不是 2D 像素平移
- `rotation`
  - 4 元四元数
  - 顺序是 `wxyz`
  - 表示物体局部坐标系相对场景 / 相机坐标系的朝向
  - 不是欧拉角
- `scale`
  - 3 元向量
  - 表示 canonical 物体形状缩放到当前实例尺寸时的尺度
  - 当前上游代码默认把它当作各向同性 scale 处理

上游可视化代码里也是把这 3 个量一起组合成一个 `l2c` 变换，再去变换物体点云，所以单独看某一个字段往往不如把它们作为一组 pose 来理解。

### 8.6 为什么 `scale` 里 3 个数字常常是一样的

这不是偶然，也不是打印格式问题，而是当前上游代码的明确设计。

默认推理路径里，pose decoder 最后返回 `scale` 时，写的是：

```python
pose_instance_dict["instance_scale_l2c"].squeeze(0).mean(-1, keepdim=True).expand(1, 3)
```

也就是：

1. 先取原始 scale 的 3 个分量
2. 求一个均值
3. 再把这个均值扩成 `[s, s, s]`

所以默认情况下你看到的 `scale` 本来就会变成三个相同的数字。

即使打开布局后处理，`InferencePipelinePointMap.refine_scale()` 里也会再次把 3 个通道取平均，再写回 3 个相同值。换句话说，当前 `SAM3D-object` 的默认思路不是输出各向异性缩放，而是把 scale 强制收敛成“统一尺度”。

这对我们做 `sam3dworker` 有两个直接结论：

- 协议层仍然可以保留 `scale: [sx, sy, sz]` 的 3 元数组形式
- 但在当前上游实现里，大多数情况下你会看到它接近 `[s, s, s]`

如果后面你真的需要“长宽高分别不同”的各向异性 scale，就不能直接沿用现在这段上游返回逻辑，得自己改它的 pose decoder 或后处理策略。

## 9. 单目标和多目标怎么跑

### 9.1 单目标

单目标是最自然的使用方式：

```python
output = inference(image, single_mask, seed=42, pointmap=pointmap)
```

适合我们服务里“对某个指定目标做一次 3D 重建”。

### 9.2 多目标

官方 notebook 的多目标做法不是一次性端到端推理，而是：

```python
outputs = [inference(image, mask, seed=42) for mask in masks]
scene_gs = make_scene(*outputs)
```

也就是：

1. 每个 mask 单独跑一次
2. 再把多个 3D 结果合到一个 scene 里

这对我们很重要，因为它说明服务端最合适的设计不是“整图一次全搞定”，而是：

- 先产生多个目标 mask
- 再逐个 mask 跑 `SAM3D-object`
- 每个目标保留自己的 2D/3D 对应关系

## 10. 适合接进当前 `sam3dworker` 的方式

推荐把 `SAM3D-object` 当成“单目标 3D 重建引擎”来包一层，而不是直接照搬 notebook。

### 10.1 推荐服务链路

建议链路如下：

1. 输入原始 RGB 图、深度图、相机内参、任务文本
2. 用 `ultralytics` 的 SAM3 找到 2D mask
3. 在感知层内部把深度图转换成 `pointmap`
4. 对每个 mask 单独调用 `SAM3D-object`
5. 提取并标准化 3D 输出
6. 组装成 `sam3dworker` 自己的响应 schema

### 10.2 服务层推荐保留的映射

每个实例都建议保留这些字段：

- `instance_id`
- `label`
- `source_mask_id`
- `mask_file`
- `translation_m`
- `rotation_wxyz`
- `scale_m`
- 导出的 `ply` / `glb` 路径

这样后面 `llm_decision_making` 才能知道：

- 这个 3D 结果是哪个 2D mask 产生的
- 它对应哪个语义标签
- 如果结果异常，应该回看哪个 mask

### 10.3 不建议直接照搬 notebook 的部分

不建议直接把下面这些 notebook 行为照搬进服务：

- `sys.path.append("notebook")`
- 用相对路径加载样例图
- 直接把 3D 结果导出到当前工作目录
- 把可视化逻辑和推理逻辑耦在一起

更合理的服务层做法是：

- 把 `notebook/inference.py` 里的 `Inference` 视为原型
- 抽成 `sam3dworker` 自己的适配层
- 明确输入是：
  - RGB
  - mask
  - depth
  - intrinsics
  - label
  - seed
- 在适配层内部生成 `pointmap`
- 明确输出是结构化响应，而不是 notebook 对象

### 10.4 和我们任务的直接关系

如果你准备把 `ultralytics` SAM3 和 `SAM3D-object` 串起来，最自然的对应方式是：

- `ultralytics` SAM3
  - 解决“图里哪些区域是 bottle / cup / block”
- `SAM3D-object`
  - 解决“这个区域对应的 3D 形状和位姿是什么”

这两层职责不要混。

## 11. 已知坑和实现风险

### 11.1 显存和环境要求重

这是当前最大风险。官方直接写了至少 `32GB` 显存，说明它不是轻量依赖。

### 11.2 权重受 Hugging Face 访问控制

没有权限就无法真正跑起来，所以开发和 CI 都要提前考虑：

- 权重是否能合法下载
- 权重存放在哪里
- 服务启动时是否延迟加载

### 11.3 当前入口更像研究代码，不像服务代码

具体体现在：

- 入口在 `notebook/inference.py`
- 依赖相对路径
- 可视化和推理耦合
- 很多输出对象不适合直接序列化

所以接入时应该做“适配层”，不要直接把 notebook 代码当服务接口。

### 11.4 `mask` 在公开包装层里实际上是必填

这是一个代码层面的细节坑，后续如果你要把它正式服务化，建议第一步就把这个接口签名和输入校验改干净。

### 11.5 `demo.py` 的注释和实际行为不完全一致

`demo.py` 里有一句注释写的是“RGBA only, mask is embedded in the alpha channel”，但从 `Inference.__call__` 的实现看，公开用法其实是：

- 传 `image`
- 再单独传 `mask`
- 它内部自己把 mask 写进 alpha

所以接服务时不要被这个注释误导。

## 12. 我对当前仓库接入的建议

如果下一步要真正实现 `sam3dworker`，建议按这个优先级来：

1. 先把 `SAM3D-object` 包成一个最小 Python 适配器
2. 明确只支持“单目标 mask -> 单目标 3D 输出”
3. 再在服务层外面加“多 mask 调度”
4. 最后再考虑批量并发、文件落盘、缓存和 HTTP schema

不要一上来就把上游 notebook 逻辑直接塞进 HTTP handler。

## 13. 参考链接

- 上游仓库：`https://github.com/facebookresearch/sam-3d-objects`
- 官方 README：`https://github.com/facebookresearch/sam-3d-objects/blob/main/README.md`
- 官方安装文档：`https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md`
- 论文页面：`https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/`
- Hugging Face 权重：`https://huggingface.co/facebook/sam-3d-objects`
