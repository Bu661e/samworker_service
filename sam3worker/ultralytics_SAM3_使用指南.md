# ultralytics SAM3 使用指南

本文档只保留当前 `sam3worker` 真正会用到的单图用法，视频相关接口已去掉。

## 1. 先看结论

在 `ultralytics` 里，当前最值得关注的是两条单图入口：

| 入口 | 适用场景 | 提示方式 |
| --- | --- | --- |
| `SAM("sam3.pt")` | 已知目标大致位置，想切出某个具体实例 | 点、框、mask，或无提示 |
| `SAM3SemanticPredictor` | 想按语义找出所有匹配实例 | 文本、示例框 |

可以直接这样理解：

- `SAM("sam3.pt")`
  - 找“这个位置上的这个物体”
- `SAM3SemanticPredictor`
  - 找“所有符合这个概念的物体”

## 2. 安装和权重准备

### 2.1 版本要求

- SAM3 需要 `ultralytics >= 8.3.237`

```bash
pip install -U ultralytics
```

### 2.2 权重不会自动下载

`sam3.pt` 不会像常规 YOLO 权重那样自动拉取，需要先从 Hugging Face 获取并手动下载：

- 模型页：`https://huggingface.co/facebook/sam3`
- 权重文件：`https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true`

建议代码里使用绝对路径：

```python
model_path = "/absolute/path/to/sam3.pt"
```

当前机器上的约定：

- 实际权重：`/root/sam3.pt`
- 仓库入口：`/root/samworker_service/third_party/sam3-ultralytics/sam3.pt`
- 仓库入口应是指向 `/root/sam3.pt` 的软链接

如果软链接丢失，可执行：

```bash
ln -s /root/sam3.pt /root/samworker_service/third_party/sam3-ultralytics/sam3.pt
```

### 2.3 `SimpleTokenizer` 报错

如果文本推理时报：

- `TypeError: 'SimpleTokenizer' object is not callable`

优先修复 `clip` 依赖：

```bash
pip uninstall clip -y
pip install git+https://github.com/ultralytics/CLIP.git
```

## 3. `SAM("sam3.pt")` 的用法

### 3.1 适合什么场景

这条接口更适合：

- 已经知道目标大概位置
- 用点、框或已有 mask 提示模型
- 目标是切出某个具体实例

不适合：

- 直接按文本找“所有 bottle”
- 直接做概念级的“找所有类似目标”

### 3.2 最小示例

```python
from ultralytics import SAM

model = SAM("/absolute/path/to/sam3.pt")

results = model.predict(
    source="path/to/image.jpg",
    points=[[900, 370]],
    labels=[1],
)

results = model.predict(
    source="path/to/image.jpg",
    bboxes=[[100, 150, 300, 400]],
)
```

调用时注意：

- `points` 使用原图像素坐标
- `labels` 里 `1` 是前景点，`0` 是背景点
- `bboxes` 使用 `xyxy`
- 只传一个点或一个框时，也建议用二维列表包起来

### 3.3 不传提示时

如果不传点、框、mask，`SAM("sam3.pt")` 会走全图自动分割：

```python
results = model.predict(source="path/to/image.jpg")
```

这通常不是 `sam3worker` 的首选，因为会返回较多 mask，后续 3D 处理成本更高。

## 4. `SAM3SemanticPredictor` 的用法

### 4.1 适合什么场景

这条接口是文本/概念分割的主入口，适合：

- 用文本找目标
- 用一个示例框找整图里相似实例
- 在同一张图上连续查询多个概念

### 4.2 文本提示示例

```python
from ultralytics.models.sam import SAM3SemanticPredictor

overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="/absolute/path/to/sam3.pt",
    half=True,
    save=False,
    verbose=False,
)

predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image("path/to/image.jpg")

results = predictor(text=["bottle"])
results = predictor(text=["blue bottle", "red cup"])
```

这里最关键的是：

- `set_image()` 会先缓存图像特征
- 同一张图可以多次换 `text` 查询
- 文本可以是名词，也可以是短语描述

### 4.3 示例框提示

```python
from ultralytics.models.sam import SAM3SemanticPredictor

predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image("path/to/image.jpg")

results = predictor(bboxes=[[480, 290, 590, 650]])
```

这条路径的语义不是“只切这个框里的物体”，而是：

- 把这个框作为视觉 exemplar
- 在整张图里找所有相似实例

### 4.4 输入输出注意点

- `set_image()` 可接图片路径或 `cv2.imread()` 读出的 `numpy.ndarray`
- `numpy.ndarray` 按 `cv2` 习惯处理，也就是 BGR
- `bboxes` 对外仍使用原图像素坐标的 `xyxy`
- 返回值通常是 `Results` 列表，常用字段有：
  - `results[0].masks.data`
  - `results[0].boxes.xyxy`
  - `results[0].boxes.conf`

## 5. 对当前 `sam3worker` 的推荐用法

### 5.1 自然语言目标

例如：

- `"pick the bottle"`
- `"move the blue block"`

推荐：

- 用 `SAM3SemanticPredictor`
- 直接传 `text=["bottle"]`、`text=["blue block"]`

### 5.2 已有点击或 bbox

例如：

- UI 上点选某个物体
- 上游模块已经给了一个 bbox

推荐：

- 用 `SAM("sam3.pt")`
- 用点或框做视觉提示

### 5.3 已有 exemplar 框

例如先框住一个已知杯子，再找整图里类似的杯子，推荐：

- 用 `SAM3SemanticPredictor(bboxes=...)`

### 5.4 与 `SAM3D-object` 的串联

推荐流程：

1. 用 `SAM3SemanticPredictor` 或 `SAM` 先得到 2D mask
2. 从 `results[0].masks.data` 逐个取实例 mask
3. 每个实例单独送入 `SAM3D-object`

不要指望 `SAM3D-object` 自己做文本分割。

## 6. 最小项目示例

```python
from ultralytics.models.sam import SAM3SemanticPredictor

overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="/absolute/path/to/sam3.pt",
    half=True,
    verbose=False,
)

predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image("/path/to/rgb.png")

results = predictor(text=["bottle"])

r = results[0]
masks = r.masks.data.cpu().numpy() if r.masks is not None else []
boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []

for idx, mask in enumerate(masks):
    # 后续把 mask 送入 SAM3D-object
    print(idx, boxes[idx])
```

## 7. 几个容易踩的坑

- `SAM("sam3.pt")` 不是文本概念分割入口
- 文本找目标要用 `SAM3SemanticPredictor`
- `sam3.pt` 不会自动下载
- `clip` 版本不对会导致文本推理报错
- 当前更适合同一张图多次查询，不适合图像 batch

## 8. 参考链接

- 官方文档：`https://docs.ultralytics.com/models/sam-3/`
- 模型源码：`https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/model.py`
- 预测器源码：`https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/predict.py`
- Hugging Face 权重页：`https://huggingface.co/facebook/sam3`
