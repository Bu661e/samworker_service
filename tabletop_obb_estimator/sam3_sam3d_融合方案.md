# sam3 + sam3d 融合方案

这份文档只说明一种方案：**最终结果必须使用 `sam3d`**。

目标是输出单个物体的：

- `position`
- `rotation`
- `size_xyz`
- `obb`

并且这套结果不是直接照搬 `sam3d pose`，而是把 `sam3d` 作为**形状补全与三维几何约束的核心模块**来使用。

## 方案结论

推荐的单视角桌面场景方案是：

1. 用 `sam3` 做目标分割，得到高质量 `mask`
2. 用 `depth + intrinsics` 生成目标可见点云
3. 用 `sam3d` 生成目标 `mesh / gaussian`
4. 用 `sam3d` 产出的 `mesh` 与可见点云做几何对齐优化
5. 在**对齐后的 `sam3d mesh`** 上计算最终 `OBB`
6. 从这个 `OBB` 输出最终的：
   - `position`
   - `rotation`
   - `size_xyz`

一句话概括：

**`sam3` 管分割，`depth` 管可见几何，`sam3d` 管完整形状，最终结果以“对齐后的 sam3d mesh OBB”为准。**

## 为什么这套方案里必须使用 sam3d

如果只用 `mask + depth`，你只能拿到物体的**可见部分**。

这会带来两个问题：

1. 厚度通常不准
   - 单视角深度只能看到前表面
   - 背面不可见
   - 直接从可见点云算 `size_xyz`，厚度常常会偏小

2. 旋转通常不稳
   - 可见点云可能只有物体的一部分
   - 直接做 PCA / OBB，主轴容易受遮挡影响
   - 对长方体、瓶子、盒子这类物体尤其明显

`sam3d` 的作用就是补这个短板：

- 它可以从单视角输入补全成更完整的三维形状
- 它的 `mesh` 比“只看见的一层点云”更适合估计真实尺寸
- 它的形状主轴也比可见点云主轴更稳定

所以在这套方案里，`sam3d` 不是附带参考，而是**最终几何表达的主体**。

## 不直接使用 sam3d pose 的原因

虽然 `sam3d` 会输出 `pose.translation / pose.rotation / pose.scale`，但这条 `pose` 路径更像重建链路里的附带结果，不适合直接当高精度最终答案。

原因主要有三点：

1. `sam3d` 的强项是重建 `mesh / gs`
2. `pose` 常常更像“为了摆放重建结果而给的姿态”
3. 当前实测里，`sam3d pose` 的数值稳定性和精度都不够理想

所以这里的正确用法不是：

- 直接输出 `sam3d pose`

而是：

- **使用 `sam3d mesh`**
- 再用真实可见几何对这个 mesh 做约束和修正
- 最后从修正后的 mesh 上提取 OBB

## 详细流程

### 第一步：sam3 分割

输入：

- RGB 图
- 目标提示信息（点、框或已有目标 id）

输出：

- 单目标 `mask`

要求：

- `mask` 尽量紧贴物体轮廓
- 不要把桌面或邻近物体带进去

这一步的作用是给后面的所有几何处理提供干净目标区域。

### 第二步：由 depth 构造可见点云

输入：

- `mask`
- `depth`
- `fx, fy, cx, cy`

对 mask 内有效深度像素做反投影：

```text
forward = depth
left    = (cx - u) * depth / fx
up      = (cy - v) * depth / fy
```

如果相机位姿是通过：

```python
camera.get_world_pose(camera_axes="world")
```

取得的，那么再转到 IsaacSim 世界坐标：

```text
p_world = t_camera_world + R(q_camera_world_wxyz) @ p_camera_local
```

输出：

- 目标可见点云 `P_visible`

这一步得到的是**真实观测几何**，它对 `position` 很重要。

### 第三步：运行 sam3d 生成 mesh

输入：

- `RGB`
- `mask`
- `depth` 或由其生成的 pointmap

输出：

- `sam3d mesh`
- 可选：`gaussian`
- 可选：`sam3d pose` 作为初始化

这里的关键点是：

- `sam3d` 的主要任务是给出一个**完整一些的三维形状**
- 后续所有尺寸与主轴估计，都优先参考这个 mesh

### 第四步：mesh 与可见点云做对齐优化

这是这套方案最关键的一步。

输入：

- `sam3d mesh`
- 可见点云 `P_visible`
- 可选：`sam3d pose` 作为初始位姿
- 桌面法向或世界 `up` 方向

目标：

- 让 `sam3d mesh` 与真实可见点云对齐
- 同时保留 `sam3d` 对背面形状的补全能力

推荐的优化方式：

1. 初始对齐
   - 如果 `sam3d pose` 可用，就只把它当成初始化
   - 如果 `sam3d pose` 不可靠，就用可见点云中心给平移初值

2. 点到面 / 点到点对齐
   - 用 `P_visible` 和 mesh 做 ICP 或其他刚体配准

3. 桌面约束
   - 物体底面不能穿桌
   - 物体主竖直方向应当和世界 `up` 一致或接近一致

4. 轮廓 / 深度一致性约束
   - 把对齐后的 mesh 再投影回图像
   - 与原始 `mask / depth` 做一致性检查

这一步结束后，得到：

- `mesh_refined`

这是最终用于提取 OBB 的形状。

## 第五步：在 refined mesh 上计算 OBB

在 `mesh_refined` 上计算 `OBB`。

最终得到：

- `obb.center`
- `obb.rotation`
- `obb.size_xyz`
- `obb.corners`

并把它作为最终输出。

也就是说，在这套方案里：

- `position = obb.center`
- `rotation = obb.rotation`
- `size_xyz = obb.size_xyz`

## 最终输出定义

推荐统一成下面这组字段：

```json
{
  "position_world_xyz_m": [cx, cy, cz],
  "rotation_quaternion_wxyz": [w, x, y, z],
  "rotation_matrix_world_from_obb": [
    [r11, r12, r13],
    [r21, r22, r23],
    [r31, r32, r33]
  ],
  "size_xyz_m": [sx, sy, sz],
  "obb_corners_world_xyz_m": [
    [x1, y1, z1],
    [x2, y2, z2],
    [x3, y3, z3],
    [x4, y4, z4],
    [x5, y5, z5],
    [x6, y6, z6],
    [x7, y7, z7],
    [x8, y8, z8]
  ],
  "mesh_path": "...",
  "gaussian_path": "..."
}
```

## 这套方案里每个模块的职责

### sam3

职责：

- 提供高质量目标 `mask`

不负责：

- 最终三维姿态
- 最终尺寸

### depth

职责：

- 提供真实可见几何
- 给 `position` 和可见点云提供真实约束

不负责：

- 完整形状补全

### sam3d

职责：

- 生成完整三维形状
- 提供 `mesh / gaussian`
- 给 OBB 提供更完整的几何基础

在这套方案里，`sam3d` 必须参与，因为最终尺寸和旋转都要依赖它的三维形状。

## 为什么最终以 refined mesh OBB 为准

因为它同时结合了两类信息：

1. `depth` 的真实观测几何
2. `sam3d` 的完整形状补全

只用 `depth`：

- 可见部分真实
- 但缺背面

只用 `sam3d pose`：

- 有完整形状
- 但位姿精度不够稳

而“对齐后的 `sam3d mesh` + OBB”同时兼顾了：

- 真实位置约束
- 完整形状补全
- 稳定的尺寸估计

## 适用场景

这套方案最适合：

- 单视角桌面抓取
- 规则物体或近规则物体
- 有深度图
- 有相机内参
- 允许使用 `sam3d` 重建 mesh

## 已知限制

即使用了这套方案，仍然有几个要注意的点：

1. 对称物体有旋转歧义
   - 比如 cube，绕竖直轴的 yaw 可能不是唯一的

2. 如果 `sam3d mesh` 质量差，尺寸会被带偏
   - 所以必须做“mesh 与可见点云对齐”
   - 不能直接拿原始 mesh OBB 当最终结果

3. 如果 `mask` 不干净，整条链路都会受影响
   - `mask` 漏掉物体一部分
   - 或混入桌面、邻近物体
   - 都会导致点云和 mesh 对齐错误

## 最后结论

如果方案文档里**必须使用 `sam3d`**，并且目标是获得更合理的：

- `position`
- `rotation`
- `size_xyz`

那么最推荐的方法不是直接用 `sam3d pose`，而是：

**用 `sam3d` 生成完整 mesh，再用真实可见点云把这个 mesh 对齐到观测数据上，最后在对齐后的 mesh 上计算 OBB。**

这就是这份文档唯一推荐的方法。
