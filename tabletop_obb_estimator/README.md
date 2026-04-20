# tabletop_obb_estimator

这个目录提供一个独立的几何基线工具，用来从下面这些输入估计物体的三维包围盒信息：

- `mask`
- `depth`
- 相机内参
- IsaacSim 相机世界位姿

当前输出包括：

- `position_world_xyz_m`
- `rotation_quaternion_wxyz`
- `rotation_matrix_world_from_obb`
- `size_xyz_m`
- `obb_corners_world_xyz_m`

它不依赖 `sam3d` 自带的 `pose`。

另外，这个目录里还提供了一个单独的 mesh OBB 工具：

- [estimate_mesh_obb.py](/root/samworker_service/tabletop_obb_estimator/estimate_mesh_obb.py)

它可以直接从 `mesh.glb` 这类三维模型文件计算 OBB。

## 为什么要单独做这个工具

如果目标是获得更稳的：

- `position`
- `rotation`
- `长宽高`

更实用的分工通常是：

- `sam3`：负责给高质量 `mask`
- `depth + intrinsics`：负责真实几何
- `sam3d`：负责 `mesh / gaussian / shape completion`
- 最终 `position / rotation / size_xyz`：优先从几何或 mesh 的 OBB 来估计，而不是直接信 `sam3d pose`

这个目录先把“几何 + OBB”这部分独立实现出来。

## 脚本做了什么

脚本会按下面步骤工作：

1. 读取 `mask` 和 `depth`
2. 把 mask 内有效深度反投影成点云
3. 把点云从相机坐标转换到 IsaacSim 世界坐标
4. 拟合一个 OBB
5. 输出：
   - `position_world_xyz_m`
   - `rotation_quaternion_wxyz`
   - `rotation_matrix_world_from_obb`
   - `size_xyz_m`
   - `obb_corners_world_xyz_m`

## 如何运行

在仓库根目录执行：

```bash
python tabletop_obb_estimator/estimate_tabletop_obb.py \
  tabletop_obb_estimator/inputs/red_cube_0.request.json
```

如果要把结果保存成文件：

```bash
python tabletop_obb_estimator/estimate_tabletop_obb.py \
  tabletop_obb_estimator/inputs/blue_cube_0.request.json \
  --output tabletop_obb_estimator/blue_cube_0.obb.json
```

## 直接对 mesh.glb 计算 OBB

如果你已经有 `sam3d` 输出的 `mesh.glb`，可以直接运行：

```bash
/root/autodl-tmp/conda/envs/sam3d-objects/bin/python \
  tabletop_obb_estimator/estimate_mesh_obb.py \
  tabletop_obb_estimator/inputs/mesh.glb
```

保存到文件：

```bash
/root/autodl-tmp/conda/envs/sam3d-objects/bin/python \
  tabletop_obb_estimator/estimate_mesh_obb.py \
  tabletop_obb_estimator/inputs/mesh.glb \
  --output tabletop_obb_estimator/mesh.obb.json
```

说明：

- 这一步只依赖 mesh，不依赖 RGB、depth、mask
- 适合直接看 `sam3d mesh` 本身的三维尺寸与主轴
- 如果 mesh 已经在物体局部坐标系，输出就是局部 OBB
- 如果 mesh 已经被放到世界坐标系，输出就是世界系 OBB

当前目录里已经放了一份示例文件：

- `tabletop_obb_estimator/inputs/mesh.glb`

实际运行命令：

```bash
/root/autodl-tmp/conda/envs/sam3d-objects/bin/python \
  tabletop_obb_estimator/estimate_mesh_obb.py \
  tabletop_obb_estimator/inputs/mesh.glb \
  --output tabletop_obb_estimator/mesh.obb.json
```

当前示例结果摘要：

- `obb_center_xyz = [-0.004877, -0.007015, -0.000372]`
- `obb_extents_xyz = [0.466310, 0.471138, 0.478506]`
- `obb_rotation_quaternion_wxyz = [0.707082, 0.000022, 0.000114, 0.707132]`

结果文件位置：

- `tabletop_obb_estimator/mesh.obb.json`

## 模式说明

当前支持两种模式：

- `tabletop`
  - 默认模式
  - 假设物体放在桌面上
  - 会用 `world_up_axis_xyz` 约束包围盒保持竖直
- `pca3d`
  - 不加桌面约束
  - 直接做三维 PCA OBB
  - 对一般场景更通用，但对桌面物体稳定性通常不如 `tabletop`

示例：

```bash
python tabletop_obb_estimator/estimate_tabletop_obb.py \
  tabletop_obb_estimator/inputs/red_cube_0.request.json \
  --mode pca3d
```

## 请求 JSON 格式

示例：

```json
{
  "label": "red_cube_0",
  "depth_path": "../../sam3dworker/tests/inputs/emp_default_tableoverview/example.npy",
  "mask_path": "../../sam3dworker/tests/inputs/emp_default_tableoverview/red_cube_0.png",
  "intrinsics": {
    "fx": 533.3333740234375,
    "fy": 533.3333740234375,
    "cx": 320.0,
    "cy": 320.0
  },
  "camera_world_position_xyz_m": [0.0, 3.299999952316284, 3.299999952316284],
  "camera_world_quaternion_wxyz": [0.6830127, 0.1830127, 0.1830127, -0.6830127],
  "world_up_axis_xyz": [0.0, 0.0, 1.0]
}
```

可选字段：

- `expected_world_position_xyz_m`
  - 如果提供，脚本会额外输出：
    - `world_position_error_xyz_m`
    - `world_position_error_norm_m`

## 坐标约定

这个工具假设相机位姿来自 IsaacSim：

```python
camera.get_world_pose(camera_axes="world")
```

在 `camera_axes="world"` 下，这里采用的相机局部轴定义是：

- `+X`：forward
- `+Y`：left
- `+Z`：up

从图像像素反投影到相机局部坐标：

```text
forward = depth
left    = (cx - u) * depth / fx
up      = (cy - v) * depth / fy
```

再从相机坐标转世界坐标：

```text
p_world = t_camera_world + R(q_camera_world_wxyz) @ p_camera_local
```

## OBB 是什么

`OBB` 是 `Oriented Bounding Box`，中文可以理解成“有方向的包围盒”。

和普通 `AABB` 不同，`OBB` 可以跟着物体一起旋转，所以天然可以提供：

- 盒子中心 -> `position`
- 盒子方向 -> `rotation`
- 盒子三条边长度 -> `size_xyz`

所以如果想统一输出：

- `position`
- `rotation`
- `长宽高`

OBB 是一个很自然的表达方式。

## 推荐的融合方式

如果后面你想把 `sam3 + sam3d` 融合成更稳定的结果，可以按下面的思路做：

1. `sam3` 给出高质量目标 `mask`
2. `depth` 给出可见点云
3. 这个工具先给出一个基于真实几何的 world-space OBB 基线
4. 如果 `sam3d mesh` 质量够好，再从 mesh 上再算一个 OBB 做对比：
   - `position` 优先参考 depth 几何
   - `size_xyz` 可以优先参考 mesh OBB
   - `rotation` 结合桌面约束、重力方向和 mesh 主轴共同决定

简单说就是：

- `sam3` 管分割
- `depth` 管位置几何
- `sam3d` 管完整形状
- 最终位姿和尺寸用 OBB 统一表达

## 已知限制

- 单视角 depth 只能看到可见表面
- 背面厚度可能会被低估
- 对称物体（比如 cube）本身存在 yaw 歧义
- 如果相机 pose、depth、mask、物体 GT 不是同一次采集，world 坐标误差没有比较意义
