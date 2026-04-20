# isaacsim_world_position

This folder is a standalone geometry utility. It does not depend on SAM3D pose.

It estimates an object position directly from:

- `mask`
- `depth`
- camera intrinsics
- camera world pose from IsaacSim

The script returns:

- `position_camera_worker_xyz_m`: the current repo camera convention `[x_left, y_up, z_forward]`
- `position_camera_forward_left_up_m`: the Isaac camera-local convention used here
- `position_world_xyz_m`: the estimated IsaacSim world-space position

## Usage

From the repo root:

```bash
python isaacsim_world_position/estimate_world_position.py \
  isaacsim_world_position/inputs/red_cube_0.request.json
```

You can also write the output to a file:

```bash
python isaacsim_world_position/estimate_world_position.py \
  isaacsim_world_position/inputs/blue_cube_0.request.json \
  --output isaacsim_world_position/blue_cube_0.result.json
```

Estimator choices:

- `median` (default)
- `centroid`
- `bbox_center`

Example:

```bash
python isaacsim_world_position/estimate_world_position.py \
  isaacsim_world_position/inputs/red_cube_0.request.json \
  --estimator bbox_center
```

## Request Format

Each request JSON must contain:

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
  "camera_world_position_xyz_m": [0.0, 3.3, 3.3],
  "camera_world_quaternion_wxyz": [0.6830127, 0.1830127, 0.1830127, -0.6830127]
}
```

The sample request files in `inputs/` already contain working values.

## Formula

Backprojection from masked depth:

```text
X_right = (u - cx) * Z / fx
Y_down  = (v - cy) * Z / fy
Z       = depth
```

Repo camera convention:

```text
x_left    = -X_right
y_up      = -Y_down
z_forward =  Z
```

World transform used by this tool:

```text
p_camera_forward_left_up = [z_forward, x_left, y_up]
p_world = t_camera_world + R(q_camera_world_wxyz) @ p_camera_forward_left_up
```

## Caveat

This script estimates a single position from the visible masked depth points. That means:

- it does not estimate a full 6DoF pose
- it does not know the hidden back side of the object
- the result is usually a visible-surface center, not a guaranteed object-center ground truth

For symmetric objects such as cubes, this is often more stable than SAM3D pose, but it is still only a geometry estimate from one view.


  下一步如果要把 world 坐标彻底对准，不需要再猜公式了，只需要一份“同一次 get_camera_info() 返回的 camera payload +
  同一次 get_table_env_objects_info() 返回的 objects payload + 当次保存的 RGB/depth artifact”。只要这三样是同一 run
  里的，我就能把独立工具改成可靠版本。