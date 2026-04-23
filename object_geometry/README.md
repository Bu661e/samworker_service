# object_geometry

This package contains geometry-only helpers used by the pipeline.

The main pipeline obtains a segmentation mask from RGB via `sam3worker`, then this package combines that mask with depth and camera intrinsics to build visible object geometry in the camera frame.

Current output convention:

- coordinate frame: camera
- axes: `x=left`, `y=up`, `z=forward`
- OBB source: visible masked depth points

Primary API:

```python
from object_geometry import estimate_masked_camera_obb
```

`estimate_masked_camera_obb(...)` loads a depth `.npy` file and mask image, backprojects valid masked pixels into a camera-space point cloud, estimates a PCA OBB, and returns center position, rotation, dimensions, corners, visible centroid, and point count.
