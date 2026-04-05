# SAM3D Worker Inference Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `sam3dworker` from pointmap-only reconstruction to actual upstream `SAM3D-object` inference with normalized pose output and optional artifact export.

**Architecture:** Keep request parsing, pointmap generation, output normalization, and artifact export inside `sam3dworker/service.py`, with lazy upstream inference initialization so the worker can start without immediately loading the heavy SAM3D pipeline. Reuse the existing `worker_ipc` shell and drive the implementation with tests that mock the upstream inference object.

**Tech Stack:** Python 3, `numpy`, `PIL`, `pytest`, lazy `torch` and upstream `SAM3D-object`

---

## File Structure

- Modify: `sam3dworker/service.py`
- Modify: `sam3dworker/README.md`
- Create: `sam3dworker/tests/test_service.py`

### Task 1: Lock The Reconstruct Success Payload

**Files:**
- Modify: `sam3dworker/tests/test_service.py`
- Modify: `sam3dworker/service.py`

- [ ] **Step 1: Write the failing success-path test**

```python
def test_reconstruct_returns_pose_and_requested_artifacts(...):
    ...
```

- [ ] **Step 2: Implement minimal upstream inference wrapper and payload normalization**

```python
output = _run_sam3d_inference(...)
return {
    "worker": "sam3d",
    "pose": {...},
    "artifacts": {...},
}
```

### Task 2: Document The New Boundary

**Files:**
- Modify: `sam3dworker/README.md`

- [ ] **Step 1: Update describe and reconstruct sections**

```markdown
- 当前实现已接入真实 SAM3D 推理
- pointmap 仍由 worker 内部生成
- `artifact_types` 控制高斯和 mesh 导出
```
