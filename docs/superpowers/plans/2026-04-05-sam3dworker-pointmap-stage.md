# SAM3D Worker Pointmap Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first `sam3dworker` reconstruct stage that validates inputs, generates and saves `pointmap.npy`, and stops before calling upstream SAM3D inference.

**Architecture:** Keep the worker protocol surface in `sam3dworker/service.py` and keep pointmap generation as a focused internal helper inside the same module. Reuse the existing `worker_ipc` request/response shell in `sam3dworker/worker.py`, and add tests that lock in request validation, pointmap file output, and the explicit "not implemented yet" boundary after pointmap generation.

**Tech Stack:** Python 3, `numpy`, `pathlib`, `PIL`, `pytest`

---

## File Structure

- Modify: `sam3dworker/service.py`
- Modify: `sam3dworker/worker.py`
- Modify: `sam3dworker/README.md`
- Create: `sam3dworker/tests/test_service.py`
- Create: `sam3dworker/tests/test_worker.py`

### Task 1: Describe The First-Stage Reconstruct Behavior

**Files:**
- Create: `sam3dworker/tests/test_service.py`
- Modify: `sam3dworker/service.py`

- [ ] **Step 1: Write the failing tests for describe output and reconstruct validation**

```python
def test_describe_service_reports_supported_commands():
    assert describe_service()["supported_commands"] == ["ping", "describe", "reconstruct"]


def test_reconstruct_requires_pointmap_inputs_to_exist(tmp_path: Path):
    payload = {
        "image_path": str((tmp_path / "image.png").resolve()),
        "depth_path": str((tmp_path / "missing.npy").resolve()),
        "mask_path": str((tmp_path / "mask.png").resolve()),
        "output_dir": str((tmp_path / "outputs").resolve()),
        "fx": 100.0,
        "fy": 100.0,
        "cx": 10.0,
        "cy": 10.0,
        "label": "cube_0",
    }
    with pytest.raises(ValueError, match="depth_path does not exist"):
        handle_command("reconstruct", payload)
```

- [ ] **Step 2: Run the focused tests and verify they fail for missing reconstruct support**

Run:

```bash
python3 -m pytest sam3dworker/tests/test_service.py -q
```

Expected:

```text
FAIL ... unknown command: reconstruct
```

- [ ] **Step 3: Implement the describe metadata and reconstruct request parsing**

```python
SUPPORTED_COMMANDS = ["ping", "describe", "reconstruct"]

def handle_command(command: str, payload: dict[str, Any]) -> dict[str, Any]:
    if command == "reconstruct":
        return reconstruct(payload)
```

- [ ] **Step 4: Run the focused tests and verify validation behavior passes**

Run:

```bash
python3 -m pytest sam3dworker/tests/test_service.py -q
```

Expected:

```text
... passed
```

### Task 2: Generate And Save Pointmap Before Deferring Inference

**Files:**
- Modify: `sam3dworker/service.py`
- Create: `sam3dworker/tests/test_service.py`

- [ ] **Step 1: Write the failing test for pointmap output and deferred inference**

```python
def test_reconstruct_writes_pointmap_and_then_reports_not_implemented(
    tmp_path: Path,
):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    depth_path = tmp_path / "depth.npy"
    output_dir = tmp_path / "outputs"

    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(image_path)
    Image.fromarray(np.array([[255, 0], [255, 255]], dtype=np.uint8)).save(mask_path)
    np.save(depth_path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="sam3d inference not implemented yet"):
        handle_command(
            "reconstruct",
            {
                "image_path": str(image_path.resolve()),
                "depth_path": str(depth_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "fx": 100.0,
                "fy": 100.0,
                "cx": 0.5,
                "cy": 0.5,
                "label": "cube_0",
            },
        )

    pointmap = np.load(output_dir / "pointmap.npy")
    assert pointmap.shape == (2, 2, 3)
    assert pointmap.dtype == np.float32
```

- [ ] **Step 2: Run the focused test and verify it fails because pointmap generation is missing**

Run:

```bash
python3 -m pytest sam3dworker/tests/test_service.py::test_reconstruct_writes_pointmap_and_then_reports_not_implemented -q
```

Expected:

```text
FAIL ... pointmap.npy does not exist
```

- [ ] **Step 3: Implement minimal pointmap generation and explicit deferred inference error**

```python
pointmap = _build_pointmap_from_depth(depth, fx=..., fy=..., cx=..., cy=...)
np.save(output_dir / "pointmap.npy", pointmap.astype(np.float32))
raise ValueError("sam3d inference not implemented yet")
```

- [ ] **Step 4: Run the focused test and verify pointmap output now exists**

Run:

```bash
python3 -m pytest sam3dworker/tests/test_service.py::test_reconstruct_writes_pointmap_and_then_reports_not_implemented -q
```

Expected:

```text
1 passed
```

### Task 3: Keep The Worker Process Alive On Unexpected Errors

**Files:**
- Modify: `sam3dworker/worker.py`
- Create: `sam3dworker/tests/test_worker.py`

- [ ] **Step 1: Write the failing worker error-handling test**

```python
def test_handle_request_returns_error_response_for_runtime_error(monkeypatch) -> None:
    def boom(command: str, payload: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("sam3d crashed")

    monkeypatch.setattr(worker_module, "handle_command", boom)

    response = worker_module._handle_request(
        Request(request_id="req-1", command="reconstruct", payload={})
    )

    assert response.ok is False
    assert response.error == "sam3d crashed"
```

- [ ] **Step 2: Run the worker test and verify it fails without a runtime-error guard**

Run:

```bash
python3 -m pytest sam3dworker/tests/test_worker.py -q
```

Expected:

```text
FAIL ... RuntimeError: sam3d crashed
```

- [ ] **Step 3: Add the runtime-error guard in the worker request wrapper**

```python
def _handle_request(request: Request) -> Response:
    try:
        payload = handle_command(request.command, request.payload)
        return Response.success(request.request_id, payload)
    except ValueError as exc:
        return Response.error(request.request_id, str(exc))
    except Exception as exc:
        return Response.error(request.request_id, str(exc))
```

- [ ] **Step 4: Run the worker tests and verify the error response is returned instead**

Run:

```bash
python3 -m pytest sam3dworker/tests/test_worker.py -q
```

Expected:

```text
... passed
```

### Task 4: Update The Worker README To Match The Implemented Boundary

**Files:**
- Modify: `sam3dworker/README.md`

- [ ] **Step 1: Update the current status and reconstruct command section**

```markdown
- 当前第一阶段已实现到 pointmap 生成与落盘
- 真实 SAM3D-object 推理尚未接入
```

- [ ] **Step 2: Update the reconstruct error-handling section**

```markdown
- pointmap 生成成功后，当前实现会返回 `sam3d inference not implemented yet`
```

- [ ] **Step 3: Verify the README matches the implemented boundary**

Run:

```bash
rg -n "pointmap|not implemented yet|reconstruct" sam3dworker/README.md
```

Expected:

```text
... lines describing pointmap generation and deferred inference
```
