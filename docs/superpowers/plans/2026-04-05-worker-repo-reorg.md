# Worker Repository Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repository so SAM3 and SAM3D each have their own top-level worker directory, upstream code lives under `third_party/`, and this folder becomes the git repository root.

**Architecture:** Treat the change as a filesystem reorganization rather than a functional rewrite. Preserve `worker_ipc/`, move upstream directories without altering their internals, add thin worker skeletons, split docs per worker, and remove the legacy `samworker/` directory once migration is complete.

**Tech Stack:** Git, Python 3, existing `worker_ipc` package, filesystem moves, Markdown docs

---

This workspace previously had no `.git` directory. This plan intentionally initializes a new repository at the current root.

## File Structure

- Create: `.gitignore`
- Create: `sam3worker/README.md`
- Create: `sam3worker/docs/README.md`
- Create: `sam3worker/docs/ultralytics_SAM3_使用指南.md`
- Create: `sam3worker/docs/进程与环境说明.md`
- Create: `sam3worker/worker.py`
- Create: `sam3worker/service.py`
- Create: `sam3dworker/README.md`
- Create: `sam3dworker/docs/README.md`
- Create: `sam3dworker/docs/SAM3D-object_使用指南.md`
- Create: `sam3dworker/docs/进程与环境说明.md`
- Create: `sam3dworker/worker.py`
- Create: `sam3dworker/service.py`
- Create: `third_party/sam3-ultralytics/`
- Create: `third_party/SAM3D-object/`
- Move: `samworker/sam3-ultralytics/`
- Move: `samworker/SAM3D-object/`
- Remove: `samworker/`

### Task 1: Initialize Repository Root

- [ ] Run `git init` in the repository root.
- [ ] Create `.gitignore` for `.pytest_cache/`, `__pycache__/`, `.DS_Store`, and local virtual environment folders.
- [ ] Verify `.git/` exists with `git status --short`.

### Task 2: Move Third-Party Code

- [ ] Create `third_party/`.
- [ ] Move `samworker/sam3-ultralytics/` to `third_party/sam3-ultralytics/`.
- [ ] Move `samworker/SAM3D-object/` to `third_party/SAM3D-object/`.
- [ ] Verify the moved directories exist with `find third_party -maxdepth 2 -type d | sort`.

### Task 3: Create Worker Skeletons

- [ ] Create `sam3worker/` and `sam3dworker/` plus their `docs/` subdirectories.
- [ ] Create minimal `worker.py` and `service.py` files for each worker.
- [ ] Create worker README files describing ownership boundaries and third-party dependency locations.

### Task 4: Split Docs By Worker

- [ ] Move the SAM3 usage guide into `sam3worker/docs/`.
- [ ] Move the SAM3D usage guide into `sam3dworker/docs/`.
- [ ] Replace the shared environment/process doc with separate worker-specific docs.
- [ ] Add docs index files for each worker.

### Task 5: Remove Legacy Layout

- [ ] Delete the legacy `samworker/` directory after all files have been migrated.
- [ ] Verify the root layout with `find . -maxdepth 2 -type d | sort`.

### Task 6: Verification

- [ ] Run `python3 -m pytest worker_ipc/tests -q`.
- [ ] Verify `PYTHONPATH=worker_ipc python3 worker_ipc/examples/echo_parent.py`.
- [ ] Verify `git status --short` shows the new repo is tracking the reorganized layout.
