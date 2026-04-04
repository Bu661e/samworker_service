# Worker Repository Reorganization Design

## Summary

This document defines a repository reorganization for `samworker_service` so that SAM3 and SAM3D are represented as separate top-level worker projects, while shared IPC code and upstream third-party code are kept in clearly separate locations.

## Goals

- Make `sam3worker` and `sam3dworker` first-class top-level directories.
- Keep reusable IPC code isolated in `worker_ipc/`.
- Move upstream code under `third_party/` rather than mixing it with worker-owned code.
- Split documentation by worker instead of keeping a shared `samworker/docs` bucket.
- Initialize this directory as the repository root.

## Non-Goals

- Rewriting the real SAM3 or SAM3D inference logic in this change.
- Refactoring `worker_ipc/`.
- Building a full production command surface for both workers.
- Adding a complete test matrix for the new worker skeletons.

## Target Layout

```text
samworker_service/
  .git/
  .gitignore
  docs/
    superpowers/
      specs/
      plans/
  worker_ipc/
  sam3worker/
    README.md
    docs/
      README.md
      ultralytics_SAM3_使用指南.md
      进程与环境说明.md
    worker.py
    service.py
  sam3dworker/
    README.md
    docs/
      README.md
      SAM3D-object_使用指南.md
      进程与环境说明.md
    worker.py
    service.py
  third_party/
    sam3-ultralytics/
    SAM3D-object/
```

## Ownership Rules

- `worker_ipc/` contains only reusable IPC code.
- `sam3worker/` contains only SAM3 worker-owned code and docs.
- `sam3dworker/` contains only SAM3D worker-owned code and docs.
- `third_party/` contains only upstream or vendor-style code.
- The legacy `samworker/` directory is transitional only and will be removed by this change.

## Migration Rules

1. Initialize `.git` at the current `samworker_service` root.
2. Create `sam3worker/`, `sam3dworker/`, and `third_party/`.
3. Move:
   - `samworker/sam3-ultralytics/` -> `third_party/sam3-ultralytics/`
   - `samworker/SAM3D-object/` -> `third_party/SAM3D-object/`
4. Create worker skeleton files:
   - `sam3worker/README.md`
   - `sam3worker/docs/README.md`
   - `sam3worker/worker.py`
   - `sam3worker/service.py`
   - `sam3dworker/README.md`
   - `sam3dworker/docs/README.md`
   - `sam3dworker/worker.py`
   - `sam3dworker/service.py`
5. Split old shared docs into worker-specific docs.
6. Remove the legacy `samworker/` directory once its contents have been relocated.

## Worker Skeleton Expectations

The new worker skeletons are intentionally thin.

- `worker.py` should expose a minimal `worker_ipc`-based entrypoint.
- `service.py` should describe where the real third-party dependency lives and provide a small placeholder boundary for future implementation.
- `README.md` and `docs/README.md` should explain the new layout and point to the migrated docs and third-party directory.

## Verification

The reorganization is considered successful when:

- `.git` exists at the repository root.
- `third_party/sam3-ultralytics/` and `third_party/SAM3D-object/` exist.
- `sam3worker/` and `sam3dworker/` each contain README, docs, worker, and service files.
- The old `samworker/` directory no longer exists.
- The existing `worker_ipc` test suite still passes.
