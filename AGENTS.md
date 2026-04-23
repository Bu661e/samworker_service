# Repository Guidelines

## Project Structure & Module Organization

This repository contains several related Python components:

- `sam3worker/` and `sam3dworker/`: persistent worker clients, services, and model-facing logic. Each has `tests/`, `docs/`, and sample inputs under `tests/inputs/`.
- `worker_ipc/`: reusable Unix domain socket JSONL IPC package. Source lives in `worker_ipc/worker_ipc/`, tests in `worker_ipc/tests/`, and examples in `worker_ipc/examples/`.
- `sam_pipeline_api/`: FastAPI service that orchestrates SAM3 and SAM3D workers. Example requests live in `sam_pipeline_api/examples/`.
- `object_geometry/`: camera-space point cloud and OBB helpers used by the pipeline.
- `docs/superpowers/`: planning and design notes.

Check module-level `AGENTS.md` files before editing `sam3worker/` or `sam3dworker/`.

## Build, Test, and Development Commands

Use the project root as the working directory unless noted.

```bash
python3 -m pytest worker_ipc/tests -q
```

Runs the lightweight IPC test suite.

```bash
/opt/conda/bin/python -m pytest sam3worker/tests
/opt/conda/bin/python -m pytest sam3dworker/tests
```

Runs model worker tests in the expected SAM3D environment.

```bash
/opt/conda/bin/python -m sam_pipeline_api.serve
```

Starts the FastAPI pipeline service and persistent workers. Direct Uvicorn launch is also supported:

```bash
/opt/conda/bin/python -m uvicorn sam_pipeline_api.app:app --host 0.0.0.0 --port 6006
```

## Coding Style & Naming Conventions

Write Python 3.9+ code with 4-space indentation and type hints for public functions and cross-module interfaces. Keep modules focused around existing roles: `client.py` for parent-side callers, `worker.py` for inference logic, and `service.py` for IPC/server entrypoints. Use `snake_case` for functions, variables, payload keys, and test names; use `PascalCase` for classes. Prefer `pathlib.Path` for filesystem paths and structured JSON helpers over ad hoc string parsing.

## Testing Guidelines

Tests use `pytest` and should be named `test_*.py`. Keep sample inputs in the relevant `tests/inputs/` directory. Generated outputs should go under `tests/runs/` with timestamped directories such as `YYYY-MM-DD-HH-MM-SS`, and should not be committed unless intentionally added as fixtures.

## Commit & Pull Request Guidelines

History uses short imperative subjects, with optional Conventional Commit scopes such as `feat(sam3worker): add persistent bbox inference flow`. Keep commit subjects specific and under one line. Pull requests should describe the changed module, list the verification command run, mention any model/environment assumptions, and link relevant issues or design notes. Include screenshots or sample JSON only when API output or visual artifacts change.

## Security & Configuration Tips

Do not commit model weights, generated run directories, credentials, or machine-specific secrets. Treat absolute environment paths in docs as deployment hints; keep code configurable through arguments or environment variables where practical.
