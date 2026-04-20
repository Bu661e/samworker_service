from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("SAM_PIPELINE_HOST", "0.0.0.0")
    port = int(os.environ.get("SAM_PIPELINE_PORT", "6006"))
    uvicorn.run("sam_pipeline_api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
