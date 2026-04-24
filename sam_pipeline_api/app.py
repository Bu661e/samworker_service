from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from sam3worker import Sam3WorkerCommandError

from .models import ReconstructObjectsRequest, ReconstructObjectsResponse
from .pipeline import PipelineInputError, SamPipelineService


def create_app() -> FastAPI:
    service = SamPipelineService.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        app.state.pipeline_service = service
        try:
            yield
        finally:
            service.stop()

    app = FastAPI(
        title="sam-pipeline-api",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.post("/v1/objects/reconstruct", response_model=ReconstructObjectsResponse)
    def reconstruct_objects(request: ReconstructObjectsRequest) -> ReconstructObjectsResponse:
        pipeline_service: SamPipelineService = app.state.pipeline_service
        try:
            return pipeline_service.reconstruct_objects(request)
        except PipelineInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Sam3WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()
