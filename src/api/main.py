"""Module managing Web API lifecycle."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.api.router import router
from src.configuration import config
from src.data_models.data_models import ContradictorPipelinePreset
from src.pipeline.factories.contradictor_pipeline_factory import (
    get_contradictor_pipeline,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    Handle start-up and close-down of the application including resource management.

    Args:
        app (FastAPI): A Web application that starts up and closes down.

    Returns:
        AsyncGenerator[None]: Empty results, used internally by FastAPI.
    """
    try:
        # Handle the start-up and load the resources.
        pipeline = get_contradictor_pipeline(
            pipeline_preset=ContradictorPipelinePreset.FOR_API
        )

        await pipeline.prepare()
        app.state.pipeline = pipeline
        yield

    finally:
        # Handle the clean-up and release the resources.
        app.state.pipeline = None


def run_api() -> None:
    """Start running the Contradictor Web API."""
    app = FastAPI(
        title="Contradictor API",
        summary="Web API for Contradictor, the NLP counterargument finding system.",
        lifespan=lifespan,
    )
    app.include_router(router)
    uvicorn.run(
        app, host=config.api_host, port=config.api_port, log_level=config.log_level
    )
