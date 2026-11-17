"""Web API router with endpoints."""

from fastapi import APIRouter, Request

from src.data_models.api import ContradictInput, ContradictOutput
from src.pipeline.contradictor_pipeline import ContradictorPipeline
from src.utils.errors import NotPreparedError

router = APIRouter()


def get_pipeline(request: Request) -> ContradictorPipeline:
    """

    Args:
        request (Request): HTTP request carrying the application state.

    Returns:
        ContradictorPipeline: An initialised Contradictor pipeline.

    Raises:
        NotPreparedError: Raised if the pipeline has not been yet initialised and
            needs more time to load.
    """
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise NotPreparedError(
            "The contradictor pipeline has not been initialised yet. "
            "Wait a few seconds and try again."
        )
    return pipeline


@router.get("/health")
async def check_health(request: Request) -> dict[str, str]:
    """
    Check if an application if healthy.

    Args:
        request (Request): An HTTP request passed on internally to the API.

    Returns:
        dict[str, str]: Summary of health and state of the Contradictor server.
    """
    try:
        _ = get_pipeline(request)
    except NotPreparedError:
        return {"status": "not-ready"}

    return {"status": "healthy"}


@router.post("/contradict")
async def find_contradictions(
    contradict_input: ContradictInput,
    request: Request,
) -> ContradictOutput:
    """
    Find counterarguments to arguments present in the provided text.

    Args:
        contradict_input (ContradictInput): Body of a request with a text to find
            arguments in.
        request (Request): Internally used request to get the initialised
            Contradictor pipeline instance from.

    Returns:
        ContradictOutput: Argument to counter-argument pairs.
    """
    pipeline = get_pipeline(request)
    return await pipeline(contradict_input.text)
