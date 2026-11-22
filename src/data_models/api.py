"""Module with Web API data models."""

from pydantic import BaseModel, Field

from src.configuration import config
from src.data_models.data_models import ArgumentWithCounterarguments


class ContradictInput(BaseModel):
    """Input data structure for the`/contradict` Web API endpoint."""

    text: str = Field(
        ..., min_length=10, max_length=config.reference_text_max_characters
    )


class ContradictOutput(BaseModel):
    """Output with argument vs. counter-argument pairs found by Contradictor."""

    pairs: list[ArgumentWithCounterarguments]
