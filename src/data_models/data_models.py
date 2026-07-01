"""Module with project-wide data models."""

from enum import StrEnum, auto

from pydantic import BaseModel, Field


class TextSpan(BaseModel):
    """Span of text in a longer text."""

    start_index: int = Field(..., ge=0)
    length: int = Field(..., ge=1)


class SubsetName(StrEnum):
    """Name of a subset/split."""

    TRAINING = auto()
    VALIDATION = auto()
    TESTING = auto()
