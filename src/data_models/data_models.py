"""Module with project-wide data models."""

import ast
from enum import Enum, StrEnum, auto
from typing import Final

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class SubsetName(StrEnum):
    """Name of a subset/split."""

    TRAINING = auto()
    VALIDATION = auto()
    TESTING = auto()


class InterpretativeFrame(StrEnum):
    """Universal interpretative frames from the Policy Frames Codebook."""

    ECONOMIC = "Economic"
    CAPACITY_AND_RESOURCES = "Capacity and Resources"
    MORALITY = "Morality"
    FAIRNESS_AND_EQUALITY = "Fairness and Equality"
    CONSTITUTIONALITY_AND_LEGALITY = "Constitutionality and Legality"
    POLICY_PRESCRIPTION = "Policy Prescription and Evaluation"
    CRIME_AND_JUSTICE = "Crime and Justice"
    SECURITY_AND_DEFENSE = "Security and Defense"
    HEALTH_AND_SAFETY = "Health and Safety"
    QUALITY_OF_LIFE = "Quality of Life"
    CULTURAL_IDENTITY = "Cultural Identity"
    PUBLIC_OPINION = "Public Opinion"
    POLITICAL = "Political"
    EXTERNAL_REGULATION = "External Regulation and Reputation"
    OTHER = "Other"


# Mapping between MediaFrameCorpus code_frames values and domain enum.
# Dataset uses codes 1-15, while application logic uses InterpretativeFrame.
LABEL_TO_FRAME: Final[dict[int, InterpretativeFrame]] = {
    0: InterpretativeFrame.ECONOMIC,
    1: InterpretativeFrame.CAPACITY_AND_RESOURCES,
    2: InterpretativeFrame.MORALITY,
    3: InterpretativeFrame.FAIRNESS_AND_EQUALITY,
    4: InterpretativeFrame.CONSTITUTIONALITY_AND_LEGALITY,
    5: InterpretativeFrame.POLICY_PRESCRIPTION,
    6: InterpretativeFrame.CRIME_AND_JUSTICE,
    7: InterpretativeFrame.SECURITY_AND_DEFENSE,
    8: InterpretativeFrame.HEALTH_AND_SAFETY,
    9: InterpretativeFrame.QUALITY_OF_LIFE,
    10: InterpretativeFrame.CULTURAL_IDENTITY,
    11: InterpretativeFrame.PUBLIC_OPINION,
    12: InterpretativeFrame.POLITICAL,
    13: InterpretativeFrame.EXTERNAL_REGULATION,
    14: InterpretativeFrame.OTHER,
}


class Argument(BaseModel):
    """An argument made of claim and its evidence."""

    claim: str
    evidence: str

    def __str__(self) -> str:
        """
        Convert an argument to a textual form.

        Returns:
            str: Concatenated claim and evidence.
        """
        return f"{self.claim} {self.evidence}"


class FramedArgument(Argument):
    """Data model representing an argument paired with its interpretative frame."""

    primary_frame: InterpretativeFrame = Field(
        ...,
        description="The primary interpretative lens assigned to this argument.",
    )

    frame_probabilities: dict[InterpretativeFrame, float] = Field(
        ...,
        description=(
            "Dictionary mapping each interpretative frame "
            "to its prediction probability."
        ),
    )


class StyledFramedArgument(FramedArgument):
    """A framed argument with a style vector assigned to it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    style_vector: np.ndarray

    @field_validator("style_vector", mode="before")
    @classmethod
    def _coerce_style_vector(cls, value: object) -> np.ndarray:
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        if isinstance(value, np.ndarray):
            return value
        raise TypeError(f"Unsupported type for `style_vector`: {type(value)}")

    @field_serializer("style_vector")
    def _serialize_style_vector(self, value: np.ndarray) -> list:
        return value.tolist()

    @classmethod
    def from_row(cls, row: dict, suffix: str) -> "StyledFramedArgument":
        """Build a DTO from a DataFrame row's columns for the given suffix."""
        probs_dict = row.get(f"frame_probabilities_{suffix}", {})
        if isinstance(probs_dict, str):
            probs_dict = ast.literal_eval(probs_dict)
        elif probs_dict is None or (
            isinstance(probs_dict, float) and pd.isna(probs_dict)
        ):
            probs_dict = {}

        return cls(
            claim=row[f"claim_{suffix}"],
            evidence=row[f"evidence_{suffix}"],
            primary_frame=InterpretativeFrame(row[f"frame_{suffix}"]),
            frame_probabilities={
                frame: float(probs_dict.get(frame, 0.0))
                for frame in InterpretativeFrame
            },
            style_vector=np.asarray(row[f"style_vector_{suffix}"]),
        )

    def to_tensor(self) -> torch.Tensor:
        """
        Convert a StyledFramedArgument to a fixed-size input tensor.

        Returns:
            torch.Tensor: Include the following features:
                - style_vector: static length dependant on a style extractor used
                - frame one-hot: len(InterpretativeFrame) dimensions
                - frame probabilities: len(InterpretativeFrame) dimensions
        """
        style = torch.tensor(self.style_vector, dtype=torch.float32).reshape(-1)

        frame_one_hot = torch.zeros(len(InterpretativeFrame), dtype=torch.float32)
        frame_index = list(InterpretativeFrame).index(self.primary_frame)
        frame_one_hot[frame_index] = 1.0

        frame_probs = torch.tensor(
            [self.frame_probabilities.get(frame, 0.0) for frame in InterpretativeFrame],
            dtype=torch.float32,
        )
        return torch.cat([style, frame_one_hot, frame_probs])


class ArgumentWithCounterarguments(BaseModel):
    """An argument with its counter-arguments."""

    argument: FramedArgument
    counterarguments: tuple[FramedArgument, ...]


class NLIResult(Enum):
    """
    Result of a natural language inference.

    Attributes:
        NEUTRAL: Two chunks of texts are independent of each other.
            They address different issues and are not associated.
        ENTAILMENT: One chunk of text confirms, proves, or agrees with
            the other one. They are logically conformant.
        CONTRADICTION: One chunks of texts contradicts the other one.
            They disagree with each other.
    """

    NEUTRAL = 0
    ENTAILMENT = 1
    CONTRADICTION = -1


class NLIPrediction(BaseModel):
    """Prediction of an NLI model with its decision (result) and confidence."""

    result: NLIResult
    confidence: float = Field(..., ge=0.0, le=1.0)


class EncoderImplementation(StrEnum):
    """Encoder implementation name."""

    MLP = auto()


class SearchPipelineImplementation(StrEnum):
    """Name of search pipeline implementation."""

    SELF_IMPLEMENTED = auto()
    FIREFOX_READER_VIEW = auto()


class ArgumentFramerImplementation(StrEnum):
    """Name of argument framer implementation."""

    XGBOOST = auto()


class ArgumentExtractorImplementation(StrEnum):
    """Name of argument extractor implementation."""

    XGBOOST = "xgboost"


class NLIImplementation(StrEnum):
    """Name of natural language inference implementation."""

    TRANSFORMERS_MODERNBERT = auto()


class StyleExtractorImplementation(StrEnum):
    """Name of style extractor implementation."""

    SPACY = auto()


class VectorSearchImplementation(StrEnum):
    """Name of vector search implementation."""

    NUMPY_BASED = auto()
