"""Module with project-wide data models."""

from dataclasses import dataclass
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
LABEL_TO_FRAME: dict[int, InterpretativeFrame] = {
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


class FramedArgument(BaseModel):
    """Data model representing an argument paired with its interpretative frame."""

    text: str

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


@dataclass(slots=True, frozen=True)
class ClaimEvidencePair:
    """
    Data structure representing a claim-evidence pair.

    Attributes:
        claim (str):
            Extracted claim statement.

        evidence (str):
            Supporting evidence associated with the claim.
    """

    claim: str
    evidence: str
