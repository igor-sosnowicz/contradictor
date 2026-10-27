"""Definitions of models used in Golden Dataset canonical schema."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from src.data_models.data_models import InterpretativeFrame, NLIPrediction


class StanceType(StrEnum):
    """Relation direction between two arguments."""

    UNKNOWN = "unknown"
    SUPPORTS = "supports"
    ATTACKS = "attacks"
    REBUTS = "rebuts"
    UNDERMINES = "undermines"


class ArgumentRelation(BaseModel):
    """Directed link between the current argument and another argument node."""

    target_argument_id: str = Field(
        ..., description="ID of the related target argument"
    )
    stance: StanceType = Field(
        ..., description="Relationship direction (supports, attacks, etc.)"
    )
    nli_prediction: NLIPrediction | None = Field(
        default=None,
        description="Optional NLI model inference backing this relationship",
    )


class TextBounds(BaseModel):
    """Character bounds of a text inside its source document."""

    start: int = Field(..., description="Start character offset in the document")
    end: int = Field(
        ..., description="End character offset (exclusive) in the document"
    )

    @property
    def is_located(self) -> bool:
        """Whether the bounds point at a real text range."""
        return self.end > self.start >= 0


class Claim(BaseModel):
    """Primary conclusion, thesis, or main assertion of an argument."""

    text: str = Field(..., description="The claim text")
    is_main_claim: bool = Field(
        default=False,
        description="True if this claim is the main conclusion of the argument",
    )
    bounds: TextBounds | None = Field(
        default=None,
        description=(
            "Character bounds in the source document, NONE when not document-grounded"
        ),
    )


class Premise(BaseModel):
    """Logical reasoning, warrants, or intermediate assertions supporting the claim."""

    text: str = Field(..., description="The reasoning text or intermediate assertion")
    is_implicit: bool = Field(
        default=False,
        description=(
            "True if this premise is an implicit enthymeme inferred by annotators"
        ),
    )
    bounds: TextBounds | None = Field(
        default=None,
        description=(
            "Character bounds in the source document, NONE when not document-grounded"
        ),
    )


class Evidence(BaseModel):
    """Empirical backing (data, statistics, citations, or verbatim quotes)."""

    text: str = Field(..., description="Raw text excerpt of the evidence")
    source_url: str | None = Field(
        default=None, description="Source of the evidence, if available"
    )
    bounds: TextBounds | None = Field(
        default=None,
        description=(
            "Character bounds in the source document, NONE when not document-grounded"
        ),
    )


class SchemaArgument(BaseModel):
    """Canonical representation of an argument unit in the Golden Dataset."""

    id: str = Field(..., description="Unique argument identifier")
    type: Literal["raw", "canonical"] = Field(
        ...,
        description="Type of the argument (raw or canonical)",
    )
    argument: str = Field(
        ...,
        description=(
            "The full argument text, including claim and supporting premises/evidence"
        ),
    )
    claim: list[Claim] | None = Field(
        default_factory=list,
        description="The primary conclusion, thesis, or main assertion",
    )
    premises: list[Premise] = Field(
        default_factory=list,
        description="Logical warrants and intermediate assertions supporting the claim",
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Empirical data, citations, or quotes backing the premises/claim",
    )
    frame: InterpretativeFrame | None = Field(
        default=None,
        description="InterpretativeFrame label, NONE when absent or to be classified",
    )
    relations: list[ArgumentRelation] = Field(
        default_factory=list,
        description=(
            "Graph edges to other arguments (supports, counter-arguments, rebuttals)"
        ),
    )
    provenance: str | None = Field(
        default=None, description="Dataset, corpus, or web source name"
    )
    claim_bounds: TextBounds | None = Field(
        default=None,
        description=(
            "Character bounds of the claim in the source document, "
            "NONE when not document-grounded"
        ),
    )
    document_id: str | None = Field(
        default=None,
        description=(
            "Source document this argument was annotated in, "
            "NONE for dataset-sourced arguments"
        ),
    )

    @property
    def is_valid(self) -> bool:
        """Check if the argument unit is valid (claim plus some support)."""
        has_claim = bool(self.claim)
        has_support = bool(self.premises) or bool(self.evidence)
        return has_claim and has_support

    @model_validator(mode="after")
    def ensure_correct_type(self) -> Self:
        """Ensure the type field is always set to 'canonical' for SchemaArgument."""
        if self.type == "canonical" and not self.is_valid:
            raise ValueError(
                "Canonical arguments must have a valid claim and at least one "
                "supporting premise or evidence."
            )
        return self
