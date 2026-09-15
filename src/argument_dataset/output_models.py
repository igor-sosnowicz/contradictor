"""Pydantic output models for the extractor and judge agents."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.argument_dataset.arg_link import ArgumentLinkType
from src.data_models.data_models import InterpretativeFrame


class ExtractedPremise(BaseModel):
    """Premise extracted from text."""

    text: str = Field(..., description="Exact text of the premise")


class ExtractedArgument(BaseModel):
    """Argument extracted from text, with its premises and context information."""

    model_config = ConfigDict(populate_by_name=True)

    argument: str = Field(..., description="Main claim + evidence / extracted argument")
    premises: list[ExtractedPremise] = Field(
        default_factory=list, description="List of premises", alias="evidence"
    )
    domain: InterpretativeFrame = Field(
        ..., description="Domain of the argument", alias="frame"
    )  # Primary & Secondary
    language: str = Field("english", description="Language of the argument")

    target_claims_hypothesis: list[str] = Field(
        default_factory=list,
        description="Hypothesis of the target claim that this argument refutes",
    )


class ExtractionBatch(BaseModel):
    """Extractor output: all arguments found in one chunk."""

    arguments: list[ExtractedArgument] = Field(default_factory=list)


class ExtractionVerdict(BaseModel):
    """Judge verdict for a single extracted argument."""

    verdict: Literal["accept", "repair", "reject"] = Field(
        ..., description="Judge decision for the extracted argument"
    )
    repaired: ExtractedArgument | None = Field(
        default=None, description="Corrected argument when verdict is repair"
    )
    reason: str = Field(default="", description="Short justification")

    @model_validator(mode="after")
    def _check_repaired(self) -> "ExtractionVerdict":
        """Ensure repaired argument is present when verdict is repair."""
        if self.verdict == "repair" and self.repaired is None:
            raise ValueError(
                "verdict is set to 'repair', but repaired field is 'None'."
            )
        if self.verdict != "repair" and self.repaired is not None:
            raise ValueError("'repaired' field present, but verdict is not 'repair'.")
        return self


class LinkVerdict(BaseModel):
    """Judge verdict for a candidate (argument, target) link."""

    refutes: bool = Field(..., description="Whether A refutes B")
    link_type: ArgumentLinkType | None = Field(
        default=None, description="Link type when refutes is true"
    )
    repaired_target_hypothesis: str | None = Field(
        default=None, description="Corrected hypothesis of what A refutes"
    )
    reason: str = Field(default="", description="Short justification")

    @model_validator(mode="after")
    def _check_link_type(self) -> "LinkVerdict":
        """Ensure link_type is present when refutes is true."""
        if self.refutes and self.link_type is None:
            raise ValueError("refutes is set to True, but link_type field is 'None'.")
        if not self.refutes and self.link_type is not None:
            raise ValueError("link_type field present, but refutes is set to False.")
        return self


class FrameClassification(BaseModel):
    """Classifier output: the interpretative frame of a single argument."""

    frame: InterpretativeFrame = Field(
        ..., description="Interpretative frame that best fits the argument"
    )
    reason: str = Field(default="", description="Short justification")


class GeneratedPremises(BaseModel):
    """Generator output: candidate premises supporting a given conclusion."""

    premises: list[str] = Field(
        default_factory=list,
        description="2-3 premises that support the conclusion",
    )


class PremiseSupportVerdict(BaseModel):
    """Judge verdict: how well a premise set backs up one conclusion."""

    support: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How strongly the premises back the conclusion, 0.0 (not at all) "
            "to 1.0 (fully). Support, not logical entailment."
        ),
    )
    reason: str = Field(default="", description="Short justification")


class StyledDocument(BaseModel):
    """Styler output: a carrier document rewritten around verbatim sentences."""

    text: str = Field(..., description="Full text of the styled document")
