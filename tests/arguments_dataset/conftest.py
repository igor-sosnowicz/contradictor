"""Shared fixtures for the arguments dataset tests (no live LLM, no embeddings)."""

from types import SimpleNamespace

import pytest

from src.argument_dataset.arg_db import (
    ArgumentDatabase,
    ArgumentProvenance,
    ArgumentRecord,
    EmbeddingIdentifier,
    PremiseSource,
)
from src.argument_dataset.output_models import ExtractedPremise
from src.data_models.data_models import InterpretativeFrame


class StubAgent:
    """Fake pydantic-ai agent returning canned outputs in order."""

    def __init__(self, *outputs: object) -> None:
        """Store the canned outputs, repeating the last one once exhausted."""
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    async def run(self, prompt: str) -> SimpleNamespace:
        """Record the prompt and return the next canned output."""
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.outputs) - 1)
        return SimpleNamespace(output=self.outputs[index])


@pytest.fixture(autouse=True)
def _no_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real embedding computation everywhere in this package."""
    monkeypatch.setattr(
        EmbeddingIdentifier, "create", classmethod(lambda cls, content=None: cls())
    )


def make_record(
    argument: str = "Fares should stay frozen.",
    *,
    source: str = "src_a",
    original_id: str | None = "1",
    counter_ids: list[str] | None = None,
    premises: list[str] | None = None,
    premises_source: PremiseSource = PremiseSource.NONE,
    domain: InterpretativeFrame = InterpretativeFrame.ECONOMIC,
) -> ArgumentRecord:
    """Build a gold record (no carrier) without computing embeddings."""
    return ArgumentRecord(
        argument=argument,
        evidence=[ExtractedPremise(text=text) for text in premises or []],
        frame=domain,
        language="english",
        target_claims_hypothesis=[],
        provenance=ArgumentProvenance(
            source_dataset=source,
            original_id=original_id,
            premises_source=premises_source,
        ),
        native_counter_ids=counter_ids or [],
    )


def make_db(*records: ArgumentRecord) -> ArgumentDatabase:
    """Build a database holding the given records."""
    db = ArgumentDatabase()
    for record in records:
        db += record
    return db
