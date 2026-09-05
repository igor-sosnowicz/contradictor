"""Unit tests for arguments dataset creation (no live LLM calls)."""

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.argument_dataset import pipeline
from src.argument_dataset.arg_db import (
    ArgumentDatabase,
    ArgumentRecord,
    EmbeddingIdentifier,
)
from src.argument_dataset.arg_link import ArgumentLinkType
from src.argument_dataset.backend import ExtractionBatch
from src.argument_dataset.config import PipelineConfig
from src.argument_dataset.output_models import (
    ExtractedArgument,
    ExtractedPremise,
    ExtractionVerdict,
    LinkVerdict,
)
from src.configuration import config as global_config
from src.data_models.data_models import InterpretativeFrame


class StubAgent:
    """Fake pydantic-ai agent returning a canned output."""

    def __init__(self, output: object) -> None:
        """Store the canned output."""
        self.output = output
        self.prompts: list[str] = []

    async def run(self, prompt: str) -> SimpleNamespace:
        """Record the prompt and return the canned output."""
        self.prompts.append(prompt)
        return SimpleNamespace(output=self.output)


def _extracted(text: str = "Stocks will rise on strong earnings.") -> ExtractedArgument:
    """Build a minimal extracted argument."""
    return ExtractedArgument(
        argument=text,
        evidence=[ExtractedPremise(text=text)],
        frame=InterpretativeFrame.ECONOMIC,
        language="english",
        target_claims_hypothesis=["Stocks will fall."],
    )


def _record(text: str = "Stocks will rise on strong earnings.") -> ArgumentRecord:
    """Build a grounded record without computing embeddings."""
    extracted = _extracted(text)
    return ArgumentRecord(
        **extracted.model_dump(exclude={"premises"}, by_alias=True),
        document_id="doc.txt",
        start_idx=0,
        end_idx=len(text),
        premises=[],
    )


@pytest.fixture
def sources_dir(tmp_path: Path) -> Path:
    """Create a sources directory with two small documents."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "a.txt").write_text(
        "Stocks will rise on strong earnings. " * 50, encoding="utf-8"
    )
    (sources / "b.md").write_text("Rates will stay high. " * 50, encoding="utf-8")
    (sources / "empty.txt").write_text("", encoding="utf-8")
    return sources


@pytest.fixture
def cfg(
    sources_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> PipelineConfig:
    """Pipeline config pointed at temporary directories."""
    monkeypatch.setattr(
        global_config, "counterargument_dataset_path", tmp_path / "args_db.json"
    )
    monkeypatch.setattr(
        global_config,
        "counterargument_dataset_embeddings_path",
        tmp_path / "args_db.pkl",
    )
    monkeypatch.setattr(
        EmbeddingIdentifier, "create", classmethod(lambda cls, content=None: cls())
    )
    config = PipelineConfig()
    config.sources_dir = sources_dir
    config.records_dir = tmp_path / "records"
    config.chunk_size = 200
    config.chunk_overlap = 20
    return config.validated()


@pytest.fixture
def stub_agents(monkeypatch: pytest.MonkeyPatch) -> dict[str, StubAgent]:
    """Stub extractor / judge / link agents used by the pipeline."""
    agents = {
        "extractor": StubAgent(ExtractionBatch(arguments=[_extracted()])),
        "judge": StubAgent(ExtractionVerdict(verdict="accept", reason="ok")),
        "link": StubAgent(LinkVerdict(refutes=False, reason="unrelated topics")),
    }
    extractors = (agents["extractor"], agents["judge"], agents["link"])
    monkeypatch.setattr(pipeline, "make_agents", lambda _extractor, _judge: extractors)
    return agents


def test_chunk_text_offsets_cover_full_text() -> None:
    """Chunks cover the whole text with the requested overlap."""
    text = "word " * 500
    chunks = pipeline.chunk_text(text, size=200, overlap=20)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for previous, current in itertools.pairwise(chunks):
        assert current.start < previous.end
        assert current.start >= previous.end - 20


def test_load_documents_skips_empty(sources_dir: Path) -> None:
    """Empty files are skipped, txt and md files are loaded."""
    documents = pipeline.load_documents(sources_dir)
    assert {d.document_id for d in documents} == {"a.txt", "b.md"}


def test_run_phase1_accepts_and_saves(
    cfg: PipelineConfig, stub_agents: dict[str, StubAgent]
) -> None:
    """Accepted extractions are grounded and the database is saved."""
    db = pipeline.run_phase1(cfg)
    assert len(db.arguments) > 0
    record = db.arguments[0]
    assert record.document_id in {"a.txt", "b.md"}
    assert record.start_idx >= 0
    assert record.end_idx > record.start_idx
    assert global_config.counterargument_dataset_path.exists()
    reloaded = ArgumentDatabase.load(load_embeddings=False)
    assert len(reloaded.arguments) == len(db.arguments)


def test_run_phase1_repair_uses_fixed_text(
    cfg: PipelineConfig, stub_agents: dict[str, StubAgent]
) -> None:
    """Repair verdicts replace the extracted argument text."""
    fixed = _extracted("Repaired claim about markets.")
    stub_agents["judge"].output = ExtractionVerdict(
        verdict="repair", repaired=fixed, reason="fix wording"
    )
    db = pipeline.run_phase1(cfg)
    assert db.arguments
    assert all(a.argument == fixed.argument for a in db.arguments)


def test_run_phase1_no_repair_keeps_original(
    cfg: PipelineConfig, stub_agents: dict[str, StubAgent]
) -> None:
    """With no_repair, the original extraction is kept despite repair verdicts."""
    fixed = _extracted("Repaired claim about markets.")
    stub_agents["judge"].output = ExtractionVerdict(
        verdict="repair", repaired=fixed, reason="fix wording"
    )
    db = pipeline.run_phase1(cfg, no_repair=True)
    assert db.arguments
    assert all(a.argument != fixed.argument for a in db.arguments)


def test_run_phase1_reject_skips(
    cfg: PipelineConfig, stub_agents: dict[str, StubAgent]
) -> None:
    """Rejected extractions never reach the database."""
    stub_agents["judge"].output = ExtractionVerdict(
        verdict="reject", reason="fabricated"
    )
    db = pipeline.run_phase1(cfg)
    assert db.arguments == []


def test_run_phase2_confirms_link(
    cfg: PipelineConfig,
    stub_agents: dict[str, StubAgent],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirmed links are stored in the database and persisted."""
    db = ArgumentDatabase()
    db += _record("Stocks will rise.")
    db += _record("Stocks will fall.")
    db.save()
    _, target = db.arguments
    monkeypatch.setattr(
        ArgumentDatabase,
        "fetch_similar_arguments",
        lambda self, query, top_k=5, **kwargs: [(target, 0.8)],
    )
    stub_agents["link"].output = LinkVerdict(
        refutes=True, link_type=ArgumentLinkType.CONTRADICTION, reason="opposite claims"
    )
    result = pipeline.run_phase2(cfg)
    assert len(result.argument_links) >= 1
    reloaded = ArgumentDatabase.load(load_embeddings=False)
    assert len(reloaded.argument_links) == len(result.argument_links)


def test_run_phase2_no_refutation_no_link(
    cfg: PipelineConfig,
    stub_agents: dict[str, StubAgent],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Denied pairs produce no links."""
    db = ArgumentDatabase()
    db += _record("Stocks will rise.")
    db += _record("Unrelated weather report.")
    db.save()
    target = db.arguments[1]
    monkeypatch.setattr(
        ArgumentDatabase,
        "fetch_similar_arguments",
        lambda self, query, top_k=5, **kwargs: [(target, 0.1)],
    )
    result = pipeline.run_phase2(cfg)
    assert result.argument_links == set()


def test_add_argument_link_guards() -> None:
    """Missing arguments raise, duplicates are rejected."""
    db = ArgumentDatabase()
    record = _record()
    db += record
    with pytest.raises(ValueError, match="Target argument not found"):
        db.add_argument_link(record, _record("Other."), ArgumentLinkType.REBUTTAL)
    other = _record("Other.")
    db += other
    db.add_argument_link(record, other, ArgumentLinkType.REBUTTAL)
    with pytest.raises(ValueError, match="already exists"):
        db.add_argument_link(other, record, ArgumentLinkType.REBUTTAL)
