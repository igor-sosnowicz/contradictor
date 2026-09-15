"""Unit tests for the dataset pipelines (no live LLM, no embeddings)."""

import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastembed import SparseEmbedding

from src.argument_dataset.arg_db import (
    ArgumentDatabase,
    ArgumentRecord,
    EmbeddingIdentifier,
    PremiseSource,
)
from src.argument_dataset.arg_link import ArgumentLinkType
from src.argument_dataset.config import PipelineConfig
from src.argument_dataset.datasets_handlers.adapters import HITZ_COUNTER_ARGUMENT
from src.argument_dataset.datasets_handlers.raw_schema import (
    RawArgumentRow,
    normalize,
    read_canonical,
    to_dataframe,
)
from src.argument_dataset.output_models import (
    ExtractedArgument,
    ExtractedPremise,
    ExtractionBatch,
    ExtractionVerdict,
    FrameClassification,
    GeneratedPremises,
    LinkVerdict,
    PremiseSupportVerdict,
    StyledDocument,
)
from src.argument_dataset.pipelines import arg_to_argrec, carrier_to_argdb
from src.argument_dataset.pipelines import argrec_to_carrier as synth
from src.argument_dataset.pipelines.merge_arg_dbs import merge_databases
from src.data_models.data_models import InterpretativeFrame
from tests.arguments_dataset.conftest import StubAgent, make_db, make_record

FILLER = [
    "The city council met on Tuesday evening to review the transport budget "
    "for the coming year, a session that ran well past its scheduled close.",
    "Local businesses along the main corridor have reported mixed results over "
    "the past quarter, with several owners pointing to the roadworks.",
    "Residents of the eastern district submitted a petition last month asking "
    "the authority to reconsider the timetable it introduced in spring.",
    "The regional authority published its annual report in March, covering "
    "passenger numbers, maintenance spending and the state of the fleet.",
    "A public consultation is scheduled for the end of the year, and the "
    "authority has said every submission will be published in full.",
]


def with_embeddings(record: ArgumentRecord, vector: list[float]) -> ArgumentRecord:
    """Give a record dense embeddings so retrieval can actually score it."""
    dense = np.array(vector, dtype=np.float32)
    empty = SparseEmbedding(values=np.array([]), indices=np.array([]))
    record.argument_embedding = EmbeddingIdentifier(
        dense_embedding=dense, sparse_embedding=empty
    )
    record.target_claims_embedding = EmbeddingIdentifier(
        dense_embedding=dense, sparse_embedding=empty
    )
    return record


class StubScorer:
    """NLI scorer returning canned contradiction scores in order."""

    def __init__(self, *contradictions: float) -> None:
        """Store the canned scores, repeating the last one once exhausted."""
        self.contradictions = list(contradictions) or [0.0]
        self.calls: list[tuple[str, str]] = []

    def contradiction(self, premise: str, conclusion: str) -> float:
        """Return the next canned contradiction score."""
        self.calls.append((premise, conclusion))
        index = min(len(self.calls) - 1, len(self.contradictions) - 1)
        return self.contradictions[index]

    def contradiction_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score each pair in turn."""
        return [self.contradiction(premise, text) for premise, text in pairs]

    def score(self, premise: str, conclusion: str) -> float:
        """Entailment is unused by the gate; report none."""
        return 0.0

    def score_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Entailment is unused by the gate; report none."""
        return [0.0] * len(pairs)


# --- datasets_handlers: adapters and canonical schema ---


def test_hitz_adapter_splits_pairs_into_mutually_referencing_rows() -> None:
    """One source row becomes two canonical rows pointing at each other."""
    frame = pd.DataFrame(
        {
            "argument": ["Taxes should fall.", ""],
            "counter-argument": ["Taxes fund services.", "Orphan."],
        }
    )
    rows = normalize(HITZ_COUNTER_ARGUMENT, frame)

    assert [row.id for row in rows] == ["0:arg", "0:counter"]  # blank pair dropped
    assert rows[0].counter_argument_ids == ["0:counter"]
    assert rows[1].counter_argument_ids == ["0:arg"]
    assert all(row.frame is None for row in rows)
    assert not rows[0].has_native_premises


def test_canonical_roundtrip_survives_parquet_shape() -> None:
    """Rows serialised to a dataframe read back identically."""
    rows = [
        RawArgumentRow(
            id="1",
            argument="Rents keep rising.",
            counter_argument_ids=["2"],
            premises=["Supply is flat."],
            frame=InterpretativeFrame.ECONOMIC,
        ),
        RawArgumentRow(id="2", argument="Rents are stable."),
    ]
    restored = read_canonical(to_dataframe(rows))

    assert [row.id for row in restored] == ["1", "2"]
    assert restored[0].counter_argument_ids == ["2"]
    assert restored[0].premises == ["Supply is flat."]
    assert restored[0].frame == InterpretativeFrame.ECONOMIC
    assert restored[1].counter_argument_ids == []


def test_canonical_reader_rejects_a_foreign_dataframe() -> None:
    """A non-canonical dataframe fails loudly instead of yielding junk."""
    with pytest.raises(KeyError, match="not canonical"):
        read_canonical(pd.DataFrame({"text": ["Something."]}))


# --- args_to_argrec: rows -> gold records ---


def test_build_record_carries_provenance_and_counter_ids() -> None:
    """A canonical row becomes a gold record with no carrier."""
    row = RawArgumentRow(
        id="0:arg", argument="Taxes.", counter_argument_ids=["0:counter"]
    )
    record = arg_to_argrec.build_record(row, "hitz", InterpretativeFrame.ECONOMIC)

    assert record.provenance.lookup_key == "hitz:0:arg"
    assert record.provenance.premises_source == PremiseSource.NONE
    assert record.native_counter_ids == ["0:counter"]
    assert record.carrier is None


def test_build_source_database_classifies_only_missing_frames() -> None:
    """Native frames are kept; rows without one go through the classifier."""
    classifier = StubAgent(
        FrameClassification(frame=InterpretativeFrame.HEALTH_AND_SAFETY, reason="ok")
    )
    rows = [
        RawArgumentRow(id="1", argument="A.", frame=InterpretativeFrame.MORALITY),
        RawArgumentRow(id="2", argument="Masks reduce spread."),
    ]
    db = arg_to_argrec.build_source_database("mixed", rows, classifier)

    assert [record.domain for record in db.arguments] == [
        InterpretativeFrame.MORALITY,
        InterpretativeFrame.HEALTH_AND_SAFETY,
    ]
    assert len(classifier.prompts) == 1


def test_build_source_database_requires_a_classifier_when_frames_are_missing() -> None:
    """A missing classifier is an error, not a silent OTHER label."""
    with pytest.raises(ValueError, match="frame classifier agent is required"):
        arg_to_argrec.build_source_database(
            "hitz", [RawArgumentRow(id="1", argument="A.")], classifier=None
        )


def test_premise_synthesis_accepts_only_well_supported_candidates() -> None:
    """Premises are kept when the judge's support clears the threshold."""
    supported = make_db(make_record("Remote work raises output."))
    accepted = arg_to_argrec.synthesize_missing_premises(
        supported,
        StubAgent(GeneratedPremises(premises=["Output rose.", "Costs fell."])),
        StubAgent(PremiseSupportVerdict(support=0.9)),
        StubScorer(0.0),
        threshold=0.7,
        retries=1,
    )
    record = supported.arguments[0]
    assert accepted == 1
    assert [premise.text for premise in record.premises] == [
        "Output rose.",
        "Costs fell.",
    ]
    assert record.provenance.premises_source == PremiseSource.SYNTHETIC_GENERATED
    assert record.provenance.premise_validation_score == pytest.approx(0.9)

    rejected = make_db(make_record("Remote work raises output."))
    judge = StubAgent(PremiseSupportVerdict(support=0.2))
    assert (
        arg_to_argrec.synthesize_missing_premises(
            rejected,
            StubAgent(GeneratedPremises(premises=["Unrelated."])),
            judge,
            StubScorer(0.0),
            threshold=0.7,
            retries=2,
        )
        == 0
    )
    assert rejected.arguments[0].premises == []
    assert rejected.arguments[0].provenance.premises_source == PremiseSource.NONE
    assert len(judge.prompts) == 3  # first attempt plus two regenerations


def test_premise_synthesis_drops_contradicting_candidates_before_the_judge() -> None:
    """A candidate set that contradicts its conclusion never costs a judge call."""
    db = make_db(make_record("Remote work raises output."))
    judge = StubAgent(PremiseSupportVerdict(support=1.0))

    assert (
        arg_to_argrec.synthesize_missing_premises(
            db,
            StubAgent(GeneratedPremises(premises=["Output collapsed."])),
            judge,
            StubScorer(0.99),
            threshold=0.7,
            contradiction_threshold=0.5,
            retries=0,
        )
        == 0
    )
    assert judge.prompts == []
    assert db.arguments[0].provenance.premises_source == PremiseSource.NONE


def test_premise_synthesis_force_keeps_the_best_rejected_candidates() -> None:
    """force=True never leaves a record bare, but marks it unvalidated."""
    db = make_db(make_record("Remote work raises output."))
    forced = arg_to_argrec.synthesize_missing_premises(
        db,
        StubAgent(GeneratedPremises(premises=["Weakly related."])),
        StubAgent(PremiseSupportVerdict(support=0.3)),
        StubScorer(0.0),
        threshold=0.7,
        retries=0,
        force=True,
    )
    record = db.arguments[0]
    assert forced == 1
    assert [premise.text for premise in record.premises] == ["Weakly related."]
    assert record.provenance.premises_source == PremiseSource.SYNTHETIC_UNVALIDATED
    assert record.provenance.premise_validation_score == pytest.approx(0.3)


def test_premise_synthesis_skips_records_that_already_have_premises() -> None:
    """Native premises are never overwritten by generated ones."""
    db = make_db(
        make_record(premises=["Native premise."], premises_source=PremiseSource.NATIVE)
    )
    generator = StubAgent(GeneratedPremises(premises=["Generated."]))

    judge = StubAgent(PremiseSupportVerdict(support=1.0))
    assert (
        arg_to_argrec.synthesize_missing_premises(db, generator, judge, StubScorer(0.0))
        == 0
    )
    assert generator.prompts == []


# --- merge_arg_dbs ---


def test_merge_without_judge_unions_and_links_native_ids() -> None:
    """A no-LLM merge combines the pools and rebuilds declared links only."""
    base = make_db(make_record("A", original_id="1"))
    incoming = make_db(
        make_record("C", source="src_b", original_id="1", counter_ids=["2"]),
        make_record("D", source="src_b", original_id="2"),
    )

    merged, report = merge_databases(base, incoming, judge=None)
    assert report.records == 3
    assert report.native_links == 1
    assert report.judged_links == 0
    assert len(merged) == 3


def test_merge_with_judge_links_across_databases() -> None:
    """The judge turns opposing candidates into typed links."""
    base = make_db(
        with_embeddings(
            make_record("Fares should stay frozen.", original_id="1"), [1.0, 0.0]
        )
    )
    incoming = make_db(
        with_embeddings(
            make_record(
                "Frozen fares starve the network.", source="src_b", original_id="1"
            ),
            [1.0, 0.0],
        )
    )
    judge = StubAgent(
        LinkVerdict(refutes=True, link_type=ArgumentLinkType.REBUTTAL, reason="opposes")
    )

    merged, report = merge_databases(base, incoming, judge=judge)
    assert report.judged_links == 1
    assert report.candidates_seen >= 1
    assert len(merged.argument_links) == 1
    assert next(iter(merged.argument_links)).link_type == ArgumentLinkType.REBUTTAL


def test_merge_ignores_a_judge_that_denies_the_pair() -> None:
    """A denied pair produces no link."""
    base = make_db(with_embeddings(make_record("A", original_id="1"), [1.0, 0.0]))
    incoming = make_db(
        with_embeddings(make_record("B", source="src_b", original_id="1"), [1.0, 0.0])
    )
    judge = StubAgent(LinkVerdict(refutes=False, reason="unrelated"))

    merged, report = merge_databases(base, incoming, judge=judge)
    assert report.judged_links == 0
    assert merged.argument_links == set()


def test_merge_is_idempotent() -> None:
    """Merging the same incoming database twice adds nothing the second time."""
    base = make_db(make_record("A", original_id="1"))
    incoming = make_db(make_record("B", source="src_b", original_id="1"))

    once, _ = merge_databases(base, incoming, judge=None)
    twice, report = merge_databases(once, incoming, judge=None)
    assert len(twice) == len(once)
    assert report.records == len(once)


# --- argrec_to_syntetic_sources ---


@pytest.fixture
def native_dir(tmp_path: Path) -> Path:
    """Create a native carrier directory with two real-looking documents."""
    directory = tmp_path / "native"
    directory.mkdir()
    (directory / "council.txt").write_text("\n\n".join(FILLER[:3]), encoding="utf-8")
    (directory / "review.txt").write_text("\n\n".join(FILLER[3:]), encoding="utf-8")
    return directory


def _linked_pair_db() -> ArgumentDatabase:
    """Gold database holding one linked argument pair."""
    anchor = make_record(
        "Fares should stay frozen.", original_id="1", counter_ids=["2"]
    )
    counter = make_record("Frozen fares starve the network.", original_id="2")
    db = make_db(anchor, counter)
    db.link_native_counters()
    return db


def test_filler_pool_keeps_prose_and_drops_page_furniture(tmp_path: Path) -> None:
    """Navigation labels never become carrier filler; an empty pool is an error."""
    directory = tmp_path / "noisy"
    directory.mkdir()
    (directory / "scraped.txt").write_text(
        "\n\n".join(["Afrikaans", "iOS", "Jump to content", FILLER[0]]),
        encoding="utf-8",
    )
    pool = synth.FillerPool(directory, cache_dir=tmp_path / "cache")
    assert pool.paragraphs_by_document["scraped"] == [FILLER[0]]

    empty = tmp_path / "furniture_only"
    empty.mkdir()
    (empty / "nav.txt").write_text("Afrikaans\n\niOS", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No usable native documents"):
        synth.FillerPool(empty, cache_dir=tmp_path / "cache")


def test_filler_comes_from_one_donor_as_a_contiguous_run(tmp_path: Path) -> None:
    """A carrier is never stitched from several documents, and keeps its thread."""
    directory = tmp_path / "native"
    directory.mkdir()
    for name in ("alpha", "beta"):
        paragraphs = [
            f"{FILLER[0]} This is {name} paragraph number {index} of the record."
            for index in range(12)
        ]
        (directory / f"{name}.txt").write_text(
            "\n\n".join(paragraphs), encoding="utf-8"
        )

    pool = synth.FillerPool(directory, cache_dir=tmp_path / "cache")
    rng = random.Random(3)
    for _ in range(20):
        paragraphs, donors = pool.sample(rng, 4)
        assert len(donors) == 1
        source = pool.paragraphs_by_document[donors[0]]
        start = source.index(paragraphs[0])
        assert paragraphs == source[start : start + 4]  # contiguous, in order


def test_filler_pool_routes_an_argument_to_its_topical_document(
    tmp_path: Path,
) -> None:
    """The keyword match prefers a donor that shares the argument's vocabulary."""
    directory = tmp_path / "native"
    directory.mkdir()
    topics = {
        "energy": "reactor fleet capacity decarbonisation electricity grid",
        "fisheries": "trawler quota spawning season harbour mesh netting",
    }
    for name, words in topics.items():
        (directory / f"{name}.txt").write_text(
            "\n\n".join(f"{FILLER[0]} {words} record {i}." for i in range(8)),
            encoding="utf-8",
        )

    pool = synth.FillerPool(directory, cache_dir=tmp_path / "cache")
    rng = random.Random(0)
    argument = "Reactor capacity keeps the electricity grid stable."
    assert {pool.sample(rng, 3, topic=argument)[1][0] for _ in range(10)} == {"energy"}

    # With nothing to match on, the choice stays spread across the corpus.
    assert {pool.sample(rng, 3)[1][0] for _ in range(40)} == set(topics)


def test_layout_axes_are_independent_and_respect_missing_premises() -> None:
    """Both axes vary with the seed; conclusion-only records cannot scatter."""
    bare = make_record()
    assert all(
        synth.sample_layout(random.Random(seed), [bare]).premise_dispersion
        == "together"
        for seed in range(10)
    )

    with_premises = make_record(
        premises=["Ridership fell.", "Subsidies were cut."],
        premises_source=PremiseSource.SYNTHETIC_GENERATED,
    )
    layouts = [
        synth.sample_layout(random.Random(seed), [with_premises]) for seed in range(30)
    ]
    assert len({layout.conclusion_position for layout in layouts}) > 1
    assert len({layout.premise_dispersion for layout in layouts}) > 1


def test_build_draft_places_conclusion_and_scatters_premises() -> None:
    """The conclusion lands where asked; scattered premises use distinct paragraphs."""
    record = make_record(
        premises=["Ridership fell.", "Subsidies were cut."],
        premises_source=PremiseSource.SYNTHETIC_GENERATED,
    )
    for position, expected in (("first", 0), ("middle", 2), ("last", 4)):
        layout = synth.DocumentLayout(
            conclusion_position=position, premise_dispersion="together"
        )
        paragraphs = synth.build_draft([record], FILLER, layout).split("\n\n")
        assert record.argument in paragraphs[expected]
        assert FILLER[expected] in paragraphs[expected]  # woven into real prose

    scattered = synth.build_draft(
        [record],
        FILLER,
        synth.DocumentLayout(
            conclusion_position="last", premise_dispersion="scattered"
        ),
    ).split("\n\n")
    holders = {
        index
        for index, paragraph in enumerate(scattered)
        if any(premise.text in paragraph for premise in record.premises)
    }
    assert len(holders) == 2


def test_style_document_is_discarded_when_it_loses_an_argument() -> None:
    """Styling that preserves the injected text is kept, otherwise rejected."""
    record = make_record()
    draft = synth.build_draft(
        [record],
        FILLER,
        synth.DocumentLayout(
            conclusion_position="first", premise_dispersion="together"
        ),
    )
    texts = synth.protected_texts([record])

    kept = f"A restyled opening. {record.argument} A restyled close."
    assert synth.style_document(StubAgent(StyledDocument(text=kept)), draft, texts) == (
        kept
    )
    lost = StubAgent(StyledDocument(text="An entirely different text about bees."))
    assert synth.style_document(lost, draft, texts) == draft


def test_generated_documents_are_reproducible_and_never_reuse_a_record(
    native_dir: Path, tmp_path: Path
) -> None:
    """The same seed yields the same carriers, and each record is placed once."""
    batch_dir = tmp_path / "batch"
    db = _linked_pair_db()
    entries = synth.generate_documents(
        db,
        synth.FillerPool(native_dir, cache_dir=tmp_path / "cache"),
        batch_dir,
        synth.GenerationSettings(document_count=4, seed=11),
    )

    used = [argument_id for entry in entries for argument_id in entry.argument_ids]
    assert len(used) == len(set(used))
    assert synth.pending_pool(db) == []

    placed = {record.id: record for record in db.arguments}
    for entry in entries:
        text = (batch_dir / f"{entry.document_id}.txt").read_text(encoding="utf-8")
        for argument_id in entry.argument_ids:
            carrier = placed[argument_id].carrier
            assert carrier is not None
            assert text[carrier.argument.start_idx : carrier.argument.end_idx] == (
                placed[argument_id].argument
            )

    repeat = synth.generate_documents(
        _linked_pair_db(),
        synth.FillerPool(native_dir, cache_dir=tmp_path / "cache"),
        tmp_path / "batch_again",
        synth.GenerationSettings(document_count=4, seed=11),
    )
    assert [entry.layout for entry in repeat] == [entry.layout for entry in entries]


def test_counter_pair_probability_controls_pairing(
    native_dir: Path, tmp_path: Path
) -> None:
    """Certainty puts both sides in one document; zero keeps them apart."""
    paired = synth.generate_documents(
        _linked_pair_db(),
        synth.FillerPool(native_dir, cache_dir=tmp_path / "cache"),
        tmp_path / "paired",
        synth.GenerationSettings(
            document_count=1, seed=2, counter_pair_probability=1.0
        ),
    )
    assert paired[0].contains_counter_pair
    assert len(paired[0].argument_ids) == 2

    apart = synth.generate_documents(
        _linked_pair_db(),
        synth.FillerPool(native_dir, cache_dir=tmp_path / "cache"),
        tmp_path / "apart",
        synth.GenerationSettings(
            document_count=2, seed=2, counter_pair_probability=0.0
        ),
    )
    assert all(not entry.contains_counter_pair for entry in apart)


def test_manifest_roundtrip_carries_layout_and_texts(
    native_dir: Path, tmp_path: Path
) -> None:
    """The manifest records what the evaluation slices need."""
    batch_dir = tmp_path / "batch"
    entries = synth.generate_documents(
        _linked_pair_db(),
        synth.FillerPool(native_dir, cache_dir=tmp_path / "cache"),
        batch_dir,
        synth.GenerationSettings(document_count=2, seed=9),
    )
    synth.write_manifest(batch_dir, entries)

    restored = synth.read_manifest(batch_dir)
    assert [entry.document_id for entry in restored] == [
        entry.document_id for entry in entries
    ]
    assert restored[0].source_kind == "synthetic"
    assert restored[0].filler_source_ids
    assert set(restored[0].argument_texts) == set(restored[0].argument_ids)


def test_dry_run_plans_documents_without_writing(
    native_dir: Path, tmp_path: Path
) -> None:
    """A dry run still places records but touches no files."""
    batch_dir = tmp_path / "batch"
    entries = synth.generate_documents(
        _linked_pair_db(),
        synth.FillerPool(native_dir, cache_dir=tmp_path / "cache"),
        None,
        synth.GenerationSettings(document_count=2, seed=1),
    )
    assert entries
    assert not batch_dir.exists()


# --- carriers_to_argdb ---


GOLD_TEXT = "Frozen fares keep the network accessible to low income riders."
CARRIER_TEXT = f"The council met on Tuesday. {GOLD_TEXT}\n\nOfficials declined."


@pytest.fixture
def corpus_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PipelineConfig:
    """Pipeline config with stubbed extractor and judge agents."""
    extracted = ExtractedArgument(
        argument=GOLD_TEXT,
        evidence=[ExtractedPremise(text="Ridership fell last year.")],
        frame=InterpretativeFrame.ECONOMIC,
    )
    agents = (
        StubAgent(ExtractionBatch(arguments=[extracted])),
        StubAgent(ExtractionVerdict(verdict="accept", reason="ok")),
        StubAgent(None),
    )
    monkeypatch.setattr(
        carrier_to_argdb, "make_agents", lambda _extractor, _judge: agents
    )

    config = PipelineConfig()
    config.native_sources_dir = tmp_path / "native"
    config.extracted_native_db_dir = tmp_path / "extracted_native"
    config.extracted_dbs_dir = tmp_path / "extracted"
    config.chunk_size = 4000
    config.chunk_overlap = 200
    return config.validated()


def test_native_extraction_has_no_ground_truth(
    corpus_cfg: PipelineConfig, tmp_path: Path
) -> None:
    """Native carriers yield records that claim no gold id."""
    corpus_cfg.native_sources_dir.mkdir(parents=True)
    (corpus_cfg.native_sources_dir / "harvested.txt").write_text(
        CARRIER_TEXT, encoding="utf-8"
    )

    db = ArgumentDatabase.load(
        carrier_to_argdb.run_native_sources(corpus_cfg), load_embeddings=False
    )
    assert db.arguments
    assert all(r.provenance.source_dataset == "native" for r in db.arguments)
    assert all(r.provenance.original_id is None for r in db.arguments)
    assert all(r.carrier is not None for r in db.arguments)


def test_synthetic_extraction_traces_back_to_gold(
    corpus_cfg: PipelineConfig, tmp_path: Path
) -> None:
    """With a manifest, extracted arguments carry the gold id they came from."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    (batch_dir / "doc_1.txt").write_text(CARRIER_TEXT, encoding="utf-8")
    record = make_record(GOLD_TEXT)
    synth.write_manifest(
        batch_dir,
        [
            synth.ManifestEntry(
                document_id="doc_1",
                argument_ids=[record.id],
                argument_texts={record.id: record.argument},
                layout=synth.DocumentLayout(
                    conclusion_position="middle", premise_dispersion="together"
                ),
                contains_counter_pair=False,
            )
        ],
    )

    target = carrier_to_argdb.run_synthetic_sources(corpus_cfg, "batch", batch_dir)
    db = ArgumentDatabase.load(target, load_embeddings=False)
    assert db.arguments[0].provenance.source_dataset == "synthetic:batch"
    assert db.arguments[0].provenance.original_id == record.id
    assert target == corpus_cfg.extracted_synthetic_db_dir("batch")


def test_gold_matching_rejects_unrelated_text() -> None:
    """Only a close text match counts as the injected gold argument."""
    gold_texts = {"gold-1": GOLD_TEXT}
    unrelated = ExtractedArgument(
        argument="Beekeeping requires careful hive maintenance.",
        frame=InterpretativeFrame.ECONOMIC,
    )
    assert carrier_to_argdb.match_gold_argument(unrelated, gold_texts) is None

    close = ExtractedArgument(argument=GOLD_TEXT, frame=InterpretativeFrame.ECONOMIC)
    assert carrier_to_argdb.match_gold_argument(close, gold_texts) == "gold-1"


def test_chunking_covers_the_whole_document() -> None:
    """Chunks tile the text with the requested overlap."""
    chunks = carrier_to_argdb.chunk_text("word " * 500, size=200, overlap=20)
    assert chunks[0].start == 0
    assert chunks[-1].end == len("word " * 500)
    for previous, current in itertools.pairwise(chunks):
        assert previous.end - 20 <= current.start < previous.end
