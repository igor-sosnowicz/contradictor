"""Unit tests for ArgumentRecord, carrier placement, union and native linking."""

from pathlib import Path

import numpy as np
import pytest
from fastembed import SparseEmbedding

from src.argument_dataset.arg_db import (
    ArgumentDatabase,
    ArgumentRecord,
    EmbeddingIdentifier,
    PremiseSource,
)
from src.argument_dataset.arg_link import ArgumentLinkType
from src.data_models.data_models import InterpretativeFrame
from tests.arguments_dataset.conftest import make_db, make_record

CARRIER = (
    "The council met on Tuesday. Fares should stay frozen. "
    "Ridership fell last year. Officials declined to comment."
)


def _embedded(record: ArgumentRecord, vector: list[float]) -> ArgumentRecord:
    """Give a record deterministic dense embeddings for retrieval tests."""
    dense = np.array(vector, dtype=np.float32)
    empty = SparseEmbedding(values=np.array([]), indices=np.array([]))
    record.argument_embedding = EmbeddingIdentifier(
        dense_embedding=dense, sparse_embedding=empty
    )
    record.target_claims_embedding = EmbeddingIdentifier(
        dense_embedding=dense, sparse_embedding=empty
    )
    return record


# --- Record shape ---


def test_gold_record_has_no_carrier() -> None:
    """A record built from a dataset carries no document placement."""
    record = make_record(counter_ids=["2"])
    assert record.carrier is None
    assert not record.is_placed
    assert record.native_counter_ids == ["2"]
    assert record.provenance.lookup_key == "src_a:1"


def test_lookup_key_is_none_without_original_id() -> None:
    """A record with no native id falls back to UUID identity."""
    assert make_record(original_id=None).provenance.lookup_key is None


# --- Carrier placement ---


def test_place_in_carrier_locates_argument_and_premises() -> None:
    """Placement records bounds that slice the exact texts back out."""
    record = make_record(premises=["Ridership fell last year."])
    assert record.place_in_carrier(CARRIER, "doc_1", batch_id="b1")

    carrier = record.carrier
    assert carrier is not None
    assert carrier.document_id == "doc_1"
    assert carrier.batch_id == "b1"
    assert record.is_placed
    assert CARRIER[carrier.argument.start_idx : carrier.argument.end_idx] == (
        record.argument
    )
    assert (
        CARRIER[carrier.premises[0].start_idx : carrier.premises[0].end_idx]
        == "Ridership fell last year."
    )


def test_place_in_carrier_refuses_a_document_without_the_argument() -> None:
    """A record that cannot be located stays unplaced instead of storing junk."""
    record = make_record()
    assert not record.place_in_carrier("Completely unrelated prose.", "doc_1")
    assert record.carrier is None


def test_find_text_bounds_reports_missing_text() -> None:
    """Text that is absent yields bounds that are not located."""
    assert not ArgumentRecord.find_text_bounds("nope", CARRIER).is_located
    assert ArgumentRecord.find_text_bounds("", CARRIER).is_located is False


# --- Native counter links ---


def test_link_native_counters_links_declared_pairs() -> None:
    """Ids the dataset provided become links; unmatched ids create none."""
    left = make_record("A", original_id="1", counter_ids=["2"])
    right = make_record("B", original_id="2", counter_ids=["1"])
    stray = make_record("C", original_id="3", counter_ids=["404"])
    db = make_db(left, right, stray)

    assert db.link_native_counters() == 1  # one undirected edge for the pair
    assert [r.provenance.original_id for r in db.counterparts_of(left)] == ["2"]
    assert db.counterparts_of(stray) == []


def test_link_native_counters_stays_inside_one_source() -> None:
    """The same native id in a different dataset is not a match."""
    left = make_record("A", source="src_a", original_id="1", counter_ids=["2"])
    other = make_record("B", source="src_b", original_id="2")
    db = make_db(left, other)

    assert db.link_native_counters() == 0
    assert db.counterparts_of(left) == []


# --- Union ---


def test_union_deduplicates_by_provenance_and_keeps_links() -> None:
    """Merging a rebuild of the same source updates instead of doubling."""
    left = make_record("A", original_id="1", counter_ids=["2"])
    right = make_record("B", original_id="2")
    db = make_db(left, right)
    db.link_native_counters()

    rebuild = make_db(
        make_record("A", original_id="1", counter_ids=["2"]),
        make_record("B", original_id="2"),
    )
    merged = db.union(rebuild)

    assert len(merged) == 2
    assert len(merged.argument_links) == 1
    assert len(db) == 2  # operands untouched


def test_union_adds_a_new_source_and_remaps_its_links() -> None:
    """Records from another source are added, their links following along."""
    base = make_db(make_record("A", original_id="1"))
    new_left = make_record("C", source="src_b", original_id="1", counter_ids=["2"])
    new_right = make_record("D", source="src_b", original_id="2")
    incoming = make_db(new_left, new_right)
    incoming.link_native_counters()

    merged = base.union(incoming)
    assert len(merged) == 3
    assert len(merged.argument_links) == 1

    kept_ids = {record.id for record in merged.arguments}
    link = next(iter(merged.argument_links))
    assert link.source_argument_id in kept_ids
    assert link.target_argument_id in kept_ids


def test_union_keeps_records_without_a_native_id_apart() -> None:
    """Records with no original_id keep UUID identity and are never collapsed."""
    merged = make_db(make_record("A", original_id=None)).union(
        make_db(make_record("A", original_id=None))
    )
    assert len(merged) == 2


# --- Retrieval ---


def test_similar_and_opposing_searches_look_in_opposite_directions() -> None:
    """Similarity compares arguments; opposition compares target claims."""
    query = make_record("A", original_id="1")
    query.target_claims_hypothesis = ["Fares must rise."]
    _embedded(query, [1.0, 0.0])
    query.target_claims_embedding = EmbeddingIdentifier(
        dense_embedding=np.array([0.0, 1.0], dtype=np.float32),
        sparse_embedding=SparseEmbedding(values=np.array([]), indices=np.array([])),
    )

    twin = _embedded(make_record("B", original_id="2"), [1.0, 0.0])
    opponent = _embedded(make_record("C", original_id="3"), [0.0, 1.0])
    db = make_db(query, twin, opponent)

    assert db.fetch_similar_arguments(query, top_k=1)[0][0].id == twin.id
    assert db.fetch_opposing_arguments(query, top_k=1)[0][0].id == opponent.id


def test_retrieval_filters_self_language_and_domain() -> None:
    """The query itself, other languages and other domains are excluded."""
    query = _embedded(make_record("A", original_id="1"), [1.0, 0.0])
    other_domain = _embedded(
        make_record("B", original_id="2", domain=InterpretativeFrame.MORALITY),
        [1.0, 0.0],
    )
    other_language = _embedded(make_record("C", original_id="3"), [1.0, 0.0])
    other_language.language = "polish"
    db = make_db(query, other_domain, other_language)

    matches = db.fetch_similar_arguments(
        query, domain_filter={query.domain: [query.domain]}
    )
    assert matches == []


# --- Links guard rails ---


def test_link_guards_reject_unknown_ids_and_duplicates() -> None:
    """Links need known records; try_add_link reports duplicates instead of raising."""
    record, other = make_record("A", original_id="1"), make_record("B", original_id="2")
    db = make_db(record)

    with pytest.raises(ValueError, match="Target argument not found"):
        db.add_argument_link(record, other, ArgumentLinkType.REBUTTAL)

    db += other
    assert db.try_add_link(record, other, ArgumentLinkType.REBUTTAL)
    assert not db.try_add_link(record, other, ArgumentLinkType.REBUTTAL)
    assert not db.try_add_link(record, record, ArgumentLinkType.REBUTTAL)


# --- Persistence ---


def test_save_load_roundtrip_keeps_provenance_counters_and_carrier(
    tmp_path: Path,
) -> None:
    """Every field that survives a rebuild also survives a save/load cycle."""
    record = make_record(
        source="hitz",
        original_id="0:arg",
        counter_ids=["0:counter"],
        premises=["Ridership fell last year."],
        premises_source=PremiseSource.SYNTHETIC_GENERATED,
    )
    record.provenance.premise_validation_score = 0.84
    record.place_in_carrier(CARRIER, "doc_1", batch_id="b1")
    partner = make_record("B", source="hitz", original_id="0:counter")
    db = make_db(record, partner)
    db.link_native_counters()

    db.save(tmp_path / "db")
    reloaded = ArgumentDatabase.load(tmp_path / "db", load_embeddings=False)
    restored = reloaded.arguments[0]

    assert restored.provenance.lookup_key == "hitz:0:arg"
    assert restored.provenance.premises_source == PremiseSource.SYNTHETIC_GENERATED
    assert restored.provenance.premise_validation_score == pytest.approx(0.84)
    assert restored.native_counter_ids == ["0:counter"]
    assert restored.carrier is not None
    assert restored.carrier.document_id == "doc_1"
    assert restored.carrier.batch_id == "b1"
    assert len(reloaded.argument_links) == 1


def test_load_missing_directory_raises(tmp_path: Path) -> None:
    """Loading a database that was never saved fails loudly."""
    with pytest.raises(FileNotFoundError):
        ArgumentDatabase.load(tmp_path / "nope")
