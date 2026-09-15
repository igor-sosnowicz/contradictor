"""Unit tests for the entailment gate and the synthetic-batch validation."""

from pathlib import Path

import numpy as np
import pytest

from src.argument_dataset import validation
from src.argument_dataset.arg_db import (
    ArgumentProvenance,
    ArgumentRecord,
    PremiseSource,
)
from src.argument_dataset.premice_scorer import CrossEncoderEntailmentScorer
from src.argument_dataset.pipelines import argrec_to_carrier as synth
from tests.arguments_dataset.conftest import make_db, make_record

GOLD_TEXT = "Frozen fares keep the network accessible to low income riders."
CARRIER = f"The council met on Tuesday. {GOLD_TEXT}\n\nOfficials declined."


class FakeCrossEncoder:
    """Stand-in for the sentence-transformers cross-encoder."""

    def __init__(self, logits: list[list[float]], labels: dict[int, str]) -> None:
        """Store canned logits and the model's label mapping."""
        self.config = type("Config", (), {"id2label": labels})()
        self._logits = logits

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Return the canned logits for the given pairs."""
        return np.array(self._logits[: len(pairs)], dtype=np.float32)


# --- Entailment gate ---


def test_nli_scorer_reads_each_label_column_by_name() -> None:
    """Scores come from the model's own labels, not from fixed indices."""
    scorer = CrossEncoderEntailmentScorer()
    # entailment sits at index 1 here, contradiction at 0
    scorer._model = FakeCrossEncoder(
        [[-3.0, 5.0, -1.0]], {0: "contradiction", 1: "entailment", 2: "neutral"}
    )
    scorer._label_index = {"entailment": 1, "contradiction": 0}
    assert scorer.score("Output rose 15%.", "Remote work raises output.") > 0.9
    assert scorer.contradiction("Output rose 15%.", "Output fell.") < 0.1

    # a model that orders its labels differently must still be read correctly
    reordered = CrossEncoderEntailmentScorer()
    reordered._model = FakeCrossEncoder(
        [[5.0, -3.0, -1.0]], {0: "entailment", 1: "contradiction", 2: "neutral"}
    )
    reordered._label_index = {"entailment": 0, "contradiction": 1}
    assert reordered.score("a", "b") > 0.9
    assert reordered.contradiction("a", "b") < 0.1


def test_nli_scorer_handles_no_pairs() -> None:
    """Scoring nothing returns nothing rather than touching the model."""
    assert CrossEncoderEntailmentScorer().score_many([]) == []
    assert CrossEncoderEntailmentScorer().contradiction_many([]) == []


# --- Batch validation ---


@pytest.fixture
def batch_dir(tmp_path: Path) -> Path:
    """Create a synthetic batch directory holding one carrier document."""
    directory = tmp_path / "batch"
    directory.mkdir()
    (directory / "doc_1.txt").write_text(CARRIER, encoding="utf-8")
    return directory


def _placed_record(
    premises: list[str] | None = None, **kwargs: object
) -> ArgumentRecord:
    """Build a gold record already placed inside CARRIER."""
    record = make_record(GOLD_TEXT, premises=premises, **kwargs)  # type: ignore[arg-type]
    record.place_in_carrier(CARRIER, "doc_1", batch_id="b1")
    return record


def _manifest(record: ArgumentRecord, **overrides: object) -> synth.ManifestEntry:
    """Build the manifest entry describing the single carrier document."""
    fields: dict[str, object] = {
        "document_id": "doc_1",
        "argument_ids": [record.id],
        "argument_texts": {record.id: record.argument},
        "layout": synth.DocumentLayout(
            conclusion_position="middle", premise_dispersion="together"
        ),
        "contains_counter_pair": False,
    }
    fields.update(overrides)
    return synth.ManifestEntry(**fields)  # type: ignore[arg-type]


def _save_gold(tmp_path: Path, record: ArgumentRecord) -> Path:
    """Persist a one-record gold database and return its directory."""
    gold_dir = tmp_path / "gold"
    make_db(record).save(gold_dir)
    return gold_dir


def test_consistent_batch_passes_validation(batch_dir: Path, tmp_path: Path) -> None:
    """A batch whose placements and provenance agree reports no issues."""
    record = _placed_record()
    synth.write_manifest(batch_dir, [_manifest(record)])

    report = validation.validate_batch(batch_dir, _save_gold(tmp_path, record))
    assert report.ok
    assert report.documents == 1
    assert report.injected_arguments == 1


def test_validation_catches_broken_ground_truth(
    batch_dir: Path, tmp_path: Path
) -> None:
    """Wrong bounds, unscored generated premises and bad layouts are all reported."""
    wrong_bounds = _placed_record()
    assert wrong_bounds.carrier is not None
    wrong_bounds.carrier.argument.start_idx = 0
    wrong_bounds.carrier.argument.end_idx = 5
    synth.write_manifest(batch_dir, [_manifest(wrong_bounds)])
    report = validation.validate_batch(batch_dir, _save_gold(tmp_path, wrong_bounds))
    assert any(issue.check == "offsets_match_text" for issue in report.issues)

    unscored = _placed_record(
        premises=["Ridership fell."], premises_source=PremiseSource.SYNTHETIC_GENERATED
    )
    synth.write_manifest(batch_dir, [_manifest(unscored)])
    report = validation.validate_batch(batch_dir, _save_gold(tmp_path, unscored))
    assert any(issue.check == "generated_premises_scored" for issue in report.issues)

    bare = _placed_record()
    synth.write_manifest(
        batch_dir,
        [
            _manifest(
                bare,
                layout=synth.DocumentLayout(
                    conclusion_position="last", premise_dispersion="scattered"
                ),
            )
        ],
    )
    report = validation.validate_batch(batch_dir, _save_gold(tmp_path, bare))
    assert any(issue.check == "scattered_needs_premises" for issue in report.issues)


def test_validation_flags_missing_documents_and_unlinked_counters(
    batch_dir: Path, tmp_path: Path
) -> None:
    """A manifest naming an absent file, and dangling native ids, are surfaced."""
    record = _placed_record()
    synth.write_manifest(batch_dir, [_manifest(record, document_id="doc_missing")])
    report = validation.validate_batch(batch_dir, _save_gold(tmp_path, record))
    assert any(issue.check == "document_exists" for issue in report.issues)

    dangling = _placed_record(counter_ids=["404"])
    synth.write_manifest(batch_dir, [_manifest(dangling)])
    report = validation.validate_batch(batch_dir, _save_gold(tmp_path, dangling))
    assert any(issue.check == "unlinked_native_counter" for issue in report.issues)


def test_evaluation_scores_recall_per_slice(batch_dir: Path, tmp_path: Path) -> None:
    """A found argument scores full recall; a missed one scores zero."""
    record = _placed_record()
    synth.write_manifest(batch_dir, [_manifest(record)])
    gold_dir = _save_gold(tmp_path, record)

    found = make_record(GOLD_TEXT)
    found.provenance = ArgumentProvenance(
        source_dataset="synthetic:b1", original_id=record.id
    )
    found.place_in_carrier(CARRIER, "doc_1")
    extracted_dir = tmp_path / "extracted"
    make_db(found).save(extracted_dir)

    report = validation.validate_batch(batch_dir, gold_dir, extracted_dir=extracted_dir)
    slices = {(item.dimension, item.value): item for item in report.slices}
    assert slices[("conclusion_position", "middle")].recall == pytest.approx(1.0)
    assert slices[("premises_source", "none")].recall == pytest.approx(1.0)
    # premise origin is a gold property, so precision is undefined there
    assert slices[("premises_source", "none")].precision is None

    missed = make_record("Something else entirely.")
    missed.provenance = ArgumentProvenance(source_dataset="synthetic:b1")
    missed.place_in_carrier("Something else entirely.", "doc_1")
    missed_dir = tmp_path / "missed"
    make_db(missed).save(missed_dir)

    report = validation.validate_batch(batch_dir, gold_dir, extracted_dir=missed_dir)
    slices = {(item.dimension, item.value): item for item in report.slices}
    assert slices[("conclusion_position", "middle")].recall == pytest.approx(0.0)
    assert slices[("conclusion_position", "middle")].precision == pytest.approx(0.0)


def test_validation_without_an_extracted_database_still_checks_integrity(
    batch_dir: Path, tmp_path: Path
) -> None:
    """Integrity runs before the extractor was ever pointed at the batch."""
    record = _placed_record()
    synth.write_manifest(batch_dir, [_manifest(record)])
    report = validation.validate_batch(batch_dir, _save_gold(tmp_path, record))
    assert report.slices == []
    assert report.ok
    assert "Integrity" in report.render()
