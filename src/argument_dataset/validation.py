"""
Validation and evaluation of a synthetic carrier batch.

Two jobs, deliberately kept apart:

- *Integrity checks* confirm the golden dataset says what it claims: injected
  arguments really sit where their offsets point, generated premises really
  carry a validation score, conclusion-only records never ended up in a
  scattered layout, and no counter reference was quietly dropped.

- *Evaluation* compares the extracted database against the gold pool through
  ``provenance.original_id`` and reports precision/recall **per slice**
  (premise origin, conclusion position, premise dispersion, counter pairs).
  A single aggregate number would hide where the extractor is actually weak,
  which is the whole reason provenance is tracked.

Gold and extracted databases are never merged: they are reference truth and
system output, and the comparison is what keeps them honest.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from src.argument_dataset.arg_db import ArgumentDatabase, ArgumentRecord, PremiseSource
from src.argument_dataset.pipelines.argrec_to_carrier import (
    MIN_MATCH_SCORE,
    ManifestEntry,
    read_manifest,
)

PREMISES_SOURCE_SLICE = "premises_source"
CONCLUSION_POSITION_SLICE = "conclusion_position"
PREMISE_DISPERSION_SLICE = "premise_dispersion"
COUNTER_PAIR_SLICE = "contains_counter_pair"


class ValidationIssue(BaseModel):
    """One failed integrity check."""

    check: str
    subject_id: str
    detail: str


class SliceMetrics(BaseModel):
    """Extraction quality on one slice of the injected gold arguments."""

    dimension: str
    value: str
    gold_total: int = 0
    gold_matched: int = 0
    extracted_total: int | None = None
    extracted_matched: int | None = None

    @property
    def recall(self) -> float | None:
        """Share of injected gold arguments the extractor found."""
        if self.gold_total == 0:
            return None
        return self.gold_matched / self.gold_total

    @property
    def precision(self) -> float | None:
        """Share of extracted arguments that map back to an injected gold one."""
        if not self.extracted_total:
            return None
        return (self.extracted_matched or 0) / self.extracted_total


class ValidationReport(BaseModel):
    """Outcome of validating and evaluating one synthetic batch."""

    batch_id: str
    documents: int = 0
    injected_arguments: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)
    slices: list[SliceMetrics] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every integrity check passed."""
        return not self.issues

    def render(self) -> str:
        """Format the report for a terminal."""
        lines: list[str] = [
            f"Batch {self.batch_id}: {self.documents} document(s), "
            f"{self.injected_arguments} injected argument(s)",
            f"Integrity: {'OK' if self.ok else f'{len(self.issues)} issue(s)'}",
        ]
        lines.extend(
            f"  [{issue.check}] {issue.subject_id}: {issue.detail}"
            for issue in self.issues
        )

        if self.slices:
            lines.append("Extraction quality by slice:")
            for entry in self.slices:
                recall = "n/a" if entry.recall is None else f"{entry.recall:.2f}"
                precision = (
                    "n/a" if entry.precision is None else f"{entry.precision:.2f}"
                )
                lines.append(
                    f"  {entry.dimension}={entry.value}: "
                    f"recall {recall} ({entry.gold_matched}/{entry.gold_total}), "
                    f"precision {precision}"
                )
        return "\n".join(lines)


# --- Integrity checks ---


def check_injected_offsets(
    entry: ManifestEntry, document_text: str, gold_by_id: dict[str, ArgumentRecord]
) -> list[ValidationIssue]:
    """Confirm each injected argument exists in gold and sits where it claims."""
    issues: list[ValidationIssue] = []
    for argument_id in entry.argument_ids:
        record = gold_by_id.get(argument_id)
        if record is None:
            issues.append(
                ValidationIssue(
                    check="argument_in_gold",
                    subject_id=argument_id,
                    detail=f"listed in {entry.document_id} but missing from gold",
                )
            )
            continue

        if record.carrier is None:
            issues.append(
                ValidationIssue(
                    check="record_is_placed",
                    subject_id=argument_id,
                    detail=(
                        f"listed in {entry.document_id} but carries no placement"
                    ),
                )
            )
            continue

        bounds = record.carrier.argument
        sliced = document_text[bounds.start_idx : bounds.end_idx]
        if not sliced or not ArgumentRecord.find_text_bounds(
            record.argument, sliced, MIN_MATCH_SCORE
        ).is_located:
            issues.append(
                ValidationIssue(
                    check="offsets_match_text",
                    subject_id=argument_id,
                    detail=(
                        f"{bounds.start_idx}:{bounds.end_idx} in "
                        f"{entry.document_id} does not match the argument text"
                    ),
                )
            )
    return issues


def check_premise_provenance(
    entry: ManifestEntry, gold_by_id: dict[str, ArgumentRecord], threshold: float
) -> list[ValidationIssue]:
    """Confirm generated premises are scored and scattered layouts have premises."""
    issues: list[ValidationIssue] = []
    for argument_id in entry.argument_ids:
        record = gold_by_id.get(argument_id)
        if record is None:
            continue

        source = record.provenance.premises_source
        score = record.provenance.premise_validation_score
        if source == PremiseSource.SYNTHETIC_GENERATED and (
            score is None or score < threshold
        ):
            issues.append(
                ValidationIssue(
                    check="generated_premises_scored",
                    subject_id=argument_id,
                    detail=(
                        f"premises_source is synthetic_generated but the "
                        f"validation score is {score}"
                    ),
                )
            )

        if entry.layout.premise_dispersion == "scattered" and (
            source == PremiseSource.NONE
        ):
            issues.append(
                ValidationIssue(
                    check="scattered_needs_premises",
                    subject_id=argument_id,
                    detail=(
                        f"conclusion-only record placed in the scattered layout "
                        f"of {entry.document_id}"
                    ),
                )
            )
    return issues


def check_unlinked_native_counters(gold: ArgumentDatabase) -> list[ValidationIssue]:
    """Report records whose dataset-provided counter ids matched no record."""
    return [
        ValidationIssue(
            check="unlinked_native_counter",
            subject_id=record.id,
            detail=(
                f"native counter ids {record.native_counter_ids} matched nothing "
                f"in the pool - needs manual review"
            ),
        )
        for record in gold.arguments
        if record.native_counter_ids and not gold.counterparts_of(record)
    ]


# --- Evaluation ---


def _slice_values(
    record: ArgumentRecord, entry: ManifestEntry
) -> list[tuple[str, str]]:
    """List the (dimension, value) slices one injected argument belongs to."""
    return [
        (PREMISES_SOURCE_SLICE, record.provenance.premises_source.value),
        (CONCLUSION_POSITION_SLICE, entry.layout.conclusion_position),
        (PREMISE_DISPERSION_SLICE, entry.layout.premise_dispersion),
        (COUNTER_PAIR_SLICE, str(entry.contains_counter_pair).lower()),
    ]


def evaluate_extraction(
    entries: list[ManifestEntry],
    gold_by_id: dict[str, ArgumentRecord],
    extracted: ArgumentDatabase,
) -> list[SliceMetrics]:
    """
    Score the extraction pipeline against the injected gold arguments, per slice.

    Recall is measured over injected gold arguments. Precision is measured over
    extracted records, attributed to a slice through the document they came
    from; premise origin is a property of the gold record, so precision is
    undefined there and reported as n/a rather than as a misleading number.
    """
    entry_by_document = {entry.document_id: entry for entry in entries}
    matched_gold_ids = {
        record.provenance.original_id
        for record in extracted.arguments
        if record.provenance.original_id
    }

    metrics: dict[tuple[str, str], SliceMetrics] = {}

    def bucket(dimension: str, value: str) -> SliceMetrics:
        return metrics.setdefault(
            (dimension, value), SliceMetrics(dimension=dimension, value=value)
        )

    for entry in entries:
        for argument_id in entry.argument_ids:
            record = gold_by_id.get(argument_id)
            if record is None:
                continue

            found = argument_id in matched_gold_ids
            for dimension, value in _slice_values(record, entry):
                target = bucket(dimension, value)
                target.gold_total += 1
                target.gold_matched += int(found)

    for record in extracted.arguments:
        if record.carrier is None:
            continue
        carrier = entry_by_document.get(record.carrier.document_id)
        if carrier is None:
            continue

        traced = int(record.provenance.original_id is not None)
        for dimension, value in (
            (CONCLUSION_POSITION_SLICE, carrier.layout.conclusion_position),
            (PREMISE_DISPERSION_SLICE, carrier.layout.premise_dispersion),
            (COUNTER_PAIR_SLICE, str(carrier.contains_counter_pair).lower()),
        ):
            target = bucket(dimension, value)
            target.extracted_total = (target.extracted_total or 0) + 1
            target.extracted_matched = (target.extracted_matched or 0) + traced

    return [metrics[key] for key in sorted(metrics)]


# --- Driver ---


def validate_batch(
    batch_dir: Path,
    gold_dir: Path,
    batch_id: str | None = None,
    premise_threshold: float = 0.7,
    extracted_dir: Path | None = None,
) -> ValidationReport:
    """Run every integrity check and the per-slice evaluation on one batch."""
    entries = read_manifest(batch_dir)
    gold = ArgumentDatabase.load(gold_dir, load_embeddings=False)
    gold_by_id = {record.id: record for record in gold.arguments}

    report = ValidationReport(
        batch_id=batch_id or batch_dir.name,
        documents=len(entries),
        injected_arguments=sum(len(entry.argument_ids) for entry in entries),
    )

    for entry in entries:
        document_path = batch_dir / f"{entry.document_id}.txt"
        if not document_path.exists():
            report.issues.append(
                ValidationIssue(
                    check="document_exists",
                    subject_id=entry.document_id,
                    detail=f"manifest lists {document_path.name}, which is missing",
                )
            )
            continue

        document_text = document_path.read_text(encoding="utf-8")
        report.issues.extend(check_injected_offsets(entry, document_text, gold_by_id))
        report.issues.extend(
            check_premise_provenance(entry, gold_by_id, premise_threshold)
        )

    report.issues.extend(check_unlinked_native_counters(gold))

    if extracted_dir is not None and extracted_dir.is_dir():
        extracted = ArgumentDatabase.load(extracted_dir, load_embeddings=False)
        report.slices = evaluate_extraction(entries, gold_by_id, extracted)

    return report
