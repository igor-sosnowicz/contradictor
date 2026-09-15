"""
First sub-pipeline: raw argument dataframes into gold ArgumentDatabases.

Stage 1a:
- reads ``argument-dataset/arguments/compliant_datasets/<source>.parquet``
- assigns an interpretative frame (native label source has one, classifier otherwise)
- writes one ArgumentDatabase obj per source``argument-dataset/arguments/gold/<source>/``

Stage 1b:
Synthesises premises for records that arrived without any, keeping only the
ones that survive two gates: NLI rejects a candidate set that contradicts its
own conclusion, then the judge scores how well it supports it.
Generated premises are never marked NATIVE.

Counter-argument ids are copied onto the records as ``native_counter_ids`` but
NOT turned into links here: a source can only be linked against the pool it
ends up in. ``--merge-all`` folds the per-source databases together and calls
``link_native_counters()`` over the result.

Run with::

    uv run python -m src.argument_dataset.pipelines.arg_to_argrec \
        --source hitz_counter_argument
    uv run python -m src.argument_dataset.pipelines.arg_to_argrec --merge-all
"""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from src.argument_dataset.arg_db import (
    ARG_DB_JSON_FILENAME,
    ArgumentDatabase,
    ArgumentProvenance,
    ArgumentRecord,
    PremiseSource,
)
from src.argument_dataset.arg_link import ArgumentLinkType
from src.argument_dataset.backend import (
    make_frame_classifier,
    make_premise_generator,
    make_premise_judge,
    run,
)
from src.argument_dataset.config import PipelineConfig
from src.argument_dataset.datasets_handlers.raw_schema import (
    RawArgumentRow,
    read_canonical,
)
from src.argument_dataset.premice_scorer import EntailmentScorer, default_scorer
from src.argument_dataset.output_models import (
    ExtractedArgument,
    ExtractedPremise,
    FrameClassification,
    GeneratedPremises,
    PremiseSupportVerdict,
)
from src.argument_dataset.progress import track
from src.data_models.data_models import InterpretativeFrame

DEFAULT_COUNTER_LINK_TYPE = ArgumentLinkType.CONTRARGUMENT
REJECTION_ALARM_RATE = 0.35


# --- Raw sources ---


def available_sources(raw_dir: Path) -> list[str]:
    """List the source names that have a raw parquet file."""
    return sorted(path.stem for path in raw_dir.glob("*.parquet"))


def load_raw_rows(source_name: str, raw_dir: Path) -> list[RawArgumentRow]:
    """
    Load one source's already schema-compliant dataframe.

    ``fetch_datasets.py`` runs the source's adapter once, at fetch time, so
    the parquet here is canonical - read it directly rather than normalising
    it again (re-running a per-source adapter on already-canonical columns
    would silently misparse them).
    """
    path = raw_dir / f"{source_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Compliant dataset file not found at: {path}")

    frame: pd.DataFrame = pd.read_parquet(path)
    rows = read_canonical(frame)
    logger.info(f"{source_name}: loaded {len(rows)} argument rows")
    return rows


# --- Stage 1a ---


def classify_frame(
    classifier: Agent[None, FrameClassification], row: RawArgumentRow
) -> InterpretativeFrame:
    """Ask the classifier for the frame of an argument without a native label."""
    prompt = f"Argument:\n{row.argument}"
    if row.premises:
        prompt += "\n\nPremises:\n" + "\n".join(row.premises)

    try:
        verdict = run(classifier, prompt)
    except (UnexpectedModelBehavior, ValueError) as exc:
        logger.warning(f"Frame classification failed for row {row.id}: {exc}")
        return InterpretativeFrame.OTHER

    return verdict.frame


def build_record(
    row: RawArgumentRow, source_name: str, domain: InterpretativeFrame
) -> ArgumentRecord:
    """
    Turn one canonical row into a gold ArgumentRecord (no carrier).

    | 0:arg | argument | counter_argument_ids | premises | frame |
                              |
                              v
    | uuid | argument | premises | provenance(source, 0:arg) |
    | native_counter_ids ["0:counter"] | carrier=None |
    """
    extracted = ExtractedArgument(
        argument=row.argument,
        evidence=[ExtractedPremise(text=premise) for premise in row.premises],
        frame=domain,
        language=row.language,
    )
    provenance = ArgumentProvenance(
        source_dataset=source_name,
        original_id=row.id,
        premises_source=(
            PremiseSource.NATIVE if row.has_native_premises else PremiseSource.NONE
        ),
    )
    return ArgumentRecord.from_extracted(
        extracted,
        provenance=provenance,
        native_counter_ids=list(row.counter_argument_ids),
    )


def build_source_database(
    source_name: str,
    rows: list[RawArgumentRow],
    classifier: Agent[None, FrameClassification] | None = None,
) -> ArgumentDatabase:
    """Build the gold database of one raw source (Stage 1a)."""
    db = ArgumentDatabase()
    classified = 0

    needs_llm = sum(1 for row in rows if row.frame is None)  # Might need to extend this
    task = f"{source_name} | frames ({needs_llm} need classification)"
    for row in track(rows, task):
        domain: InterpretativeFrame | None = row.frame
        if domain is None:
            if classifier is None:
                raise ValueError(
                    f"Source {source_name!r} has no native frame label, so a frame "
                    f"classifier agent is required."
                )
            domain = classify_frame(classifier, row)
            classified += 1

        db += build_record(row, source_name, domain)

    logger.info(
        f"{source_name}: built {len(db.arguments)} records "
        f"({classified} frames classified by the LLM)"
    )
    return db


# --- Stage 1b ---


def _generate_premises(
    generator: Agent[None, GeneratedPremises], record: ArgumentRecord
) -> list[str]:
    """Ask the generator for candidate premises supporting one conclusion."""
    prompt = (
        f"Interpretative frame: {record.domain.value}\n"
        f"Language: {record.language}\n\n"
        f"Conclusion:\n{record.argument}"
    )
    try:
        generated = run(generator, prompt)
    except (UnexpectedModelBehavior, ValueError) as exc:
        logger.warning(f"Premise generation failed for record {record.id}: {exc}")
        return []

    return [premise.strip() for premise in generated.premises if premise.strip()]


def _support_score(
    candidates: list[str],
    record: ArgumentRecord,
    scorer: EntailmentScorer,
    judge: Agent[None, PremiseSupportVerdict],
    contradiction_threshold: float,
) -> float:
    """
    Score one candidate premise set against its conclusion, in two passes.

    | candidates | -> NLI contradiction? -> yes -> 0.0, never reaches the LLM
                              |
                              no
                              v
                     judge: how well do these support the claim? -> 0.0 .. 1.0

    The NLI pass is the cheap honesty guard: a premise that contradicts its own
    conclusion is fabricated ground truth. It cannot do the positive half of the
    job - real supporting premises score `neutral`, not `entailment` - so the
    judge answers that.
    """
    joined = " ".join(candidates)
    contradiction = scorer.contradiction(joined, record.argument)
    if contradiction > contradiction_threshold:
        logger.debug(
            f"Record {record.id}: candidates contradict the conclusion "
            f"({contradiction:.3f} > {contradiction_threshold})"
        )
        return 0.0

    prompt = f"Conclusion:\n{record.argument}\n\nPremises:\n" + "\n".join(
        f"- {premise}" for premise in candidates
    )
    try:
        verdict = run(judge, prompt)
    except (UnexpectedModelBehavior, ValueError) as exc:
        logger.warning(f"Premise judging failed for record {record.id}: {exc}")
        return 0.0

    return verdict.support


def _attach_premises(
    record: ArgumentRecord, premises: list[str], score: float, source: PremiseSource
) -> None:
    """Write an accepted premise set onto a record."""
    record.premises = [ExtractedPremise(text=premise) for premise in premises]
    record.provenance.premises_source = source
    record.provenance.premise_validation_score = score


def synthesize_missing_premises(
    db: ArgumentDatabase,
    generator: Agent[None, GeneratedPremises],
    judge: Agent[None, PremiseSupportVerdict],
    scorer: EntailmentScorer | None = None,
    threshold: float | None = None,
    contradiction_threshold: float | None = None,
    retries: int | None = None,
    *,
    force: bool = False,
) -> int:
    """
    Fill in premises for conclusion-only records, keeping only supported ones.

    A candidate set is accepted when it does not contradict its conclusion and
    the judge scores its support above ``threshold``. Records whose candidates
    never clear it keep ``PremiseSource.NONE`` rather than gaining fabricated
    ground truth.

    ``force=True`` is the "I need premises regardless" mode: the best candidate
    set is kept even when it failed the gate, marked
    ``PremiseSource.SYNTHETIC_UNVALIDATED`` so nothing downstream can mistake
    it for validated ground truth. Its score is still recorded.

    Returns the number of records that gained premises.
    """
    config = PipelineConfig()
    scorer = scorer or default_scorer()
    threshold = threshold if threshold is not None else config.premise_support_threshold
    contradiction_threshold = (
        contradiction_threshold
        if contradiction_threshold is not None
        else config.premise_contradiction_threshold
    )
    retries = retries if retries is not None else config.premise_synthesis_retries

    pending = [
        record
        for record in db.arguments
        if record.provenance.premises_source == PremiseSource.NONE
    ]
    logger.info(f"Premise synthesis: {len(pending)} conclusion-only record(s)")

    accepted = 0
    forced = 0
    for record in track(pending, "Premise synthesis"):
        best_premises: list[str] = []
        best_score = 0.0

        for _ in range(retries + 1):
            candidates = _generate_premises(generator, record)
            if not candidates:
                continue

            score = _support_score(
                candidates, record, scorer, judge, contradiction_threshold
            )
            if score > best_score or not best_premises:
                best_premises, best_score = candidates, score
            if best_score >= threshold:
                break

        if best_premises and best_score >= threshold:
            _attach_premises(
                record, best_premises, best_score, PremiseSource.SYNTHETIC_GENERATED
            )
            accepted += 1
        elif force and best_premises:
            _attach_premises(
                record, best_premises, best_score, PremiseSource.SYNTHETIC_UNVALIDATED
            )
            forced += 1
        else:
            logger.debug(
                f"Record {record.id}: premise synthesis rejected "
                f"(best support {best_score:.3f} < {threshold})"
            )

    _report_synthesis(len(pending), accepted, forced, threshold)
    return accepted + forced


def _report_synthesis(
    pending: int, accepted: int, forced: int, threshold: float
) -> None:
    """Log the outcome, loudly when the gate rejected nearly everything."""
    logger.info(
        f"Premise synthesis: {accepted} record(s) gained validated premises"
        + (f", {forced} forced through unvalidated" if forced else "")
    )
    if not pending:
        return

    rejection_rate = 1.0 - (accepted + forced) / pending
    if rejection_rate > REJECTION_ALARM_RATE:
        logger.warning(
            f"Premise synthesis rejected {rejection_rate:.0%} of {pending} record(s). "
            f"That usually means the gate is wrong, not the model: check that the "
            f"support threshold ({threshold}) suits the judge's scale before "
            f"trusting the run."
        )


# --- Drivers ---


def run_source(
    source_name: str,
    cfg: PipelineConfig,
    *,
    synthesize_premises: bool = True,
) -> ArgumentDatabase:
    """Build and save the gold database of one raw source."""
    rows = load_raw_rows(source_name, cfg.raw_arguments_dir)
    if cfg.limit is not None:
        rows = rows[: cfg.limit]

    needs_classifier = any(row.frame is None for row in rows)
    classifier = (
        make_frame_classifier(cfg.extractor_model or "") if needs_classifier else None
    )
    db = build_source_database(source_name, rows, classifier)

    if synthesize_premises:
        synthesize_missing_premises(
            db,
            make_premise_generator(cfg.extractor_model or ""),
            make_premise_judge(cfg.judge_model or ""),
            threshold=cfg.premise_support_threshold,
            contradiction_threshold=cfg.premise_contradiction_threshold,
            retries=cfg.premise_synthesis_retries,
            force=cfg.force_premises,
        )

    if not cfg.dry_run:
        target = cfg.gold_source_dir(source_name)
        db.save(target)
        logger.info(f"{source_name}: saved {len(db.arguments)} records to {target}")
    return db


def merge_gold_sources(
    cfg: PipelineConfig,
    link_type: ArgumentLinkType = DEFAULT_COUNTER_LINK_TYPE,
) -> ArgumentDatabase:
    """
    Combine every per-source gold ArgumentDatabase into one pool and link native ids.

    gold/hitz + gold/kialo + ... -> gold/merged

    Only the links the datasets themselves declared are built here. To also
    discover links BETWEEN sources, run ``pipelines/merge_arg_dbs.py``, which
    asks the LLM.
    """
    merged_dir = cfg.merged_gold_dir()
    source_dirs = [
        path
        for path in sorted(cfg.gold_arguments_dir.glob("*"))
        if path.is_dir() and path != merged_dir
    ]
    if not source_dirs:
        raise FileNotFoundError(
            f"No per-source gold databases found under {cfg.gold_arguments_dir}"
        )

    merged = ArgumentDatabase()
    for directory in source_dirs:
        merged = merged.union(ArgumentDatabase.load(directory))
        logger.info(f"Merged {directory.name} -> {len(merged.arguments)} records")

    linked = merged.link_native_counters(link_type=link_type)
    unlinked = sum(
        1
        for record in merged.arguments
        if record.native_counter_ids and not merged.counterparts_of(record)
    )

    logger.info(
        f"Merged gold database: {len(merged.arguments)} records, "
        f"{len(merged.argument_links)} links ({linked} from native ids), "
        f"{unlinked} record(s) whose native counter ids matched nothing"
    )

    if not cfg.dry_run:
        merged.save(merged_dir)
        logger.info(f"Saved merged gold database to {merged_dir}")
    return merged


def confirm_overwrite(source_name: str, target: Path) -> bool:
    """
    Ask before rebuilding a gold database that already exists on disk.

    A rebuild costs an LLM call per record, so a stray `make` is expensive.
    Anything but an explicit "y" skips the source.
    """
    if not (target / ARG_DB_JSON_FILENAME).exists():
        return True

    logger.warning(f"{source_name}: gold database already exists at {target}")
    try:
        answer = input(f"Rebuild {source_name}? [y/N] ").strip().lower()
    except EOFError:  # non-interactive run: never silently overwrite
        answer = ""

    if answer != "y":
        logger.info(f"{source_name}: skipped, kept the existing database")
        return False
    return True


def _apply_args(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Override config with CLI args."""
    if args.raw_dir is not None:
        cfg.raw_arguments_dir = args.raw_dir
    if args.gold_dir is not None:
        cfg.gold_arguments_dir = args.gold_dir
    if args.extractor_model is not None:
        cfg.extractor_model = args.extractor_model
    cfg.limit = args.limit
    cfg.dry_run = args.dry_run
    cfg.force_premises = args.force_premises
    return cfg.validated()


def main() -> None:
    """Build per-source gold databases and optionally merge them."""
    parser = argparse.ArgumentParser(
        description="Convert raw argument dataframes into gold ArgumentDatabases."
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Raw source name (repeatable). Defaults to every raw parquet found.",
    )
    parser.add_argument(
        "--merge-all",
        action="store_true",
        help="Merge per-source gold databases and resolve pending counter refs.",
    )
    parser.add_argument(
        "--force-premises",
        action="store_true",
        help=(
            "Never leave a record without premises: keep the best candidate set "
            "even when it fails the gate, marked synthetic_unvalidated."
        ),
    )
    parser.add_argument(
        "--no-premise-synthesis",
        action="store_true",
        help="Skip Stage 1b (leaves conclusion-only records without premises).",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--gold-dir", type=Path, default=None)
    parser.add_argument("--extractor-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _apply_args(PipelineConfig(), args)

    if not args.merge_all or args.source:
        sources = args.source or available_sources(cfg.raw_arguments_dir)
        if not sources:
            logger.warning(f"No raw sources found in {cfg.raw_arguments_dir}")
        for source_name in sources:
            if not cfg.dry_run and not confirm_overwrite(
                source_name, cfg.gold_source_dir(source_name)
            ):
                continue
            run_source(
                source_name, cfg, synthesize_premises=not args.no_premise_synthesis
            )

    if args.merge_all:
        merge_gold_sources(cfg)


if __name__ == "__main__":
    main()
