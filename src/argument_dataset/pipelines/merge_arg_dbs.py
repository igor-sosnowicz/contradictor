"""
Merge two ArgumentDatabase objects into one, rebuilding the links.

Typical use-case:
1. You found a new dataset.
2. You fetched it
3. You parsed it
4. You ran args_to_argrec over it. Now you want it inside the pool you already have,
and the two pools must end up knowing which of their arguments attack each other.

    old arg_db              new arg_db
        |                       |
        +----------+------------+
                   v
        1. union()               deduplicated by "<source>:<original_id>"
                   v
        2. link_native_counters()    ids the datasets themselves provided
                   v
        3. LLM cross-linking         for every record still to be linked:
                   |                   fetch_opposing_arguments() -> candidates
                   |                   judge picks the real counters
                   v                   and names the ArgumentLinkType
        4. save()

Step 3 is the expensive one, so by default it only runs for records from the
new database (old <-> old links already exist). Pass --relink-all to redo the
whole pool.

Run with::

    uv run python -m src.argument_dataset.pipelines.merge_arg_dbs \
        --into arguments/gold/merged --new arguments/gold/kialo
"""

import argparse
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from src.argument_dataset.arg_db import ArgumentDatabase, ArgumentRecord
from src.argument_dataset.backend import make_agents, run
from src.argument_dataset.config import PipelineConfig
from src.argument_dataset.output_models import LinkVerdict
from src.argument_dataset.progress import track


class MergeReport(BaseModel):
    """What one merge run changed."""

    records: int = Field(default=0, description="Records in the merged database")
    native_links: int = Field(default=0, description="Links from dataset-provided ids")
    judged_links: int = Field(default=0, description="Links confirmed by the judge")
    candidates_seen: int = Field(default=0, description="Candidate pairs judged")

    def render(self) -> str:
        """One-line summary for the log."""
        return (
            f"{self.records} records, {self.native_links} native links, "
            f"{self.judged_links}/{self.candidates_seen} judged links confirmed"
        )


def _link_prompt(source: ArgumentRecord, target: ArgumentRecord) -> str:
    """Ask whether the source argument refutes the candidate."""
    return (
        f"Argument A ({source.domain.value}):\n{source.argument}\n\n"
        f"A's target-claim hypothesis: "
        f"{'; '.join(source.target_claims_hypothesis)}\n\n"
        f"Argument B ({target.domain.value}):\n{target.argument}\n\n"
        f"B's target-claim hypothesis: "
        f"{'; '.join(target.target_claims_hypothesis)}\n\n"
        "Does A refute B or B's hypothesis?"
    )


def judge_links(
    db: ArgumentDatabase,
    records: list[ArgumentRecord],
    judge: Agent[None, LinkVerdict],
    top_k: int = 5,
) -> tuple[int, int]:
    """
    Discover links the datasets never declared, one record at a time.

    For each record: pull its opposing candidates out of the pool, let the
    judge accept or reject each pair and name the relation.

    Returns (confirmed links, candidates judged).
    """
    confirmed = 0
    seen = 0

    for record in track(records, "Link judging"):
        for candidate, score in db.fetch_opposing_arguments(record, top_k=top_k):
            seen += 1
            try:
                verdict = run(judge, _link_prompt(record, candidate))
            except (UnexpectedModelBehavior, ValueError) as exc:
                logger.warning(f"Link judging failed for {record.id}: {exc}")
                continue

            if not verdict.refutes or verdict.link_type is None:
                continue

            hypothesis = verdict.repaired_target_hypothesis
            if hypothesis and hypothesis not in record.target_claims_hypothesis:
                record.target_claims_hypothesis.append(hypothesis)

            if db.try_add_link(record, candidate, verdict.link_type, score):
                confirmed += 1

    return confirmed, seen


def merge_databases(
    base: ArgumentDatabase,
    incoming: ArgumentDatabase,
    judge: Agent[None, LinkVerdict] | None = None,
    top_k: int = 5,
    *,
    relink_all: bool = False,
) -> tuple[ArgumentDatabase, MergeReport]:
    """
    Combine two databases and rebuild the links across them.

    Without a judge, only the dataset-provided links are rebuilt (no LLM).
    """
    incoming_identities = {record.identity for record in incoming.arguments}
    merged = base.union(incoming)
    report = MergeReport(records=len(merged))
    report.native_links = merged.link_native_counters()

    if judge is None:
        return merged, report

    to_link = (
        merged.arguments
        if relink_all
        else [
            record
            for record in merged.arguments
            if record.identity in incoming_identities
        ]
    )
    logger.info(f"Judging opposing candidates for {len(to_link)} record(s)")
    report.judged_links, report.candidates_seen = judge_links(
        merged, to_link, judge, top_k
    )
    return merged, report


class MergePaths(BaseModel):
    """Which databases a merge run reads and where it writes the result."""

    base: Path = Field(..., description="Database merged into")
    incoming: Path = Field(..., description="Database merged in")
    output: Path | None = Field(
        default=None, description="Write target, defaults to base"
    )


def run_merge(
    cfg: PipelineConfig,
    paths: MergePaths,
    *,
    relink_all: bool = False,
    use_llm: bool = True,
) -> ArgumentDatabase:
    """Load two databases, merge them, and save the result over the base."""
    base = (
        ArgumentDatabase.load(paths.base) if paths.base.exists() else ArgumentDatabase()
    )
    incoming = ArgumentDatabase.load(paths.incoming)
    logger.info(
        f"Merging {len(incoming)} record(s) from {paths.incoming.name} "
        f"into {len(base)} record(s) from {paths.base.name}"
    )

    judge = None
    if use_llm:
        _, _, judge = make_agents(cfg.extractor_model or "", cfg.judge_model or "")

    merged, report = merge_databases(
        base, incoming, judge, cfg.top_k_links, relink_all=relink_all
    )
    logger.info(f"Merged database: {report.render()}")

    if not cfg.dry_run:
        target = paths.output or paths.base
        merged.save(target)
        logger.info(f"Saved merged database to {target}")
    return merged


def main() -> None:
    """Merge one database into another, rebuilding links across both."""
    parser = argparse.ArgumentParser(
        description="Merge an ArgumentDatabase into an existing pool."
    )
    parser.add_argument(
        "--into",
        type=Path,
        default=None,
        help="Database to merge into (default: the merged gold pool).",
    )
    parser.add_argument("--new", type=Path, required=True, help="Database to merge in.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Write here instead of --into."
    )
    parser.add_argument(
        "--relink-all",
        action="store_true",
        help="Re-judge every record, not only the incoming ones.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Only rebuild dataset-provided links, skip the judge.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--extractor-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.top_k is not None:
        cfg.top_k_links = args.top_k
    if args.extractor_model is not None:
        cfg.extractor_model = args.extractor_model
    if args.judge_model is not None:
        cfg.judge_model = args.judge_model
    cfg.dry_run = args.dry_run
    cfg = cfg.validated()

    paths = MergePaths(
        base=args.into or cfg.merged_gold_dir(),
        incoming=args.new,
        output=args.output,
    )
    run_merge(cfg, paths, relink_all=args.relink_all, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
