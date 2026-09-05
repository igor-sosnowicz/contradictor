"""
Two-model LLM pipeline for building the contr-argument dataset.

Pipeline overview (see ``docs/dataset_building_pipeline.md``):

- **Phase 1 (per-document extraction):**
    1. Extractor model reads and processes each document from
       ``data/counterargument-dataset/sources/``
    2. Structured :class:`ExtractedArgument` items are returned.
    3. Judge model verifies each item. Can take two type of actions:
        - Accept / Reject item (--no-repair flag)
        - Accept / Reject & Repair
    Accepted items get:
    - Index offsets in source documents
    - Embeddings (Dense & Sparse)

    Judge:
    Gets :class:`ExtractedArgument` parsed as JSON as input and outputs
    :class:`ExtractionVerdict`.

    Phase 1 results in:
    - 'data/counterargument-dataset/datasett.json' (ArgumentDatabase object)
    - 'data/counterargument-dataset/embeddings' (pickled embedding map).

- **Phase 2 (global linking):**
    1. Reads the argumentuments database from Phase 1
    2. Each argument is paired with top-K candidates matched by method
       `fetch_similar_arguments` from class:`ArgumentDatabase`
    3. Connection Judge verifies whether one argument refutes candidate arguments,
       repairing the link type/direction when needed. Verified links are saved
       to the `argument_links` field of class:`ArgumentDatabase`.

    Phase 2 results in:
    - `argument_links` field updated in the database

    Judge:
    Gets simplified class:`ArgumentRecord` as base and top-K candidates of
    class:`ArgumentRecord` matched by method `fetch_similar_arguments` from
    class:`ArgumentDatabase`. Verifies whether one argument refutes the candidates
    and outputs :class:`LinkVerdict`.

Setup:
    - Run LM Studio server running locally
    - Configure model names in ``config.toml`` (``extractor_model``/``judge_model``)
    - Run 'uv run python -m src.argument_dataset.pipeline --phase all'
"""

import argparse
from pathlib import Path

import httpx
from loguru import logger
from pydantic import BaseModel

from src.argument_dataset.arg_db import ArgumentDatabase, ArgumentRecord
from src.argument_dataset.backend import list_loaded_models, make_agents, run
from src.argument_dataset.config import PipelineConfig


class SourceDocument(BaseModel):
    """Raw source document loaded from the sources directory."""

    document_id: str
    text: str


class TextChunk(BaseModel):
    """Chunk of a document with its global character offsets."""

    text: str
    start: int
    end: int


def load_documents(sources_dir: Path) -> list[SourceDocument]:
    """Load [*.txt | *.md] files from the sources directory."""
    documents: list[SourceDocument] = []
    for pattern in ("*.txt", "*.md"):
        for path in sorted(sources_dir.glob(pattern)):
            text = path.read_text(encoding="utf-8", errors="replace")
            text = text.replace("\r\n", "\n").strip()
            if not text:
                logger.warning(f"Skipping empty source file: {path.name}")
                continue
            documents.append(SourceDocument(document_id=path.name, text=text))
    return documents


def chunk_text(text: str, size: int, overlap: int) -> list[TextChunk]:
    """Split text into overlapping chunks, preferring whitespace boundaries."""
    chunks: list[TextChunk] = []
    start = 0
    end_limit = len(text)
    while start < end_limit:
        end = min(start + size, end_limit)
        if end < end_limit:
            boundary = text.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary
        chunks.append(TextChunk(text=text[start:end], start=start, end=end))
        if end >= end_limit:
            break
        start = max(end - overlap, start + 1)
    return chunks


def run_phase1(cfg: PipelineConfig, no_repair: bool = False) -> ArgumentDatabase:  # noqa: FBT001, FBT002
    """Extract arguments from all sources, judge them, save the database."""
    extractor, judge, _ = make_agents(cfg.extractor_model or "", cfg.judge_model or "")
    db = ArgumentDatabase()
    documents = load_documents(cfg.sources_dir)

    if cfg.limit is not None:
        documents = documents[: cfg.limit]

    for doc in documents:
        for chunk in chunk_text(doc.text, cfg.chunk_size, cfg.chunk_overlap):
            batch = run(extractor, f"Document chunk:\n\n{chunk.text}")
            for item in batch.arguments:
                verdict = run(
                    judge,
                    f"Source chunk:\n\n{chunk.text}\n\nExtracted argument (JSON):\n"
                    f"{item.model_dump_json(indent=2, by_alias=True)}",
                )
                if verdict.verdict == "reject":
                    continue
                final = item if no_repair else (verdict.repaired or item)
                db += ArgumentRecord.from_extracted(final, doc.text, doc.document_id)

    if not cfg.dry_run:
        db.save()
    return db


def run_phase2(cfg: PipelineConfig) -> ArgumentDatabase:
    """Match candidates via the database, judge links, save them to the database."""
    _, _, judge = make_agents(cfg.extractor_model or "", cfg.judge_model or "")
    db = ArgumentDatabase.load()

    for record in db.arguments:
        matches = db.fetch_similar_arguments(record, top_k=cfg.top_k_links)
        for target, score in matches:
            verdict = run(
                judge,
                f"Argument A ({record.domain.value}):\n{record.argument}\n\n"
                f"A's target-claim hypothesis: "
                f"{'; '.join(record.target_claims_hypothesis)}\n\n"
                f"Argument B ({target.domain.value}):\n{target.argument}\n\n"
                f"B's target-claim hypothesis: "
                f"{'; '.join(target.target_claims_hypothesis)}\n\n"
                "Does A refute B or B's hypothesis?",
            )
            if not verdict.refutes or verdict.link_type is None:
                continue
            if (
                verdict.repaired_target_hypothesis
                and verdict.repaired_target_hypothesis
                not in record.target_claims_hypothesis
            ):
                record.target_claims_hypothesis.append(
                    verdict.repaired_target_hypothesis
                )
            try:
                db.add_argument_link(record, target, verdict.link_type, score)
            except ValueError as exc:
                logger.debug(f"Skipping link: {exc}")

    if not cfg.dry_run:
        db.save()
    return db


def _apply_args(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Override config with CLI args."""
    if args.sources is not None:
        cfg.sources_dir = args.sources
    if args.extractor_model is not None:
        cfg.extractor_model = args.extractor_model
    if args.judge_model is not None:
        cfg.judge_model = args.judge_model
    if args.top_k is not None:
        cfg.top_k_links = args.top_k
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        cfg.chunk_overlap = args.chunk_overlap
    cfg.limit = args.limit
    cfg.dry_run = args.dry_run
    return cfg.validated()


def _check_models(cfg: PipelineConfig) -> None:
    """Warn when configured models are not loaded in LM Studio."""
    try:
        loaded = list_loaded_models()
    except httpx.HTTPError as exc:
        logger.warning(f"Could not list LM Studio models: {exc}")
        return
    for role, name in (("extractor", cfg.extractor_model), ("judge", cfg.judge_model)):
        if name not in loaded:
            logger.warning(f"{role} model {name!r} not loaded in LM Studio.")


def main() -> None:
    """Parse args and run the requested pipeline phase(s)."""
    parser = argparse.ArgumentParser(description="Build the contr-argument dataset.")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument("--sources", type=Path, default=None)
    parser.add_argument("--extractor-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _apply_args(PipelineConfig(), args)
    _check_models(cfg)

    if args.phase in ("1", "all"):
        db = run_phase1(cfg, no_repair=args.no_repair)
        logger.info(f"Phase 1 done: {len(db.arguments)} arguments.")
    if args.phase in ("2", "all"):
        db = run_phase2(cfg)
        logger.info(f"Phase 2 done: {len(db.argument_links)} links.")


if __name__ == "__main__":
    main()
