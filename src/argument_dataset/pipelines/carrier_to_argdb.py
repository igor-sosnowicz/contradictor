"""
Two-model LLM pipeline for building the contr-argument dataset.

Uses argument_sources from `contr-argument-dataset/argument_sources`.

Argument sources consist of:
- Synthetic sources
- Native sources

Pipeline overview (see ``docs/dataset_building_pipeline.md``):

- **Phase 1 (per-document extraction):**
    1. Extractor model reads and processes each document from
       ``data/counterargument-dataset/argument_carriers/``
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

- **Source-corpus mode (``--native`` / ``--synthetic``):**
    Runs the very same extractor + judge flow over the carrier documents in
    ``contr-argument-dataset/argument_sources/``, which is what makes the
    synthetic corpus a genuine test of the extraction pipeline rather than a
    replay of the gold database.

    ``--native`` reads real harvested documents; there is no known ground
    truth for them, so ``provenance.original_id`` stays None and the output is
    for qualitative review only. ``--synthetic`` reads one generated batch and,
    using that batch's manifest, matches every extracted argument back to the
    gold argument that was injected, storing its id in
    ``provenance.original_id`` so gold and extracted can be compared.

    The result is saved under ``arg_db_objects/`` and is NEVER merged into
    ``arguments/gold/``: gold holds pure arguments from datasets, arg_db_objects
    holds what the extractor found in carrier documents. Comparing the two is
    the whole point.

Setup:
    - Run LM Studio server running locally
    - Configure model names in ``config.toml`` (``extractor_model``/``judge_model``)
    - Run 'uv run python -m src.argument_dataset.pipelines.carriers_to_argdb
      --phase all'
    - Run 'uv run python -m src.argument_dataset.pipelines.carriers_to_argdb
      --synthetic BATCH'
"""

import argparse
from pathlib import Path

import httpx
from loguru import logger
from pydantic import BaseModel
from rapidfuzz import fuzz

from src.argument_dataset.arg_db import (
    ArgumentDatabase,
    ArgumentProvenance,
    ArgumentRecord,
    PremiseSource,
)
from src.argument_dataset.backend import list_loaded_models, make_agents, run
from src.argument_dataset.config import PipelineConfig
from src.argument_dataset.output_models import ExtractedArgument
from src.argument_dataset.pipelines.argrec_to_carrier import (
    NATIVE_TEXT_CACHE_DIRNAME,
    read_manifest,
)
from src.argument_dataset.progress import track
from src.configuration import config as global_config
from src.utils.file_loader import SourceFileLoader

EXTRACTED_DB_DIRNAME = "extracted_argdb"
NATIVE_SOURCE_DATASET = "native"
GOLD_MATCH_MIN_SCORE = 75.0


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


def run_phase1(cfg: PipelineConfig, *, no_repair: bool = False) -> ArgumentDatabase:
    """Extract arguments from all sources, judge them, save the database."""
    extractor, judge, _ = make_agents(cfg.extractor_model or "", cfg.judge_model or "")
    db = ArgumentDatabase()
    documents = load_documents(cfg.sources_dir)

    if cfg.limit is not None:
        documents = documents[: cfg.limit]

    for doc in track(documents, "Phase 1 | extracting"):
        for chunk in track(
            chunk_text(doc.text, cfg.chunk_size, cfg.chunk_overlap),
            f"  {doc.document_id} | chunks",
        ):
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
                db += ArgumentRecord.from_extracted(
                    final,
                    provenance=ArgumentProvenance(source_dataset=doc.document_id),
                    document_text=doc.text,
                    document_id=doc.document_id,
                )

    if not cfg.dry_run:
        db.save(cfg.database_dir)
    return db


def run_phase2(cfg: PipelineConfig) -> ArgumentDatabase:
    """Match candidates via the database, judge links, save them to the database."""
    _, _, judge = make_agents(cfg.extractor_model or "", cfg.judge_model or "")
    db = ArgumentDatabase.load(cfg.database_dir)

    for record in track(db.arguments, "Phase 2 | linking"):
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
        db.save(cfg.database_dir)
    return db


# --- Source corpora (native / synthetic carriers) ---


def load_corpus_documents(
    source_dir: Path, cache_dir: Path | None = None
) -> list[SourceDocument]:
    """
    Load every carrier document of a corpus directory.

    Uses the shared file loader, so harvested native documents may be html or
    pdf as they came off the web, not only plain text - and it converts each
    of them once, reusing the same extracted-text cache the carrier pipeline
    fills.
    """
    if cache_dir is None:
        cache_dir = global_config.cache_directory / NATIVE_TEXT_CACHE_DIRNAME
    return [
        SourceDocument(document_id=file.path.stem, text=file.content)
        for file in SourceFileLoader(
            dir_path=source_dir, recursive=False, cache_dir=cache_dir
        ).load()
    ]


def load_gold_texts(batch_dir: Path) -> dict[str, dict[str, str]]:
    """
    Read a synthetic batch manifest into document -> {gold id: argument text}.

    Returns an empty mapping when the batch has no manifest: extraction still
    runs, it just cannot be traced back to the injected gold arguments.
    """
    try:
        entries = read_manifest(batch_dir)
    except FileNotFoundError:
        logger.warning(
            f"No manifest in {batch_dir}; extracted arguments will not carry "
            f"gold ids, so Step 6 cannot score this batch."
        )
        return {}

    return {entry.document_id: dict(entry.argument_texts) for entry in entries}


def match_gold_argument(
    extracted: ExtractedArgument, gold_texts: dict[str, str]
) -> str | None:
    """Fuzzy-match an extracted argument back to the gold argument injected."""
    if not gold_texts:
        return None

    best_id: str | None = None
    best_score = 0.0
    for gold_id, text in gold_texts.items():
        score = float(fuzz.token_set_ratio(extracted.argument, text))
        if score > best_score:
            best_id, best_score = gold_id, score

    if best_score < GOLD_MATCH_MIN_SCORE:
        return None
    return best_id


def extract_from_documents(
    cfg: PipelineConfig,
    documents: list[SourceDocument],
    source_dataset: str,
    gold_texts_by_document: dict[str, dict[str, str]] | None = None,
    *,
    no_repair: bool = False,
) -> ArgumentDatabase:
    """
    Run the production extractor + judge over a corpus of carrier documents.

    This is the same flow as Phase 1, so what it finds is what the production
    system would find; only the provenance written onto each record differs.
    """
    extractor, judge, _ = make_agents(cfg.extractor_model or "", cfg.judge_model or "")
    gold_texts_by_document = gold_texts_by_document or {}
    db = ArgumentDatabase()

    for doc in track(documents, f"Extracting from {source_dataset}"):
        gold_texts = gold_texts_by_document.get(doc.document_id, {})
        chunks = chunk_text(doc.text, cfg.chunk_size, cfg.chunk_overlap)
        for chunk in track(chunks, f"  {doc.document_id} | chunks"):
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
                provenance = ArgumentProvenance(
                    source_dataset=source_dataset,
                    original_id=match_gold_argument(final, gold_texts),
                    premises_source=(
                        PremiseSource.NATIVE if final.premises else PremiseSource.NONE
                    ),
                )
                db += ArgumentRecord.from_extracted(
                    final,
                    provenance=provenance,
                    document_text=doc.text,
                    document_id=doc.document_id,
                )

    return db


def run_native_sources(cfg: PipelineConfig, source_dir: Path | None = None) -> Path:
    """Extract arguments from the harvested native corpus."""
    corpus_dir = source_dir or cfg.native_sources_dir
    documents = load_corpus_documents(corpus_dir)
    if cfg.limit is not None:
        documents = documents[: cfg.limit]

    db = extract_from_documents(cfg, documents, NATIVE_SOURCE_DATASET)
    target = cfg.extracted_native_db_dir
    logger.info(
        f"Native corpus: {len(documents)} document(s) -> "
        f"{len(db.arguments)} extracted argument(s)"
    )

    if not cfg.dry_run:
        db.save(target)
        logger.info(f"Saved extracted native database to {target}")
    return target


def run_synthetic_sources(
    cfg: PipelineConfig, batch_id: str, source_dir: Path | None = None
) -> Path:
    """Extract arguments from one synthetic batch and trace them back to gold."""
    corpus_dir = source_dir or cfg.synthetic_batch_dir(batch_id)
    documents = load_corpus_documents(corpus_dir)
    if cfg.limit is not None:
        documents = documents[: cfg.limit]

    gold_texts = load_gold_texts(corpus_dir)
    db = extract_from_documents(cfg, documents, f"synthetic:{batch_id}", gold_texts)
    traced = sum(1 for record in db.arguments if record.provenance.original_id)
    target = cfg.extracted_synthetic_db_dir(batch_id)
    logger.info(
        f"Synthetic batch {batch_id}: {len(documents)} document(s) -> "
        f"{len(db.arguments)} extracted argument(s), {traced} matched to a gold id"
    )

    if not cfg.dry_run:
        db.save(target)
        logger.info(f"Saved extracted synthetic database to {target}")
    return target


# --- CLI ---


def _apply_args(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Override config with CLI args."""
    if args.sources is not None:
        cfg.sources_dir = args.sources
    if args.database_dir is not None:
        cfg.database_dir = args.database_dir
    if args.native_dir is not None:
        cfg.native_sources_dir = args.native_dir
    if args.synthetic_dir is not None:
        cfg.synthetic_sources_dir = args.synthetic_dir
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
    """Warn when configured models are not loaded."""
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
    corpus = parser.add_mutually_exclusive_group()
    corpus.add_argument(
        "--native",
        action="store_true",
        help="Extract from the harvested native corpus (no known ground truth).",
    )
    corpus.add_argument(
        "--synthetic",
        metavar="BATCH_ID",
        default=None,
        help="Extract from one synthetic batch, tracing arguments back to gold.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Corpus directory override, for running on a subset.",
    )
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument("--sources", type=Path, default=None)
    parser.add_argument("--database-dir", type=Path, default=None)
    parser.add_argument("--native-dir", type=Path, default=None)
    parser.add_argument("--synthetic-dir", type=Path, default=None)
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

    if args.native:
        run_native_sources(cfg, args.source_dir)
        return
    if args.synthetic is not None:
        run_synthetic_sources(cfg, args.synthetic, args.source_dir)
        return

    if args.phase in ("1", "all"):
        db = run_phase1(cfg, no_repair=args.no_repair)
        logger.info(f"Phase 1 done: {len(db.arguments)} arguments.")
    if args.phase in ("2", "all"):
        db = run_phase2(cfg)
        logger.info(f"Phase 2 done: {len(db.argument_links)} links.")


if __name__ == "__main__":
    main()
