"""
Second sub-pipeline: gold ArgumentRecords into synthetic carrier documents.

Gold arguments are hidden inside realistic documents built from real text
harvested into ``contr-argument-dataset/argument_sources/native/``. Filler is
never invented from scratch: a native fragment is always the starting point,
which is what keeps the carriers from reading as pure LLM output.

Two independent, seed-driven layout axes are sampled per document:

- conclusion position: ``first`` / ``middle`` / ``last``
- premise dispersion: ``together`` / ``scattered`` (forced to ``together`` for
  conclusion-only records, which have no premises to scatter)

When the anchor argument has a counter-argument in the gold database and that
counter is still unfitted, both may land in the same document, the way a real
article covers both sides of a dispute.

The injected argument and premise texts are kept verbatim through the optional
LLM styling pass, and any styling that loses them is discarded: the carrier is
ground truth, so its offsets must stay exact. Every injected record has its
bounds recomputed against the assembled document and is marked fitted by
gaining a ``CarrierPlacement``.

Run with::

    uv run python -m src.argument_dataset.pipelines.argrec_to_carrier \
        --documents 20 --seed 7
"""

import argparse
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from src.argument_dataset.arg_db import ArgumentDatabase, ArgumentRecord, PremiseSource
from src.argument_dataset.backend import make_document_styler, run
from src.argument_dataset.config import PipelineConfig
from src.argument_dataset.output_models import StyledDocument
from src.configuration import config as global_config
from src.utils.file_loader import SourceFileLoader

MANIFEST_FILENAME = "manifest.jsonl"
MIN_FILLER_PARAGRAPHS = 4
MAX_FILLER_PARAGRAPHS = 9
MIN_MATCH_SCORE = 75.0
NATIVE_TEXT_CACHE_DIRNAME = "native_text"

ConclusionPosition = Literal["first", "middle", "last"]
PremiseDispersion = Literal["together", "scattered"]

CONCLUSION_POSITIONS: tuple[ConclusionPosition, ...] = ("first", "middle", "last")
PREMISE_DISPERSIONS: tuple[PremiseDispersion, ...] = ("together", "scattered")


# --- Manifest models ---


class DocumentLayout(BaseModel):
    """The two independent layout axes of one carrier document."""

    conclusion_position: ConclusionPosition
    premise_dispersion: PremiseDispersion


class GenerationSettings(BaseModel):
    """Seed-driven knobs of one synthetic batch."""

    document_count: int = Field(default=10, ge=1)
    seed: int = Field(default=0)
    counter_pair_probability: float = Field(default=0.3, ge=0.0, le=1.0)


class ManifestEntry(BaseModel):
    """One line of a synthetic batch manifest."""

    document_id: str
    source_kind: Literal["synthetic"] = "synthetic"
    argument_ids: list[str]
    argument_texts: dict[str, str] = Field(
        default_factory=dict,
        description="Injected gold id -> argument text, for extracted-vs-gold matching",
    )
    layout: DocumentLayout
    contains_counter_pair: bool
    filler_source_ids: list[str] = Field(default_factory=list)


# --- Filler ---


class FillerPool:
    """
    Paragraphs of real native documents, kept grouped by their donor.

    One carrier document draws its filler from ONE donor, as a contiguous run
    of paragraphs:

        transcript_012.pdf                       carrier
        | ...        |                           | para 41 |
        | para 41    |  -- contiguous window ->  | para 42 |  + injected argument
        | para 42    |                           | para 43 |
        | para 43    |                           | para 44 |
        | ...        |

    Drawing uniformly across every document instead would stitch each carrier
    from eight or nine unrelated sources, and the injected gold argument would
    be the only coherent thing on the page.
    """

    MIN_PARAGRAPH_CHARS = 80
    WINDOW_CANDIDATES = 32
    MIN_TOPIC_MATCHES = 2

    KEYWORD_RE = re.compile(r"[a-z]{4,}")
    KEYWORD_STOPWORDS = frozenset(
        [
            "about",
            "above",
            "after",
            "again",
            "against",
            "because",
            "been",
            "before",
            "being",
            "below",
            "between",
            "both",
            "cannot",
            "could",
            "does",
            "doing",
            "down",
            "during",
            "each",
            "from",
            "further",
            "have",
            "having",
            "here",
            "into",
            "itself",
            "more",
            "most",
            "only",
            "other",
            "over",
            "same",
            "should",
            "some",
            "such",
            "than",
            "that",
            "their",
            "them",
            "then",
            "there",
            "these",
            "they",
            "this",
            "those",
            "through",
            "under",
            "until",
            "very",
            "were",
            "what",
            "when",
            "where",
            "which",
            "while",
            "with",
            "would",
            "your",
        ]
    )

    @classmethod
    def keywords(cls, text: str) -> set[str]:
        """Reduce a text to the content words a cheap topical match can use."""
        return {
            word
            for word in cls.KEYWORD_RE.findall(text.lower())
            if word not in cls.KEYWORD_STOPWORDS
        }

    @classmethod
    def is_usable_filler(cls, paragraph: str) -> bool:
        """
        Decide whether a paragraph is real prose rather than page furniture.

        Harvested documents carry navigation labels, language lists and menu
        entries. Dropped here they cannot end up surrounding a gold argument
        and making the carrier read as scraped chrome instead of an article.
        """
        stripped = paragraph.strip()
        return len(stripped) >= cls.MIN_PARAGRAPH_CHARS and " " in stripped

    def __init__(self, native_dir: Path, cache_dir: Path | None = None) -> None:
        """Load every native document, split it into paragraphs and index it."""
        if not native_dir.is_dir():
            raise FileNotFoundError(
                f"Native sources directory not found at: {native_dir}. "
                f"Harvest real documents there before generating carriers."
            )
        if cache_dir is None:
            cache_dir = global_config.cache_directory / NATIVE_TEXT_CACHE_DIRNAME

        self.paragraphs_by_document: dict[str, list[str]] = {}
        self.keywords_by_document: dict[str, set[str]] = {}

        loader = SourceFileLoader(
            dir_path=native_dir, recursive=True, cache_dir=cache_dir
        )
        for document in loader.load():
            paragraphs = [
                block
                for block in (line.strip() for line in document.content.split("\n\n"))
                if self.is_usable_filler(block)
            ]
            if paragraphs:
                document_id = document.path.stem
                self.paragraphs_by_document[document_id] = paragraphs
                self.keywords_by_document[document_id] = self.keywords(
                    " ".join(paragraphs)
                )

        if not self.paragraphs_by_document:
            raise FileNotFoundError(
                f"No usable native documents found under {native_dir}. "
                f"Carrier documents cannot be built without real filler text. "
                f"Scanned PDFs extract no text - check that yours carry a text layer."
            )

        self.document_ids: list[str] = sorted(self.paragraphs_by_document)
        logger.info(
            f"Filler pool: {len(self.document_ids)} document(s), "
            f"{sum(len(p) for p in self.paragraphs_by_document.values())} paragraph(s)"
        )

    def pick_document(self, rng: random.Random, topic: str | None = None) -> str:
        """
        Choose the donor document, preferring one that shares the topic's words.

        A bag-of-words overlap is enough here: the point is to avoid dropping a
        nuclear-energy argument into a fisheries transcript, not to retrieve
        anything precisely. It takes MIN_TOPIC_MATCHES shared content words to
        beat a uniform draw - one word in common is coincidence, and letting it
        decide would be worse than admitting the corpus covers nothing relevant.
        """
        if topic is None:
            return rng.choice(self.document_ids)

        wanted = self.keywords(topic)
        if not wanted:
            return rng.choice(self.document_ids)

        scored = [
            (len(wanted & self.keywords_by_document[document_id]), document_id)
            for document_id in self.document_ids
        ]
        best = max(score for score, _ in scored)
        if best < self.MIN_TOPIC_MATCHES:
            return rng.choice(self.document_ids)
        return rng.choice(
            [document_id for score, document_id in scored if score == best]
        )

    def _window_start(
        self, rng: random.Random, paragraphs: list[str], count: int, topic: str | None
    ) -> int:
        """
        Choose where the contiguous run starts, favouring on-topic paragraphs.

        Only a bounded number of candidate windows is examined, so the cost per
        carrier does not grow with the length of the donor: a 400-page
        transcript is scored as cheaply as a two-page article. The aim is an
        on-topic region, not the single best one in the document.
        """
        last_start = max(0, len(paragraphs) - count)
        wanted = self.keywords(topic) if topic is not None else set()
        if not wanted or not last_start:
            return rng.randint(0, last_start)

        candidates = {rng.randint(0, last_start) for _ in range(self.WINDOW_CANDIDATES)}
        best_start, best_hits = last_start, -1
        for start in candidates:
            window = " ".join(paragraphs[start : start + count]).lower()
            hits = sum(1 for word in wanted if word in window)
            if hits > best_hits:
                best_start, best_hits = start, hits
        return best_start

    def sample(
        self, rng: random.Random, count: int, topic: str | None = None
    ) -> tuple[list[str], list[str]]:
        """Draw ``count`` contiguous filler paragraphs from a single donor."""
        document_id = self.pick_document(rng, topic)
        paragraphs = self.paragraphs_by_document[document_id]

        start = self._window_start(rng, paragraphs, count, topic)
        window = paragraphs[start : start + count]
        while len(window) < count:  # short document: wrap around rather than pad
            window += paragraphs[: count - len(window)]

        return window, [document_id]


# --- Layout ---


def sample_layout(rng: random.Random, records: list[ArgumentRecord]) -> DocumentLayout:
    """
    Sample the two layout axes independently.

    Dispersion collapses to ``together`` when no injected record has premises:
    a conclusion-only argument has nothing to scatter.
    """
    conclusion_position: ConclusionPosition = rng.choice(CONCLUSION_POSITIONS)
    scatterable = any(
        record.provenance.premises_source != PremiseSource.NONE and record.premises
        for record in records
    )
    dispersion: PremiseDispersion = (
        rng.choice(PREMISE_DISPERSIONS) if scatterable else "together"
    )
    return DocumentLayout(
        conclusion_position=conclusion_position, premise_dispersion=dispersion
    )


def _conclusion_slot(position: ConclusionPosition, paragraph_count: int) -> int:
    """Map a conclusion position onto a paragraph index."""
    if position == "first":
        return 0
    if position == "last":
        return paragraph_count - 1
    return paragraph_count // 2


def _inject(paragraphs: list[str], index: int, sentence: str) -> None:
    """Append a sentence to the end of one paragraph."""
    slot = max(0, min(index, len(paragraphs) - 1))
    paragraphs[slot] = f"{paragraphs[slot]} {sentence}".strip()


def build_draft(
    records: list[ArgumentRecord],
    filler_paragraphs: list[str],
    layout: DocumentLayout,
) -> str:
    """
    Weave the injected arguments into real filler paragraphs.

    Sentences are appended into existing filler paragraphs rather than added as
    standalone blocks, so an argument reads as part of the surrounding prose
    instead of standing out as an inserted quote.
    """
    paragraphs: list[str] = list(filler_paragraphs)
    count = len(paragraphs)

    for offset, record in enumerate(records):
        conclusion_slot = (
            _conclusion_slot(layout.conclusion_position, count) + offset * 2
        ) % count
        premise_texts = [premise.text for premise in record.premises]

        if layout.premise_dispersion == "scattered" and premise_texts:
            available = [index for index in range(count) if index != conclusion_slot]
            for position, premise in enumerate(premise_texts):
                if not available:
                    break
                _inject(paragraphs, available[position % len(available)], premise)
        else:
            premise_slot = max(0, conclusion_slot - 1)
            for premise in premise_texts:
                _inject(paragraphs, premise_slot, premise)

        _inject(paragraphs, conclusion_slot, record.argument)

    return "\n\n".join(paragraphs)


# --- Styling ---


def protected_texts(records: list[ArgumentRecord]) -> list[str]:
    """List every text whose exact wording the carrier must preserve."""
    texts: list[str] = []
    for record in records:
        texts.append(record.argument)
        texts.extend(premise.text for premise in record.premises)
    return texts


def all_texts_located(document: str, texts: list[str]) -> bool:
    """Check that every protected text is still findable in the document."""
    return all(
        ArgumentRecord.find_text_bounds(text, document, MIN_MATCH_SCORE).is_located
        for text in texts
    )


def style_document(
    styler: Agent[None, StyledDocument], draft: str, texts: list[str]
) -> str:
    """
    Restyle the filler around the injected arguments, keeping them verbatim.

    Styling that drops or rewrites a protected text is discarded: an
    unrecoverable argument would make the carrier useless as ground truth.
    """
    quoted = "\n".join(f"- {text}" for text in texts)
    prompt = (
        "Sentences that must appear verbatim in your output:\n"
        f"{quoted}\n\n"
        "Draft document:\n"
        f"{draft}"
    )
    try:
        styled = run(styler, prompt).text.strip()
    except (UnexpectedModelBehavior, ValueError, AttributeError) as exc:
        logger.warning(f"Styling failed, keeping the unstyled draft: {exc}")
        return draft

    if not styled or not all_texts_located(styled, texts):
        logger.warning("Styling lost an injected text, keeping the unstyled draft.")
        return draft
    return styled


# --- Grounding ---


def reposition_record(
    record: ArgumentRecord, document: str, document_id: str, batch_id: str | None = None
) -> bool:
    """
    Give a record its CarrierPlacement, which also marks it fitted.

    All bound-finding lives in ArgumentRecord, so one definition of "where is
    this text" governs the whole dataset.
    """
    placed = record.place_in_carrier(document, document_id, batch_id)
    if not placed:
        logger.warning(
            f"Could not locate argument {record.id} in document {document_id}."
        )
    return placed


# --- Generation ---


def counter_argument_of(
    db: ArgumentDatabase, record: ArgumentRecord
) -> ArgumentRecord | None:
    """Find an unfitted counter-argument of a record, if the gold links have one."""
    by_id = {argument.id: argument for argument in db.arguments}
    for link in db.argument_links:
        partner_id: str | None = None
        if link.source_argument_id == record.id:
            partner_id = link.target_argument_id
        elif link.target_argument_id == record.id:
            partner_id = link.source_argument_id

        if partner_id is None:
            continue

        partner = by_id.get(partner_id)
        if partner is not None and not partner.is_placed:
            return partner
    return None


def pending_pool(db: ArgumentDatabase) -> list[ArgumentRecord]:
    """List the gold records not yet injected into any carrier document."""
    return [
        record
        for record in db.arguments
        if not record.is_placed and record.argument.strip()
    ]


def generate_documents(
    db: ArgumentDatabase,
    filler: FillerPool,
    batch_dir: Path | None,
    settings: GenerationSettings,
    styler: Agent[None, StyledDocument] | None = None,
) -> list[ManifestEntry]:
    """
    Generate a batch of carrier documents and return their manifest entries.

    A ``batch_dir`` of None keeps the run in memory (dry run): records are
    still repositioned and marked fitted, but nothing is written to disk.
    """
    rng = random.Random(settings.seed)
    entries: list[ManifestEntry] = []

    if batch_dir is not None:
        batch_dir.mkdir(parents=True, exist_ok=True)

    for index in range(settings.document_count):
        pool = pending_pool(db)
        if not pool:
            logger.info(f"Pending pool exhausted after {index} document(s).")
            break

        anchor = rng.choice(pool)
        records = [anchor]
        counter = counter_argument_of(db, anchor)
        contains_counter_pair = counter is not None and (
            rng.random() < settings.counter_pair_probability
        )
        if contains_counter_pair and counter is not None:
            records.append(counter)

        layout = sample_layout(rng, records)
        paragraph_count = rng.randint(MIN_FILLER_PARAGRAPHS, MAX_FILLER_PARAGRAPHS)
        filler_paragraphs, donors = filler.sample(
            rng, paragraph_count, topic=anchor.argument
        )

        document_id = f"synthetic_{settings.seed}_{index:04d}"
        draft = build_draft(records, filler_paragraphs, layout)
        document = (
            style_document(styler, draft, protected_texts(records))
            if styler is not None
            else draft
        )

        grounded = [
            record
            for record in records
            if reposition_record(record, document, document_id)
        ]
        if not grounded:
            logger.warning(f"Skipping {document_id}: no argument could be grounded.")
            continue

        if batch_dir is not None:
            (batch_dir / f"{document_id}.txt").write_text(document, encoding="utf-8")

        entries.append(
            ManifestEntry(
                document_id=document_id,
                argument_ids=[record.id for record in grounded],
                argument_texts={record.id: record.argument for record in grounded},
                layout=layout,
                contains_counter_pair=len(grounded) > 1,
                filler_source_ids=donors,
            )
        )

    return entries


def write_manifest(batch_dir: Path, entries: list[ManifestEntry]) -> Path:
    """Write one JSON line per generated document."""
    path = batch_dir / MANIFEST_FILENAME
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.model_dump_json() + "\n")
    return path


def read_manifest(batch_dir: Path) -> list[ManifestEntry]:
    """Read the manifest of a synthetic batch."""
    path = batch_dir / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return [ManifestEntry(**json.loads(line)) for line in handle if line.strip()]


# --- Driver ---


def run_batch(
    cfg: PipelineConfig,
    batch_id: str,
    settings: GenerationSettings,
    source_name: str | None = None,
    *,
    use_llm: bool = True,
) -> list[ManifestEntry]:
    """Generate one synthetic batch and persist the updated gold database."""
    gold_dir = (
        cfg.gold_source_dir(source_name)
        if source_name is not None
        else cfg.merged_gold_dir()
    )
    db = ArgumentDatabase.load(gold_dir)
    filler = FillerPool(cfg.native_sources_dir)
    styler = make_document_styler(cfg.extractor_model or "") if use_llm else None

    batch_dir = None if cfg.dry_run else cfg.synthetic_batch_dir(batch_id)
    entries = generate_documents(db, filler, batch_dir, settings, styler)

    injected = sum(len(entry.argument_ids) for entry in entries)
    logger.info(
        f"Batch {batch_id}: {len(entries)} document(s), {injected} argument(s) "
        f"injected, {len(pending_pool(db))} still pending"
    )

    if batch_dir is not None:
        write_manifest(batch_dir, entries)
        db.save(gold_dir)
        logger.info(f"Updated gold database written back to {gold_dir}")
    return entries


def _apply_args(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Override config with CLI args."""
    if args.gold_dir is not None:
        cfg.gold_arguments_dir = args.gold_dir
    if args.native_dir is not None:
        cfg.native_sources_dir = args.native_dir
    if args.synthetic_dir is not None:
        cfg.synthetic_sources_dir = args.synthetic_dir
    if args.extractor_model is not None:
        cfg.extractor_model = args.extractor_model
    cfg.dry_run = args.dry_run
    return cfg.validated()


def main() -> None:
    """Generate a batch of synthetic carrier documents."""
    parser = argparse.ArgumentParser(
        description="Inject gold arguments into synthetic carrier documents."
    )
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--documents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source",
        default=None,
        help="Use one per-source gold database instead of the merged pool.",
    )
    parser.add_argument("--counter-pair-probability", type=float, default=0.3)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM styling pass (deterministic assembly only).",
    )
    parser.add_argument("--gold-dir", type=Path, default=None)
    parser.add_argument("--native-dir", type=Path, default=None)
    parser.add_argument("--synthetic-dir", type=Path, default=None)
    parser.add_argument("--extractor-model", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _apply_args(PipelineConfig(), args)
    batch_id = args.batch_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    settings = GenerationSettings(
        document_count=args.documents,
        seed=args.seed,
        counter_pair_probability=args.counter_pair_probability,
    )
    run_batch(cfg, batch_id, settings, args.source, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
