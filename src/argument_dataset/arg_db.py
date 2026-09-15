"""
Argument database schema as Pydantic models.

An ArgumentRecord is a pure argument. Where it sits inside a carrier document
is a separate, optional concern:

    ArgumentRecord (gold, from a dataset)     ArgumentRecord (in a carrier)
    | id | argument | premises | frame |      | id | argument | premises | frame |
    | provenance | native_counter_ids  |      | provenance | native_counter_ids  |
    | carrier = None                   |      | carrier = CarrierPlacement(...)  |

Two databases are combined by `union()` (structural, deduplicated by
provenance) and cross-linked by `link_native_counters()` (ids the source
dataset itself provided). Links that no dataset knows about are discovered by
the LLM merge pipeline, see `pipelines/merge_arg_dbs.py`.
"""

import json
import pickle
import warnings
from collections.abc import Mapping
from enum import StrEnum, auto
from pathlib import Path
from typing import Self
from uuid import uuid4

import numpy as np
from fastembed import SparseEmbedding
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from rapidfuzz import fuzz

from src.argument_dataset.arg_link import ArgumentLinkType
from src.argument_dataset.output_models import ExtractedArgument, ExtractedPremise
from src.configuration import config as global_config
from src.data_models.data_models import InterpretativeFrame
from src.utils.fast_embedders import (
    SparseEmbedder,
    create_dense_embedder,
    create_sparse_embedder,
)

ARG_DB_JSON_FILENAME = "argdb.json"
ARG_DB_EMBEDDINGS_FILENAME = "argdb_embeddings.pkl"

MIN_MATCH_SCORE = 75.0
DENSE_MATRIX_DIMS = 2
NOT_FOUND = (-1, -1)


class EmbeddingIdentifier(BaseModel):
    """Identifier for an embedding, used for indexing and retrieval."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense_embedding: np.ndarray | None = Field(default=None)
    sparse_embedding: SparseEmbedding | None = Field(default=None)

    def is_empty(self) -> bool:
        """Check if both embeddings are empty."""
        return (self.dense_embedding is None or self.dense_embedding.size == 0) and (
            self.sparse_embedding is None or self.sparse_embedding.values.size == 0
        )

    @classmethod
    def create(cls, content: str | None = None) -> "EmbeddingIdentifier":
        """Build embeddings for the given content (empty when content is empty)."""
        if not content:
            return cls(
                dense_embedding=np.array([]),
                sparse_embedding=SparseEmbedding(
                    values=np.array([]), indices=np.array([])
                ),
            )

        dense_embedder = create_dense_embedder(
            global_config.provider,
            global_config.lm_studio_api_base_url,
            global_config.lm_studio_api_key,
        )

        return cls(
            dense_embedding=dense_embedder.embed(content),
            sparse_embedding=create_sparse_embedder().embed(content),
        )


# --- Provenance ---


class PremiseSource(StrEnum):
    """Where the premises of an argument record came from."""

    NATIVE = auto()
    SYNTHETIC_GENERATED = auto()
    SYNTHETIC_UNVALIDATED = auto()
    NONE = auto()


class ArgumentProvenance(BaseModel):
    """
    Where a record came from. Its identity across rebuilds and merges.
    """

    source_dataset: str = Field(
        ..., description="Name of the source dataset or carrier corpus"
    )
    original_id: str | None = Field(
        default=None, description="Id in the source dataframe, if any"
    )
    premises_source: PremiseSource = Field(
        default=PremiseSource.NATIVE, description="Origin of the record's premises"
    )
    premise_validation_score: float | None = Field(
        default=None,
        description=(
            "Entailment score of the generated premises. "
            "Only set when premises_source == SYNTHETIC_GENERATED."
        ),
    )

    @property
    def lookup_key(self) -> str | None:
        """Return "<source_dataset>:<original_id>", or None without a native id."""
        if self.original_id is None:
            return None
        return f"{self.source_dataset}:{self.original_id}"


# --- Carrier placement ---


class TextBounds(BaseModel):
    """Character bounds of a piece of text inside a carrier document."""

    start_idx: int = Field(..., description="Start index, -1 when not located")
    end_idx: int = Field(..., description="End index, -1 when not located")

    @property
    def is_located(self) -> bool:
        """Whether the text was actually found in the document."""
        return self.start_idx >= 0 and self.end_idx > self.start_idx


class CarrierPlacement(BaseModel):
    """
    Where one argument sits inside one carrier document.

    None on a gold record: a dataset argument is not in any document yet.

    | document_id | argument bounds | premise bounds (parallel to premises) |
    """

    document_id: str = Field(..., description="Carrier document this record sits in")
    argument: TextBounds = Field(..., description="Bounds of the conclusion text")
    premises: list[TextBounds] = Field(
        default_factory=list,
        description="Bounds per premise, index-aligned with record.premises",
    )
    batch_id: str | None = Field(
        default=None, description="Synthetic batch id, None for native carriers"
    )


# --- Links ---


class ArgumentLink(BaseModel):
    """Link between an argument and its counter-argument."""

    source_argument_id: str = Field(..., description="ID of the argument")
    target_argument_id: str = Field(..., description="ID of the counter-argument")
    link_type: ArgumentLinkType = Field(
        ..., description="Type of the counter-argument link"
    )
    confidence: float = Field(default=1.0, description="Confidence score of the link")

    def __eq__(self, other: object) -> bool:
        """Compare links ignoring direction (undirected edge)."""
        if not isinstance(other, ArgumentLink):
            return NotImplemented
        return all(
            [
                self.source_argument_id
                in [other.source_argument_id, other.target_argument_id],
                self.target_argument_id
                in [other.source_argument_id, other.target_argument_id],
                self.link_type == other.link_type,
            ]
        )

    def __hash__(self) -> int:
        """Hash the directed edge for set storage."""
        return hash((self.source_argument_id, self.target_argument_id, self.link_type))


# --- Records ---


class ArgumentRecord(ExtractedArgument):
    """One argument, optionally placed inside a carrier document."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the argument",
    )
    provenance: ArgumentProvenance = Field(
        ..., description="Where this record came from"
    )
    native_counter_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Source-local ids of counter-arguments, exactly as the dataset gave "
            "them. EMPTY when the source provides no counter mapping. Resolved "
            "into ArgumentLinks by ArgumentDatabase.link_native_counters()."
        ),
    )
    carrier: CarrierPlacement | None = Field(
        default=None,
        description="Placement in a carrier document, None while still a pure argument",
    )

    # Searchability (private attributes: excluded from JSON dumps by default)
    _argument_embedding: EmbeddingIdentifier = PrivateAttr(
        default_factory=EmbeddingIdentifier
    )
    _target_claims_embedding: EmbeddingIdentifier = PrivateAttr(
        default_factory=EmbeddingIdentifier
    )

    @property
    def is_placed(self) -> bool:
        """Whether this record has been injected into a carrier document."""
        return self.carrier is not None

    @property
    def identity(self) -> str:
        """Identity used for deduplication: provenance first, UUID as fallback."""
        return self.provenance.lookup_key or f"uuid:{self.id}"

    @property
    def argument_embedding(self) -> EmbeddingIdentifier:
        """Return the argument embedding, warning when it is empty."""
        if self._argument_embedding.is_empty():
            warnings.warn(
                "Argument embedding is empty. "
                "This might be due to the embedding not being loaded.",
                UserWarning,
                stacklevel=2,
            )
        return self._argument_embedding

    @argument_embedding.setter
    def argument_embedding(self, value: EmbeddingIdentifier) -> None:
        self._argument_embedding = value

    @property
    def target_claims_embedding(self) -> EmbeddingIdentifier:
        """Return the target-claims embedding, warning when it is empty."""
        if self._target_claims_embedding.is_empty():
            warnings.warn(
                "Target claims embedding is empty. "
                "This might be due to the embedding not being loaded.",
                UserWarning,
                stacklevel=2,
            )
        return self._target_claims_embedding

    @target_claims_embedding.setter
    def target_claims_embedding(self, value: EmbeddingIdentifier) -> None:
        self._target_claims_embedding = value

    def has_argument_embedding(self) -> bool:
        """Whether the argument embedding is loaded and usable for search."""
        return not self._argument_embedding.is_empty()

    def embedding_pair(self) -> tuple[EmbeddingIdentifier, EmbeddingIdentifier]:
        """
        Return (argument, target-claims) embeddings without empty warnings.

        Silent on purpose: persistence writes whatever is loaded, even when
        that is nothing. Use the properties when a missing embedding should
        warn.
        """
        return (self._argument_embedding, self._target_claims_embedding)

    @staticmethod
    def find_text_bounds(
        query: str, source_text: str, min_score: float = MIN_MATCH_SCORE
    ) -> TextBounds:
        """
        Locate query inside source_text: exact match first, fuzzy alignment after.

        Returns bounds of (-1, -1) when the text cannot be found.
        """
        if not query or not source_text:
            return TextBounds(start_idx=-1, end_idx=-1)

        exact_start = source_text.find(query)
        if exact_start != -1:
            return TextBounds(start_idx=exact_start, end_idx=exact_start + len(query))

        alignment = fuzz.partial_ratio_alignment(query, source_text)
        if alignment and alignment.score >= min_score:
            return TextBounds(
                start_idx=alignment.dest_start, end_idx=alignment.dest_end
            )

        return TextBounds(start_idx=-1, end_idx=-1)

    def place_in_carrier(
        self, document_text: str, document_id: str, batch_id: str | None = None
    ) -> bool:
        """
        Point this record at a carrier document, recomputing every text bound.

        Returns False when the argument cannot be located, leaving the record
        unplaced rather than storing bounds that do not match the text.
        """
        argument_bounds = ArgumentRecord.find_text_bounds(self.argument, document_text)
        if not argument_bounds.is_located:
            return False

        self.carrier = CarrierPlacement(
            document_id=document_id,
            argument=argument_bounds,
            premises=[
                ArgumentRecord.find_text_bounds(premise.text, document_text)
                for premise in self.premises
            ],
            batch_id=batch_id,
        )
        return True

    def build_embeddings(self) -> None:
        """Compute the argument and target-claim embeddings for this record."""
        domain = self.domain.value.upper()
        premises = "; ".join(premise.text for premise in self.premises)
        self.argument_embedding = EmbeddingIdentifier.create(
            content=f"[{domain}] Argument: {self.argument} Premises: {premises}"
        )
        self.target_claims_embedding = EmbeddingIdentifier.create(
            content=(
                f"[{domain}] Target Claims: {'; '.join(self.target_claims_hypothesis)}"
            )
        )

    @staticmethod
    def from_extracted(
        extracted: ExtractedArgument,
        provenance: ArgumentProvenance,
        native_counter_ids: list[str] | None = None,
        document_text: str | None = None,
        document_id: str | None = None,
    ) -> "ArgumentRecord":
        """
        Build a record from an extracted argument, with embeddings.

        Passing document_text + document_id also places it in that carrier;
        omitting them leaves `carrier=None` (a pure gold argument).
        """
        record = ArgumentRecord(
            **extracted.model_dump(by_alias=True),
            provenance=provenance,
            native_counter_ids=native_counter_ids or [],
        )
        record.build_embeddings()

        if document_text is not None and document_id is not None:
            record.place_in_carrier(document_text, document_id)
        return record


# --- Database ---


class ArgumentDatabase(BaseModel):
    """
    Arguments plus the links between them, with retrieval over embeddings.

    Two searches, opposite directions:
    - fetch_similar_arguments  : argument      -> argument  (says the same)
    - fetch_opposing_arguments : target claims -> argument  (says what we attack)
    """

    arguments: list[ArgumentRecord] = Field(
        default_factory=list, description="All argument records"
    )
    argument_links: set[ArgumentLink] = Field(
        default_factory=set, description="All argument links"
    )

    def __iadd__(self, argument: ArgumentRecord) -> Self:
        """Add an argument record to the database."""
        self.arguments.append(argument)
        return self

    def __len__(self) -> int:
        """Return the number of argument records held."""
        return len(self.arguments)

    def by_id(self) -> dict[str, ArgumentRecord]:
        """Map record id to record."""
        return {record.id: record for record in self.arguments}

    # --- Links ---

    def add_argument_link(
        self,
        source_argument: ArgumentRecord,
        target_argument: ArgumentRecord,
        link_type: ArgumentLinkType,
        confidence: float = 1.0,
    ) -> None:
        """Add a typed link between two arguments already in the database."""
        known_ids = {arg.id for arg in self.arguments}
        if source_argument.id not in known_ids:
            raise ValueError("Source argument not found in the database.")
        if target_argument.id not in known_ids:
            raise ValueError("Target argument not found in the database.")

        new_link = ArgumentLink(
            source_argument_id=source_argument.id,
            target_argument_id=target_argument.id,
            link_type=link_type,
            confidence=confidence,
        )

        if any(existing == new_link for existing in self.argument_links):
            raise ValueError("Link already exists in the database.")
        self.argument_links.add(new_link)

    def try_add_link(
        self,
        source_argument: ArgumentRecord,
        target_argument: ArgumentRecord,
        link_type: ArgumentLinkType,
        confidence: float = 1.0,
    ) -> bool:
        """Add a link, returning False when it is a self-link or already known."""
        if source_argument.id == target_argument.id:
            return False
        try:
            self.add_argument_link(
                source_argument, target_argument, link_type, confidence
            )
        except ValueError:
            return False
        return True

    def counterparts_of(self, record: ArgumentRecord) -> list[ArgumentRecord]:
        """List records linked to this one, in either direction."""
        records = self.by_id()
        partners: list[ArgumentRecord] = []
        for link in self.argument_links:
            partner_id: str | None = None
            if link.source_argument_id == record.id:
                partner_id = link.target_argument_id
            elif link.target_argument_id == record.id:
                partner_id = link.source_argument_id

            partner = records.get(partner_id) if partner_id else None
            if partner is not None:
                partners.append(partner)
        return partners

    def link_native_counters(
        self,
        link_type: ArgumentLinkType = ArgumentLinkType.CONTRARGUMENT,
        confidence: float = 1.0,
    ) -> int:
        """
        Turn the ids datasets gave us into links.

        | record (dataset, id 0:arg) | native_counter_ids ["0:counter"] |
                                  |
                                  v  exact match within the same source_dataset
        | ArgumentLink(0:arg -> 0:counter) |

        Returns the number of links created.
        """
        index: dict[tuple[str, str], ArgumentRecord] = {
            (record.provenance.source_dataset, record.provenance.original_id): record
            for record in self.arguments
            if record.provenance.original_id is not None
        }

        linked = 0
        for record in self.arguments:
            for native_id in record.native_counter_ids:
                target = index.get((record.provenance.source_dataset, native_id))
                if target is None:
                    continue
                if self.try_add_link(record, target, link_type, confidence):
                    linked += 1
        return linked

    # --- Combining ---

    def union(self, other: "ArgumentDatabase") -> "ArgumentDatabase":
        """
        Structural union of two databases. No link discovery.

        Records are deduplicated by provenance ("<source>:<original_id>"), so
        rebuilding a source and merging it again updates instead of doubling.
        Links of dropped duplicates are remapped onto the record that was kept.

        Cross-database links are NOT discovered here - that is the job of
        `pipelines/merge_arg_dbs.py`.
        """
        merged = ArgumentDatabase()
        kept_by_identity: dict[str, ArgumentRecord] = {}
        id_remap: dict[str, str] = {}

        for record in [*self.arguments, *other.arguments]:
            identity = record.identity
            kept = kept_by_identity.get(identity)
            if kept is None:
                kept_by_identity[identity] = record
                merged += record
                id_remap[record.id] = record.id
            else:
                id_remap[record.id] = kept.id

        for link in [*self.argument_links, *other.argument_links]:
            source_id = id_remap.get(link.source_argument_id)
            target_id = id_remap.get(link.target_argument_id)
            if source_id is None or target_id is None or source_id == target_id:
                continue
            merged.argument_links.add(
                ArgumentLink(
                    source_argument_id=source_id,
                    target_argument_id=target_id,
                    link_type=link.link_type,
                    confidence=link.confidence,
                )
            )

        return merged

    # --- Retrieval ---

    def _candidates(
        self,
        query: ArgumentRecord,
        domain_filter: Mapping[InterpretativeFrame, list[InterpretativeFrame]] | None,
        *,
        language_filter: bool,
    ) -> list[ArgumentRecord]:
        """Filter out the query itself, other languages and other domains."""
        candidates: list[ArgumentRecord] = []
        for record in self.arguments:
            if record.id == query.id:
                continue
            if language_filter and record.language != query.language:
                continue
            if domain_filter is not None:
                allowed = domain_filter.get(query.domain, [query.domain])
                if record.domain not in allowed:
                    continue
            candidates.append(record)
        return candidates

    def _hybrid_search(
        self,
        query_embedding: EmbeddingIdentifier,
        candidates: list[ArgumentRecord],
        top_k: int,
        alpha: float,
    ) -> list[tuple[ArgumentRecord, float]]:
        """
        Score candidates by alpha * dense + (1 - alpha) * sparse similarity.

        Records without embeddings are skipped rather than crashing the search:
        a database loaded with ``load_embeddings=False`` is searchable, it just
        has nothing to match on.
        """
        if not candidates or query_embedding.is_empty():
            return []

        candidates = [
            record for record in candidates if record.has_argument_embedding()
        ]
        if not candidates:
            return []

        dense_matrix = np.array(
            [record.argument_embedding.dense_embedding for record in candidates],
            dtype=np.float32,
        )
        query_dense = np.array(query_embedding.dense_embedding, dtype=np.float32)
        if (
            dense_matrix.ndim != DENSE_MATRIX_DIMS
            or dense_matrix.shape[1] != query_dense.size
        ):
            return []

        norm_matrix = np.linalg.norm(dense_matrix, axis=1)
        norm_query: float = float(np.linalg.norm(query_dense)) or 1e-10
        norm_matrix[norm_matrix == 0] = 1e-10
        dense_scores = np.dot(dense_matrix, query_dense) / (norm_matrix * norm_query)

        sparse_scores = np.zeros(len(candidates), dtype=np.float32)
        if query_embedding.sparse_embedding:
            for idx, record in enumerate(candidates):
                if record.argument_embedding.sparse_embedding:
                    sparse_scores[idx] = SparseEmbedder.sparse_cosine_similarity(
                        query_embedding.sparse_embedding,
                        record.argument_embedding.sparse_embedding,
                    )

        final_scores = (alpha * dense_scores) + ((1 - alpha) * sparse_scores)
        top_k_indices = np.argsort(final_scores)[::-1][:top_k]
        return [(candidates[idx], float(final_scores[idx])) for idx in top_k_indices]

    def fetch_similar_arguments(
        self,
        query: ArgumentRecord,
        top_k: int = 5,
        domain_filter: Mapping[InterpretativeFrame, list[InterpretativeFrame]]
        | None = None,
        *,
        language_filter: bool = True,
        alpha: float = 0.7,
    ) -> list[tuple[ArgumentRecord, float]]:
        """
        Find arguments that say roughly the same thing as the query.

        query.argument  ->  candidate.argument
        """
        candidates = self._candidates(
            query, domain_filter, language_filter=language_filter
        )
        return self._hybrid_search(query.argument_embedding, candidates, top_k, alpha)

    def fetch_opposing_arguments(
        self,
        query: ArgumentRecord,
        top_k: int = 5,
        domain_filter: Mapping[InterpretativeFrame, list[InterpretativeFrame]]
        | None = None,
        *,
        language_filter: bool = True,
        alpha: float = 0.7,
    ) -> list[tuple[ArgumentRecord, float]]:
        """
        Find counter-argument candidates: arguments asserting what the query attacks.

        query.target_claims_hypothesis  ->  candidate.argument

        These are candidates only. The LLM judge decides which are real
        counters and names the ArgumentLinkType.
        """
        candidates = self._candidates(
            query, domain_filter, language_filter=language_filter
        )
        return self._hybrid_search(
            query.target_claims_embedding, candidates, top_k, alpha
        )

    # --- Persistence ---

    def save(self, directory: Path) -> None:
        """Save the database into JSON records and a Pickle embedding map."""
        directory.mkdir(parents=True, exist_ok=True)
        json_path: Path = directory / ARG_DB_JSON_FILENAME
        embeddings_path: Path = directory / ARG_DB_EMBEDDINGS_FILENAME

        arguments_json = {
            "arguments": [
                json.loads(record.model_dump_json()) for record in self.arguments
            ],
            "links": [link.model_dump() for link in self.argument_links],
        }

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(arguments_json, f, indent=2, ensure_ascii=False)

        embeddings_map: dict[str, dict[str, EmbeddingIdentifier]] = {}
        for record in self.arguments:
            argument_embedding, target_claims_embedding = record.embedding_pair()
            embeddings_map[record.id] = {
                "argument_embedding": argument_embedding,
                "target_claims_embedding": target_claims_embedding,
            }

        with embeddings_path.open("wb") as f:
            pickle.dump(embeddings_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        load_embeddings: bool = True,
    ) -> "ArgumentDatabase":
        """
        Load the database primarily from JSON.
        Optionally load embeddings from the binary pickle file if available.
        """
        json_path: Path = directory / ARG_DB_JSON_FILENAME
        embeddings_path: Path = directory / ARG_DB_EMBEDDINGS_FILENAME

        if not json_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        db = cls()

        for arg_dict in data.get("arguments", []):
            db += ArgumentRecord(**arg_dict)

        for link_dict in data.get("links", []):
            db.argument_links.add(ArgumentLink(**link_dict))

        if load_embeddings:
            if not embeddings_path.exists():
                warnings.warn(
                    "Embeddings file not found at "
                    f"{embeddings_path}. Database loaded without embeddings.",
                    UserWarning,
                    stacklevel=2,
                )
                return db

            with embeddings_path.open("rb") as f:
                embeddings_map: dict[str, dict[str, EmbeddingIdentifier]] = pickle.load(  # noqa: S301
                    f
                )

            for record in db.arguments:
                if record.id in embeddings_map:
                    record.argument_embedding = embeddings_map[record.id][
                        "argument_embedding"
                    ]
                    record.target_claims_embedding = embeddings_map[record.id][
                        "target_claims_embedding"
                    ]

        return db


__all__ = [
    "ARG_DB_EMBEDDINGS_FILENAME",
    "ARG_DB_JSON_FILENAME",
    "ArgumentDatabase",
    "ArgumentLink",
    "ArgumentProvenance",
    "ArgumentRecord",
    "CarrierPlacement",
    "EmbeddingIdentifier",
    "ExtractedPremise",
    "PremiseSource",
    "TextBounds",
]
