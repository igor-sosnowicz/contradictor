"""Argument database schema as Pydantic models."""

import json
import pickle
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Self
from uuid import uuid4

import numpy as np
from fastembed import SparseEmbedding
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from rapidfuzz import fuzz

from src.argument_dataset.arg_link import ArgumentLinkType
from src.argument_dataset.output_models import ExtractedArgument
from src.configuration import config as global_config
from src.data_models.data_models import InterpretativeFrame
from src.utils.fast_embedders import DenseEmbedderFactory, SparseEmbedder

if TYPE_CHECKING:
    from pathlib import Path


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

        dense_embedder = DenseEmbedderFactory.create(
            global_config.provider,
            server_url=global_config.lm_studio_api_base_url,
            api_key=global_config.lm_studio_api_key,
        )
        sparse_embedder = SparseEmbedder()

        return cls(
            dense_embedding=dense_embedder.embed(content),
            sparse_embedding=sparse_embedder.embed(content),
        )


class PremiseContext(BaseModel):
    """Context information for a premise."""

    text: str = Field(..., description="The premise text")
    start_idx: int = Field(
        ..., description="Start index of the premise in the argument"
    )
    end_idx: int = Field(..., description="End index of the premise in the argument")


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

    @classmethod
    def peek_hash(
        cls,
        source_argument_id: str,
        target_argument_id: str,
        link_type: ArgumentLinkType,
    ) -> int:
        """Precompute the set hash for a directed edge."""
        return hash((source_argument_id, target_argument_id, link_type))


class ArgumentRecord(ExtractedArgument):
    """Context information for an argument."""

    # General metadata
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the argument",
    )
    document_id: str = Field(..., description="The source document or file path")

    # Overrides
    premises: list[PremiseContext] = Field(default_factory=list)  # type: ignore[assignment]

    # Positioning
    start_idx: int = Field(
        ..., description="Start index of the argument in the source document"
    )
    end_idx: int = Field(
        ..., description="End index of the argument in the source document"
    )

    # Searchability (private attributes: excluded from JSON dumps by default)
    _argument_embedding: EmbeddingIdentifier = PrivateAttr(
        default_factory=EmbeddingIdentifier
    )
    _target_claims_embedding: EmbeddingIdentifier = PrivateAttr(
        default_factory=EmbeddingIdentifier
    )

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

    @staticmethod
    def _find_text_bounds(
        query: str, source_text: str, min_score: float = 75.0
    ) -> tuple[int, int]:
        """
        Find the start and end character indices of query within source_text.
        Tries exact match first, then falls back to rapidfuzz alignment.
        """
        if not query or not source_text:
            return -1, -1

        # 1. Exact match (fast path)
        exact_start = source_text.find(query)
        if exact_start != -1:
            return exact_start, exact_start + len(query)

        # 2. Fuzzy match alignment (fallback)
        alignment = fuzz.partial_ratio_alignment(query, source_text)
        if alignment and alignment.score >= min_score:
            return alignment.dest_start, alignment.dest_end

        return -1, -1

    @staticmethod
    def from_extracted(
        extracted: ExtractedArgument,
        document_text: str,
        document_id: str,
    ) -> "ArgumentRecord":
        """
        Create an ArgumentRecord from an ExtractedArgument.

        Calculates positions using fuzzy matching and generates embeddings.
        """
        # Matching the argument text to the document to find its position
        arg_start, arg_end = ArgumentRecord._find_text_bounds(
            extracted.argument, document_text
        )

        # Matching the premise texts to the document to find their positions
        premises_with_ctx = []
        for p in extracted.premises:
            p_start, p_end = ArgumentRecord._find_text_bounds(p.text, document_text)
            premises_with_ctx.append(
                PremiseContext(text=p.text, start_idx=p_start, end_idx=p_end)
            )

        # Embed identifiers for the argument and its target claims
        argument_embedding = EmbeddingIdentifier.create(
            content=f"\
            [{extracted.domain.value.upper()}] \
            Argument: {extracted.argument} \
            Premises: {'; '.join(p.text for p in extracted.premises)}"
        )
        target_claims_hypothesis = EmbeddingIdentifier.create(
            content=f"\
            [{extracted.domain.value.upper()}] \
            Target Claims: {'; '.join(extracted.target_claims_hypothesis)}"
        )

        record = ArgumentRecord(
            **extracted.model_dump(exclude={"premises"}, by_alias=True),
            document_id=document_id,
            start_idx=arg_start,
            end_idx=arg_end,
            premises=premises_with_ctx,
        )
        record.argument_embedding = argument_embedding
        record.target_claims_embedding = target_claims_hypothesis
        return record


class ArgumentDatabase(BaseModel):
    """
    The main db for arguments, stores links between arguments and their contexts.
    Holds filtering and retrieval methods for arguments based on various criteria.
    """

    arguments: list[ArgumentRecord] = Field(
        default_factory=list, description="Set of all argument records"
    )
    argument_links: set[ArgumentLink] = Field(
        default_factory=set, description="Set of all argument links"
    )

    def __iadd__(self, argument: ArgumentRecord) -> Self:
        """Add an argument record to the database."""
        self.arguments.append(argument)
        return self

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

    def fetch_similar_arguments(
        self,
        query: ArgumentRecord,
        top_k: int = 5,
        domain_filter: Mapping[InterpretativeFrame, list[InterpretativeFrame]]
        | None = None,
        language_filter: bool = True,  # noqa: FBT001, FBT002
        alpha: float = 0.7,  # weight: alpha * Dense + (1 - alpha) * Sparse
    ) -> list[tuple[ArgumentRecord, float]]:
        """
        Fetch the top_k most similar arguments to the query using embeddings.

            Filter by:
                - domain (1:n soft filter), default: Hard filter
                - language (1:1 hard filter)

        Compares with:
            - Dense embedding for similarity.
            - Sparse embedding for keyword accuracy.
        """
        if not self.arguments:
            return []

        # Early filtering
        candidate_indices: list[int] = []
        for idx, arg in enumerate(self.arguments):
            _self_filter = arg.id == query.id
            _language_filter = (
                arg.language != query.language if language_filter else False
            )
            if domain_filter is None:
                _domain_filter = False
            else:
                allowed = domain_filter.get(query.domain, [query.domain])
                _domain_filter = arg.domain not in allowed

            if any([_self_filter, _language_filter, _domain_filter]):
                continue

            candidate_indices.append(idx)

        if not candidate_indices:
            return []

        candidates = [self.arguments[i] for i in candidate_indices]

        # Dense Matrix (N, D)
        dense_matrix = np.array(
            [arg.argument_embedding.dense_embedding for arg in candidates],
            dtype=np.float32,
        )

        # Query vector (D,)
        query_dense = np.array(
            query.target_claims_embedding.dense_embedding, dtype=np.float32
        )

        # Dense Similarity
        norm_matrix = np.linalg.norm(dense_matrix, axis=1)
        norm_query: float = float(np.linalg.norm(query_dense))
        norm_matrix[norm_matrix == 0] = 1e-10
        if norm_query == 0:
            norm_query = 1e-10

        dense_scores = np.dot(dense_matrix, query_dense) / (norm_matrix * norm_query)

        # Sparse Similarity
        sparse_scores = np.zeros(len(candidates), dtype=np.float32)
        if query.target_claims_embedding.sparse_embedding:
            for idx, arg in enumerate(candidates):
                if arg.argument_embedding.sparse_embedding:
                    sparse_scores[idx] = SparseEmbedder.sparse_cosine_similarity(
                        query.target_claims_embedding.sparse_embedding,
                        arg.argument_embedding.sparse_embedding,
                    )

        # Hybrid Score
        final_scores = (alpha * dense_scores) + ((1 - alpha) * sparse_scores)
        top_k_indices = np.argsort(final_scores)[::-1][:top_k]
        return [(candidates[idx], float(final_scores[idx])) for idx in top_k_indices]

    def save(self) -> None:
        """Save the database into JSON records and a Pickle embedding map."""
        json_path: Path = global_config.counterargument_dataset_path
        embeddings_path: Path = global_config.counterargument_dataset_embeddings_path

        arguments_json = {
            "arguments": [
                json.loads(
                    arg.model_dump_json(
                        exclude={"_argument_embedding", "_target_claims_embedding"}
                    )
                )
                for arg in self.arguments
            ],
            "links": [link.model_dump() for link in self.argument_links],
        }

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(arguments_json, f, indent=2, ensure_ascii=False)

        embeddings_map: dict[str, dict[str, EmbeddingIdentifier]] = {}
        for arg in self.arguments:
            embeddings_map[arg.id] = {
                "argument_embedding": arg._argument_embedding,  # noqa: SLF001
                "target_claims_embedding": arg._target_claims_embedding,  # noqa: SLF001
            }

        with embeddings_path.open("wb") as f:
            pickle.dump(embeddings_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, load_embeddings: bool = True) -> "ArgumentDatabase":  # noqa: FBT001, FBT002
        """
        Load the database primarily from JSON.
        Optionally load embeddings from the binary pickle file if available.
        """
        json_path: Path = global_config.counterargument_dataset_path
        embeddings_path: Path = global_config.counterargument_dataset_embeddings_path

        if not json_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        db = cls()

        for arg_dict in data.get("arguments", []):
            record = ArgumentRecord(**arg_dict)
            db += record

        for link_dict in data.get("links", []):
            db.argument_links.add(ArgumentLink(**link_dict))

        # Optional Lazy / Fast Loading
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
                loaded = pickle.load(f)  # noqa: S301
                embeddings_map: dict[str, dict[str, EmbeddingIdentifier]] = loaded

            for arg in db.arguments:
                if arg.id in embeddings_map:
                    arg._argument_embedding = embeddings_map[arg.id][  # noqa: SLF001
                        "argument_embedding"
                    ]
                    arg._target_claims_embedding = embeddings_map[arg.id][  # noqa: SLF001
                        "target_claims_embedding"
                    ]

        return db
