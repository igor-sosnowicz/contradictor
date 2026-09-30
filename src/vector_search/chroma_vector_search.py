"""Module with a Chroma-based vector search implementation."""

from typing import override
from uuid import uuid4

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from src.vector_search.vector_search import VectorSearch


class ChromaVectorSearch(VectorSearch):
    """The chroma-based vector database."""

    HARD_LIMIT_MAX_CANDIDATES = 100

    def __init__(self, chroma_client: ClientAPI | None = None) -> None:
        """
        Initialise a Chroma client.

        Args:
            chroma_client (ClientAPI | None): An instance of Chroma client to be used.
                Defaults to a new in-memory instance.
        """
        self._client = chroma_client or chromadb.EphemeralClient()

    def _query_with_batching(
        self, collection: Collection, query_embeddings: list[np.ndarray], n_results: int
    ) -> list[list[str]]:
        """Split queries into batches to avoid 'too many SQL variables' error."""
        batch_size = 1024
        all_ids = []

        for i in range(0, len(query_embeddings), batch_size):
            batch = query_embeddings[i : i + batch_size]
            batch_results = collection.query(
                query_embeddings=batch,
                n_results=n_results,
                include=[],  # Return only IDs.
            )
            all_ids.extend(batch_results["ids"])

        return all_ids

    @override
    async def max_distance(
        self,
        references: list[np.ndarray],
        other: list[np.ndarray],
        max_candidates: int | None = None,
    ) -> list[list[int]]:
        output = self._validate_arguments(
            references=references, other=other, max_candidates=max_candidates
        )
        if output is not None:
            return output

        n_candidates = min(self.max_candidates, len(other))
        collection_name = f"temporary_collection_{uuid4()}"

        collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Pre-negate vectors.
        negated_embeddings = [-vector for vector in other]
        other_ids = [str(i) for i in range(len(other))]

        try:
            collection.add(
                ids=other_ids,
                embeddings=negated_embeddings,
            )

            results_ids = self._query_with_batching(
                collection=collection,
                query_embeddings=references,
                n_results=n_candidates,
            )

            # Convert string IDs back to int indices.
            return [
                [int(vector_id) for vector_id in reference_ids]
                for reference_ids in results_ids
            ]

        finally:
            self._client.delete_collection(name=collection_name)
