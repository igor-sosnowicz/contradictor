"""Module with Milvus-based vector search."""

from typing import cast, override
from uuid import uuid4

import numpy as np
from pymilvus import MilvusClient

from src.configuration import config
from src.vector_search.vector_search import VectorSearch


class MilvusVectorSearch(VectorSearch):
    """Milvus-based implementation of vector search engine."""

    SEARCH_RESULTS_HARD_LIMIT = 100
    DATABASE_FILE = config.cache_directory / "vector_search" / "milvus" / "milvus.db"

    def __init__(self, client: MilvusClient | None = None) -> None:
        """
        Initialise a Milvus vector database client.

        Args:
            client (MilvusClient | None): An instance of a Milvus client of
                the vector database.
        """
        self.DATABASE_FILE.parent.mkdir(exist_ok=True, parents=True)
        self._client = client or MilvusClient(str(self.DATABASE_FILE))

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

        collection_name = f"collection_{uuid4()}"
        self._client.create_collection(
            collection_name, dimension=references[0].shape[0]
        )

        negated_embeddings = [-vector for vector in other]
        other_data = [
            {"id": i, "vector": vector} for i, vector in enumerate(negated_embeddings)
        ]

        _result = self._client.insert(
            collection_name,
            data=other_data,
        )

        results = self._client.search(
            collection_name=collection_name,
            data=[reference.tolist() for reference in references],
            limit=self.max_candidates,
        )

        self._client.drop_collection(collection_name)
        return [[cast("int", hit["id"]) for hit in hits] for hits in results]
