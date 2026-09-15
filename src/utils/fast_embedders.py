"""Dense and sparse text embedders (local FastEmbed or LM Studio API)."""

import warnings
from abc import ABC, abstractmethod
from functools import lru_cache
from math import sqrt
from typing import TYPE_CHECKING, Literal

from fastembed import SparseEmbedding
from numpy import array, ndarray
from numpy.linalg import norm

from src.configuration import config as global_config
from src.utils.errors import RETRIABLE_LM_STUDIO_ERRORS, ConfigurationError
from src.utils.retry import DEFAULT_RETRY_POLICY, RetrySettings, retry

if TYPE_CHECKING:
    from openai.types import CreateEmbeddingResponse


class DenseEmbedderBase(ABC):
    """Interface for dense text embedders."""

    @abstractmethod
    def embed(self, text: str, **kwargs: object) -> ndarray:
        """Embed a single text."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str], **kwargs: object) -> list[ndarray]:
        """Embed a batch of texts."""
        ...

    @staticmethod
    def dense_cosine_similarity(v1: ndarray, v2: ndarray) -> float:
        """Compute cosine similarity between two dense vectors."""
        magnitude = norm(v1) * norm(v2)
        if not magnitude:
            return 0.0
        cos_sim: float = ((v1 @ v2.T) / magnitude).item()
        return cos_sim


class DenseEmbedderAPI(DenseEmbedderBase):
    """Dense embedder served by the LM Studio OpenAI-compatible endpoint."""

    def __init__(
        self,
        server_url: str,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        retry_policy: RetrySettings | None = None,
    ) -> None:
        """Create the embedder for the given LM Studio server address."""
        from openai import OpenAI

        self.model_name = model_name
        base_url = f"http://{server_url}/v1"
        api_key = api_key or global_config.lm_studio_api_key

        self.retry = retry_policy or DEFAULT_RETRY_POLICY
        self.openai_client = OpenAI(base_url=base_url, api_key=api_key)

    @retry(on=RETRIABLE_LM_STUDIO_ERRORS)
    def embed(self, text: str, **kwargs: object) -> ndarray:
        """Embed a single text, retrying transient API failures."""
        response: CreateEmbeddingResponse = self.openai_client.embeddings.create(
            input=text,
            model=self.model_name,
            **kwargs,  # type: ignore[arg-type]
        )
        return array(response.data[0].embedding)

    @retry(on=RETRIABLE_LM_STUDIO_ERRORS)
    def embed_batch(self, texts: list[str], **kwargs: object) -> list[ndarray]:
        """Embed a batch of texts, retrying transient API failures."""
        response: CreateEmbeddingResponse = self.openai_client.embeddings.create(
            input=texts,
            model=self.model_name,
            **kwargs,  # type: ignore[arg-type]
        )
        return [array(item.embedding) for item in response.data]


class DenseEmbedderLocal(DenseEmbedderBase):
    """Dense embedder running locally via FastEmbed."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        """Load the local FastEmbed model."""
        from fastembed import TextEmbedding

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = TextEmbedding(
                model_name, cache_dir=str(global_config.fastembed_model_cache_directory)
            )

    def embed(self, text: str, **kwargs: object) -> ndarray:
        """Embed a single text."""
        return next(iter(self.model.embed(documents=text, **kwargs)))  # type: ignore[arg-type]

    def embed_batch(self, texts: list[str], **kwargs: object) -> list[ndarray]:
        """Embed a batch of texts."""
        return list(self.model.embed(documents=texts, **kwargs))  # type: ignore[arg-type]


class DenseEmbedderFactory:
    """Factory for dense embedders."""

    @staticmethod
    def create(
        provider: Literal["api", "local"],
        model_name: str | None = None,
        server_url: str | None = None,
        api_key: str | None = None,
    ) -> DenseEmbedderBase:
        """Create a dense embedder for the given provider."""
        if provider == "api":
            if server_url is None:
                raise ConfigurationError(
                    "server_url must be provided for API provider."
                )
            return DenseEmbedderAPI(
                model_name=model_name or "text-embedding-3-small",
                server_url=server_url,
                api_key=api_key,
            )
        if provider == "local":
            return DenseEmbedderLocal(
                model_name=model_name
                or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        raise ValueError(f"Unknown provider: {provider}. Use one of ['api', 'local']")


class SparseEmbedder:
    """Sparse (BM25) embedder running locally via FastEmbed."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        """Load the local sparse model."""
        from fastembed import SparseTextEmbedding

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = SparseTextEmbedding(
                model_name, cache_dir=str(global_config.fastembed_model_cache_directory)
            )

    def embed(self, text: str, **kwargs: object) -> SparseEmbedding:
        """Embed a single text."""
        return next(iter(self.model.embed(documents=text, **kwargs)))  # type: ignore[arg-type]

    def embed_batch(self, texts: list[str], **kwargs: object) -> list[SparseEmbedding]:
        """Embed a batch of texts."""
        return list(self.model.embed(documents=texts, **kwargs))  # type: ignore[arg-type]

    @staticmethod
    def sparse_cosine_similarity(v1: SparseEmbedding, v2: SparseEmbedding) -> float:
        """Compute cosine similarity between two sparse embeddings."""
        v1_dict = v1.as_dict()
        v2_dict = v2.as_dict()
        intersecting_keys = set(v1_dict.keys()) & set(v2_dict.keys())
        dot_product = sum(v1_dict[k] * v2_dict[k] for k in intersecting_keys)

        mag1 = sqrt(sum(val**2 for val in v1_dict.values()))
        mag2 = sqrt(sum(val**2 for val in v2_dict.values()))

        if not mag1 or not mag2:
            return 0.0

        return dot_product / (mag1 * mag2)


# --- Multi-Call-save Initializers ---


@lru_cache(maxsize=4)
def create_dense_embedder(
    provider: str, server_url: str, api_key: str
) -> DenseEmbedderBase:
    """
    Return the dense embedder for a configuration, loading each model once.

    Building the model is expensive, so keep it in cache.
    """
    return DenseEmbedderFactory.create(
        provider,  # type: ignore[arg-type]
        server_url=server_url,
        api_key=api_key,
    )


@lru_cache(maxsize=1)
def create_sparse_embedder() -> SparseEmbedder:
    """Return the shared sparse embedder, loading its model once."""
    return SparseEmbedder()
