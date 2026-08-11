"""spaCy-based sentence embedder."""

from typing import TYPE_CHECKING, cast

import numpy as np
import spacy
from tqdm import tqdm

from src.configuration import config

if TYPE_CHECKING:
    from src.argument_detection.config import SpacyConfig


class SpacyEmbedder:
    """Generate dense vector embeddings for text using a spaCy pipeline."""

    def __init__(self) -> None:
        """Initialize the embedder and load the configured spaCy pipeline."""
        self._config: SpacyConfig = config.xgboost_extractor.spacy
        self._nlp = self._load_pipeline()

    def _load_pipeline(self) -> spacy.language.Language:
        """
        Load the configured spaCy model.

        Returns:
            spacy.language.Language: Loaded spaCy language pipeline.
        """
        import spacy.cli

        try:
            return spacy.load(
                self._config.model,
                exclude=self._config.exclude,
            )
        except OSError:
            spacy.cli.download(self._config.model)
            return spacy.load(
                self._config.model,
                exclude=self._config.exclude,
            )

    def embed(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Generate dense vector embeddings for a batch of texts.

        Args: texts (list[str]): Input texts to embed.

        Returns: np.ndarray: A 2D array of shape ``(len(texts), embedding_dim)``
                containing one embedding vector per input text.
        """
        return np.vstack(
            [
                cast("np.ndarray", doc.vector)
                for doc in tqdm(
                    self._nlp.pipe(
                        texts,
                        batch_size=self._config.batch_size,
                        n_process=self._config.n_process,
                    ),
                    total=len(texts),
                    desc="Embedding texts",
                )
            ]
        )
