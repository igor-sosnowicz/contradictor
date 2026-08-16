"""spaCy-based sentence embedder."""

from typing import TYPE_CHECKING, cast

import numpy as np
import spacy
from spacy.cli import download
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
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _load_pipeline(self) -> spacy.language.Language:
        """
        Load the configured spaCy model.

        Returns:
            spacy.language.Language: Loaded spaCy language pipeline.
        """
        try:
            return spacy.load(
                self._config.model,
                exclude=self._config.exclude,
            )
        except OSError:
            download(self._config.model)
            return spacy.load(
                self._config.model,
                exclude=self._config.exclude,
            )

    def embed(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Generate dense vector embeddings for text using a local cache."""
        missing_texts = [text for text in texts if text not in self._embedding_cache]

        if missing_texts:
            embeddings = [
                cast("np.ndarray", doc.vector)
                for doc in tqdm(
                    self._nlp.pipe(
                        missing_texts,
                        batch_size=self._config.batch_size,
                        n_process=self._config.n_process,
                    ),
                    total=len(missing_texts),
                    desc="Embedding texts",
                )
            ]

            for text, embedding in zip(
                missing_texts,
                embeddings,
                strict=True,
            ):
                self._embedding_cache[text] = embedding

        return np.vstack([self._embedding_cache[text] for text in texts])
