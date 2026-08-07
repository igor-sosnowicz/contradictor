"""spaCy-based sentence embedder."""

import subprocess
import sys
from typing import cast

import numpy as np
import spacy
from tqdm import tqdm

from src.configuration import config


class SpacyEmbedder:
    """Generate dense vector embeddings for text using a spaCy pipeline."""

    def __init__(self) -> None:
        """Initialize the embedder and load the configured spaCy pipeline."""
        self._config = config.xgboost_extractor.spacy
        self._nlp = self._load_pipeline()

    def _load_pipeline(
        self,
    ) -> spacy.language.Language:
        """
        Load the configured spaCy model.

        If the model is not installed locally, it is downloaded automatically
        and loaded afterwards.

        Returns:
            spacy.language.Language: Loaded spaCy language pipeline.
        """
        try:
            return spacy.load(
                self._config.model,
                exclude=self._config.exclude,
            )
        except OSError:
            subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "spacy",
                    "download",
                    self._config.model,
                ],
                check=True,
            )
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

        Args:
            texts (list[str]): Input texts to embed.

        Returns:
            np.ndarray: A 2D array of shape ``(len(texts), embedding_dim)``
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
