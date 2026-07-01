"""Module with XGBoost for argument detection."""

import pickle
import subprocess
import sys
from pathlib import Path
from typing import Final, cast, override

import numpy as np
import pandas as pd
import spacy
from diskcache import Cache
from sklearn.metrics import f1_score
from tqdm import tqdm
from xgboost import XGBClassifier

from src.argument_detection.argument_detection_dataset import ArgumentDetectionDataset
from src.argument_detection.argument_detector import ArgumentDetector
from src.configuration import config
from src.data_models.data_models import SubsetName
from src.utils.sentence_splitter import SentenceSplitter


class XGBoostDetector(ArgumentDetector):
    """Class for XGBoost argument detection."""

    PATH_TO_MODEL: Final = Path(
        config.data_directory / config.model_subdirectory / "xgboost_detector.pkl"
    )
    CACHE_DIRECTORY: Final = Path(config.cache_directory / "xgboost")

    def __init__(
        self, dataset: ArgumentDetectionDataset, *, proof_of_concept_mode: bool = False
    ) -> None:
        """
        Initialise required prerequisites for detector.

        Args:
            dataset (ArgumentDetectionDataset): Preprocessed dataset to be used.
            proof_of_concept_mode (bool, optional): Whether the detector should be used
                in a proof of concept mode where a limited number of examples is used.
                Defaults to False (all samples are used).
        """
        self.PATH_TO_MODEL.parent.mkdir(parents=True, exist_ok=True)
        self._sentence_splitter = SentenceSplitter()
        self._dataset = dataset
        self._model: XGBClassifier | None = None

        # Cache
        self.CACHE_DIRECTORY.mkdir(exist_ok=True, parents=True)
        self._cache = Cache(self.CACHE_DIRECTORY)

        self._threshold: float | None = self._cache.get("threshold")

        # Proof of concept mode.
        self._max_samples: int | None = 100_000 if proof_of_concept_mode else None

        # Text embedding pipeline.
        self._nlp = self._prepare_spacy_embedding_pipeline()

    def _prepare_spacy_embedding_pipeline(self) -> spacy.language.Language:
        spacy_pipeline_name = "en_core_web_sm"
        exclude = [
            "tagger",
            "parser",
            "senter",
            "attribute_ruler",
            "lemmatizer",
            "ner",
        ]

        try:
            return spacy.load(spacy_pipeline_name, exclude=exclude)
        except OSError:
            subprocess.run(  # noqa: S603, Trused spacy module.
                [sys.executable, "-m", "spacy", "download", spacy_pipeline_name],
                check=True,
            )
            return spacy.load(spacy_pipeline_name, exclude=exclude)

    @override
    def _load(self) -> XGBClassifier | None:
        if not self.PATH_TO_MODEL.exists():
            return None
        with self.PATH_TO_MODEL.open("rb") as file:
            return pickle.load(file)  # noqa: S301, Trusted source.

    @override
    async def _train(self) -> XGBClassifier:
        model = XGBClassifier()

        await self._dataset.prepare()
        df = self._dataset.get_split(SubsetName.TRAINING, max_samples=self._max_samples)

        # Generate features.
        X = pd.DataFrame(
            np.vstack(self._embed_texts(df["sentence"].tolist())),
            index=df.index,
        )

        y = df["is_argument"].astype("int32")

        model.fit(
            X,
            y,
            verbose=True,
        )

        with self.PATH_TO_MODEL.open("wb") as file:
            pickle.dump(model, file)

        return model

    def _embed_texts(
        self,
        texts: list[str],
    ) -> list[np.ndarray]:
        """
        Convert texts into fixed-size embedding vectors.

        Args:
            texts (list[str]): Texts to be embedded.

        Returns:
            list[np.ndarray]: List of fixed-size embedding vectors.
        """
        return [
            cast("np.ndarray", document.vector)
            for document in tqdm(
                self._nlp.pipe(texts, batch_size=1024, n_process=-1),
                total=len(texts),
                desc="Embedding texts",
            )
        ]

    async def _initialise_model(self, *, train_if_missing: bool) -> None:
        if self._model is not None:
            return

        self._model = self._load()
        if self._model is not None:
            return

        if not train_if_missing:
            raise RuntimeError("Model must be trained before tuning.")

        self._model = await self._train()
        # Adjust the threshold after training.
        await self.perform_tuning()

    @override
    async def detect(self, text: str) -> list[str]:
        await self._initialise_model(train_if_missing=True)

        # Convert sentence list to fixed-size features.
        sentences = list(self._sentence_splitter(text))
        features_batch = np.vstack(self._embed_texts(texts=sentences))

        # Extract probability of positive class (class 1)
        predictions = self._model.predict_proba(features_batch)[:, 1]  # type: ignore[union-attr]

        # Set threshold experimentally if missing.
        if self._threshold is None:
            await self.perform_tuning()

        # Filter by threshold.
        return [
            sentence
            for sentence, pred in zip(sentences, predictions, strict=True)
            if pred >= self._threshold
        ]

    @override
    async def perform_tuning(self) -> dict[str, float]:
        """Tune the classification threshold on the validation set."""
        await self._initialise_model(train_if_missing=False)

        # Get samples.
        await self._dataset.prepare()
        df = self._dataset.get_split(
            SubsetName.VALIDATION, max_samples=self._max_samples
        )

        X = np.vstack(self._embed_texts(df["sentence"].tolist()))
        y = df["is_argument"].astype("int32").to_numpy()

        probabilities = self._model.predict_proba(X)[:, 1]  # type: ignore[union-attr]

        best_threshold = 0.5
        best_score = -1.0

        for threshold in np.linspace(0.0, 1.0, 101):
            predictions = probabilities >= threshold
            score = f1_score(y, predictions)

            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        # Store the best threshold.
        self._threshold = best_threshold
        self._cache.set("threshold", self._threshold)

        return {
            "threshold": best_threshold,
            "f1": float(best_score),
        }
