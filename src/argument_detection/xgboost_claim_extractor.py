"""XGBoost model for claim extraction."""

from typing import Final, override

from xgboost import XGBClassifier

from src.argument_detection.argument_detection_dataset import (
    ArgumentDetectionDataset,
)
from src.argument_detection.base_xgboost_extractor import (
    BaseXGBoostExtractor,
)
from src.configuration import config
from src.data_models.data_models import SubsetName


class XGBoostClaimExtractor(BaseXGBoostExtractor):
    """
    XGBoost-based claim extraction model.

    This extractor uses spaCy sentence embeddings combined with an
    XGBoost classifier to identify claim sentences in documents.

    The class is responsible for:
    - preparing text embeddings,
    - training the classifier,
    - loading persisted models,
    - extracting claims from text,
    - tuning the classification threshold.
    """

    PATH_TO_THRESHOLD: Final = (
        config.data_directory
        / config.model_subdirectory
        / "xgboost_claim_threshold.pkl"
    )

    def __init__(
        self,
        dataset: ArgumentDetectionDataset,
        *,
        proof_of_concept_mode: bool = False,
    ) -> None:
        """
        Initialize XGBoost extractor.

        Args:
            dataset:
                Dataset used for training and validation.

            model_name:
                Name of the persisted model file.

            cache_name:
                Name of the cache directory used for storing metadata.

            proof_of_concept_mode:
                Whether to limit dataset size for faster experiments.
        """
        super().__init__(
            dataset,
            model_name="xgboost_claim_extractor.pkl",
            cache_name="claim_extractor",
            proof_of_concept_mode=proof_of_concept_mode,
        )

    def _get_max_samples(self) -> int:
        """Return claim extractor proof-of-concept limit."""
        return self._config.dataset.claim_proof_of_concept_max_samples

    async def _train(
        self,
    ) -> XGBClassifier:
        """
        Train the XGBoost claim classifier.

        The method prepares the training dataset, generates sentence
        embeddings, trains the classifier, and stores the trained model
        and default classification threshold.

        Returns:
            XGBClassifier:
                Trained claim classification model.

        Raises:
            DatasetError:
                If dataset preparation or loading fails.
        """
        await self._dataset.prepare()

        df = self._dataset.get_claim_split(
            SubsetName.TRAINING,
            max_samples=self._max_samples,
        )
        X = self._embedder.embed(df["sentence"].astype(str).tolist())
        y = df["is_claim"].astype("int32")
        training = self._config.training
        model = XGBClassifier(
            n_estimators=training.n_estimators,
            max_depth=training.max_depth,
            learning_rate=training.learning_rate,
            subsample=training.subsample,
            colsample_bytree=training.colsample_bytree,
            tree_method=training.tree_method,
            objective=training.objective,
            eval_metric=training.eval_metric,
            random_state=training.random_state,
            n_jobs=training.n_jobs,
        )

        model.fit(X, y)
        self._save_model(model)
        return model

    @override
    async def extract_claims(
        self,
        text: str,
    ) -> list[str]:
        """
        Extract claim sentences from a document.

        The input text is split into sentences, transformed into embeddings,
        and classified using the trained XGBoost model.

        Args:
            text (str):
                Document or text fragment to analyze.

        Returns:
            list[str]:
                Sentences classified as claims.

        Raises:
            RuntimeError:
                If the model cannot be initialized.
        """
        await self._initialise_model()
        if self._model is None:
            raise RuntimeError("Model was not initialized.")
        sentences = list(self._sentence_splitter(text))
        if not sentences:
            return []
        X = self._embedder.embed(sentences)
        probabilities = (
            self._model.predict_proba(X)[:, 1]  # type: ignore[union-attr]
        )
        threshold = self._config.threshold.claim
        return [
            sentence
            for sentence, probability in zip(
                sentences,
                probabilities,
                strict=True,
            )
            if probability >= threshold
        ]

    @override
    async def extract_evidence(
        self,
        claim: str,
        text: str,
    ) -> list[str]:
        raise NotImplementedError(
            "Claim extractor does not support evidence extraction.",
        )

    @override
    async def perform_tuning(
        self,
    ) -> dict[str, float | int]:
        """Tune claim classifier."""
        await self._dataset.prepare()

        df = self._dataset.get_claim_split(
            SubsetName.TRAINING,
            max_samples=self._max_samples,
        )

        X = self._embedder.embed(
            df["sentence"].astype(str).tolist(),
        )

        y = df["is_claim"].astype("int32").to_numpy()

        best_params = self._tune_hyperparameters(
            X,
            y,
        )

        self._model = XGBClassifier(
            objective=self._config.training.objective,
            eval_metric=self._config.training.eval_metric,
            tree_method=self._config.training.tree_method,
            random_state=self._config.training.random_state,
            n_jobs=self._config.training.n_jobs,
            **best_params,
        )

        self._model.fit(
            X,
            y,
        )

        return {
            **best_params,
        }
