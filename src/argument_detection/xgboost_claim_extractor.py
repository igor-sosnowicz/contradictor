"""XGBoost model for claim extraction."""

from typing import override

from xgboost import XGBClassifier

from src.argument_detection.argument_detection_dataset import (
    ArgumentDetectionDataset,
)
from src.argument_detection.base_xgboost_extractor import (
    BaseXGBoostExtractor,
)
from src.data_models.data_models import SubsetName
from src.utils.errors import ModelNotTrainedError


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

    def __init__(
        self,
        dataset: ArgumentDetectionDataset,
        *,
        proof_of_concept_mode: bool = False,
    ) -> None:
        """
        Initialize XGBoost extractor.

        Args:
            dataset (ArgumentDetectionDataset): Dataset used for training and
                validation.
            proof_of_concept_mode (bool): Whether to limit dataset size for
                faster experiments.
        """
        super().__init__(
            dataset,
            model_name="xgboost_claim_extractor.json",
            proof_of_concept_mode=proof_of_concept_mode,
        )

    def _get_max_samples(self) -> int:
        """Return claim extractor proof-of-concept limit."""
        return self._config.dataset.claim_proof_of_concept_max_samples

    async def _train_model(
        self,
    ) -> XGBClassifier:
        """
        Train the XGBoost claim classifier.

        Returns:
            XGBClassifier: Trained claim classification model.

        Raises:
            DatasetError: If dataset preparation or loading fails.
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

    async def extract_claims(
        self,
        text: str,
    ) -> list[str]:
        """
        Extract claim sentences from a document.

        Args:
            text (str): Document or text fragment to analyze.

        Returns:
            list[str]: Sentences classified as claims.

        Raises:
            RuntimeError: If the model cannot be initialized.
        """
        await self._initialise_model()
        if self._model is None:
            raise ModelNotTrainedError("Model was not initialized.")
        sentences = list(self._sentence_splitter(text))
        if not sentences:
            return []
        X = self._embedder.embed(sentences)
        model = self._model
        if model is None:
            raise ModelNotTrainedError("Model was not initialized.")
        probabilities = model.predict_proba(X)[:, 1]
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
    async def perform_tuning(
        self,
    ) -> dict[str, float | int]:
        await self._dataset.prepare()

        df = self._dataset.get_claim_split(
            SubsetName.TRAINING,
            max_samples=self._max_samples,
        )

        x = self._embedder.embed(
            df["sentence"].astype(str).tolist(),
        )

        y = df["is_claim"].astype("int32").to_numpy()

        return self._apply_tuned_parameters(
            x,
            y,
        )

    async def extract_claims_batch(self, texts: list[str]) -> list[list[str]]:
        """
        Extract claim sentences from a batch of documents in a vectorised manner.

        Args:
            texts (list[str]): List of documents or text fragments to analyze.

        Returns:
            list[list[str]]: Lists of sentences classified as claims,
                one list per input document. Order is preserved.

        Raises:
            ModelNotTrainedError: If the model is not trained or initialized.
        """
        await self._initialise_model()
        if self._model is None:
            raise ModelNotTrainedError("Model was not initialised.")

        all_candidates: list[str] = []
        doc_mapping: list[int] = []

        for doc_idx, text in enumerate(texts):
            sentences = [s.strip() for s in self._sentence_splitter(text) if s.strip()]
            for sentence in sentences:
                all_candidates.append(sentence)
                doc_mapping.append(doc_idx)

        if not all_candidates:
            return [[] for _ in texts]

        X = self._embedder.embed(all_candidates)
        probabilities = self._model.predict_proba(X)[:, 1]

        threshold = self._config.threshold.claim

        results: list[list[str]] = [[] for _ in texts]
        zipped_data = zip(
            all_candidates,
            probabilities,
            doc_mapping,
            strict=True,
        )
        for sentence, prob, doc_idx in zipped_data:
            if prob >= threshold:
                results[doc_idx].append(sentence)

        return results
