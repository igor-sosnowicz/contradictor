"""XGBoost model for claim-evidence pair extraction."""

import pickle
from pathlib import Path
from typing import Final, override

import numpy as np
from xgboost import XGBClassifier

from src.argument_detection.argument_detection_dataset import (
    ArgumentDetectionDataset,
)
from src.argument_detection.base_xgboost_extractor import (
    BaseXGBoostExtractor,
)
from src.configuration import config
from src.data_models.data_models import SubsetName


class XGBoostEvidenceExtractor(BaseXGBoostExtractor):
    """
    XGBoost-based evidence extraction model.

    This extractor identifies sentences that provide evidence for a given
    claim using sentence embeddings and an XGBoost binary classifier.

    The class is responsible for:
    - preparing text embeddings,
    - training the classifier,
    - loading persisted models,
    - extracting supporting evidence,
    - tuning the classification threshold.
    """

    PATH_TO_MODEL: Final = Path(
        config.data_directory
        / config.model_subdirectory
        / "xgboost_evidence_extractor.pkl"
    )

    CACHE_DIRECTORY: Final = Path(config.cache_directory / "evidence_extractor")

    def __init__(
        self,
        dataset: ArgumentDetectionDataset,
        *,
        proof_of_concept_mode: bool = False,
    ) -> None:
        """
        Initialize the XGBoost evidence extractor.

        Args:
            dataset (ArgumentDetectionDataset):
                Dataset used for training and validation.

            proof_of_concept_mode (bool):
                Whether to limit the number of samples during training
                for faster experimentation.

        Returns:
            None.
        """
        super().__init__(
            dataset,
            model_name="xgboost_evidence_extractor.pkl",
            cache_name="evidence_extractor",
            proof_of_concept_mode=proof_of_concept_mode,
        )

    def _get_max_samples(self) -> int:
        """Return evidence extractor proof-of-concept limit."""
        return self._config.dataset.evidence_proof_of_concept_max_samples

    def _create_features(
        self,
        claims: list[str],
        evidences: list[str],
    ) -> np.ndarray:
        """
        Create feature vectors from claim-evidence pairs.

        Claim and evidence embeddings are concatenated into a single feature
        representation used by the classifier.

        Args:
            claims (list[str]):
                Claim sentences.

            evidences (list[str]):
                Evidence sentences associated with claims.

        Returns:
            np.ndarray:
                Feature matrix containing combined claim and evidence embeddings.
        """
        claim_embeddings = self._embedder.embed(claims)
        evidence_embeddings = self._embedder.embed(evidences)
        return np.hstack(
            [
                claim_embeddings,
                evidence_embeddings,
            ]
        )

    async def _train(
        self,
    ) -> XGBClassifier:
        """Train evidence classifier."""
        await self._dataset.prepare()
        df = self._dataset.get_evidence_split(
            SubsetName.TRAINING,
            max_samples=self._max_samples,
        )
        X = self._create_features(
            df["claim"].astype(str).tolist(),
            df["evidence"].astype(str).tolist(),
        )
        y = df["is_evidence"].astype("int32")
        cfg = self._config.training

        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            tree_method=cfg.tree_method,
            objective=cfg.objective,
            eval_metric=cfg.eval_metric,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        model.fit(
            X,
            y,
        )
        self._model = model
        with self.PATH_TO_MODEL.open("wb") as file:
            pickle.dump(
                model,
                file,
            )
        return model

    @override
    async def extract_evidence(
        self,
        claim: str,
        text: str,
    ) -> list[str]:
        """
        Extract evidence sentences supporting a claim.

        The document is split into candidate sentences. Each candidate is
        classified by the trained model and returned if it exceeds the selected
        confidence threshold.

        Args:
            claim (str):
                Claim for which supporting evidence should be found.

            text (str):
                Full document containing candidate evidence sentences.

        Returns:
            list[str]:
                Sentences classified as supporting evidence.
        """
        await self._initialise_model()
        if self._model is None:
            raise RuntimeError("Model was not initialized.")

        def normalize(sentence: str) -> str:
            return " ".join(sentence.lower().split())

        normalized_claim = normalize(claim)
        candidates = [
            sentence.strip()
            for sentence in self._sentence_splitter(text)
            if sentence.strip()
        ]

        # Remove claims from evidence candidates.
        candidates = [
            sentence
            for sentence in candidates
            if normalize(sentence) != normalized_claim
        ]
        if not candidates:
            return []
        X = self._create_features(
            [claim] * len(candidates),
            candidates,
        )
        probabilities = self._model.predict_proba(X)[:, 1]
        if self._threshold is None:
            await self.perform_tuning()
        threshold = self._config.threshold.claim
        return [
            evidence
            for evidence, probability in zip(
                candidates,
                probabilities,
                strict=True,
            )
            if (probability >= threshold and normalize(evidence) != normalized_claim)
        ]

    @override
    async def extract_claims(
        self,
        text: str,
    ) -> list[str]:
        raise NotImplementedError(
            "Evidence extractor does not support claim extraction.",
        )

    @override
    async def perform_tuning(
        self,
    ) -> dict[str, float | int]:
        """Tune evidence classifier."""
        await self._dataset.prepare()

        df = self._dataset.get_evidence_split(
            SubsetName.TRAINING,
            max_samples=self._max_samples,
        )

        X = self._create_features(
            df["claim"].astype(str).tolist(),
            df["evidence"].astype(str).tolist(),
        )

        y = df["is_evidence"].astype("int32").to_numpy()

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
