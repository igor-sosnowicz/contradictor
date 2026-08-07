"""Base functionality for XGBoost extractors."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import optuna
from diskcache import Cache
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.argument_detection.argument_detection_dataset import (
    ArgumentDetectionDataset,
)
from src.argument_detection.spacy_embedder import SpacyEmbedder
from src.configuration import config
from src.utils.sentence_splitter import SentenceSplitter

if TYPE_CHECKING:
    from src.argument_detection.config import XGBoostExtractorConfig

ModelType = TypeVar(
    "ModelType",
)


# pylint: disable=too-many-instance-attributes
class BaseXGBoostExtractor[ModelType](ABC):
    """Base class for XGBoost-based extractors."""

    def __init__(
        self,
        dataset: ArgumentDetectionDataset,
        *,
        model_name: str,
        cache_name: str,
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
        self._config: XGBoostExtractorConfig = config.xgboost_extractor
        self._dataset = dataset
        self._sentence_splitter = SentenceSplitter()
        self._model: ModelType | None = None
        self._embedder = SpacyEmbedder()
        self._model_name = model_name

        self._get_model_path().parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._cache = Cache(config.cache_directory / cache_name)
        self._max_samples = self._get_max_samples() if proof_of_concept_mode else 0

    def _get_model_path(self) -> Path:
        """Generate path to the model file dynamically."""
        return config.data_directory / config.model_subdirectory / self._model_name

    @abstractmethod
    def _get_max_samples(
        self,
    ) -> int:
        """Return proof-of-concept sample limit."""

    @abstractmethod
    async def extract_claims(
        self,
        text: str,
    ) -> list[str]:
        """
        Extract claims from text.

        Args:
            text (str):
                Input document.

        Returns:
            list[str]:
                Extracted claims.
        """

    @abstractmethod
    async def extract_evidence(
        self,
        claim: str,
        text: str,
    ) -> list[str]:
        """
        Extract evidence sentences.

        Args:
            claim:
                Claim to find evidence for.

            text:
                Document text.

        Returns:
            List of extracted evidence sentences.
        """

    @abstractmethod
    async def perform_tuning(
        self,
    ) -> dict[str, float]:
        """
        Tune extractor parameters.

        Returns:
            Dictionary with tuning results.
        """

    @abstractmethod
    async def _train(
        self,
    ) -> ModelType:
        """Train model."""

    def _load(
        self,
    ) -> ModelType | None:
        """Load trained model."""
        model_path = self._get_model_path()
        if not model_path.exists():
            return None

        with model_path.open("rb") as file:
            return pickle.load(file)  # noqa: S301

    def _save_model(
        self,
        model: ModelType,
    ) -> None:
        """Persist trained model."""
        with self._get_model_path().open("wb") as file:
            pickle.dump(
                model,
                file,
            )

    def _fit_xgboost_model(
        self, x: np.ndarray, y: np.ndarray, best_params: dict
    ) -> XGBClassifier:
        """Shared method for initialization and training XGBoost model."""
        model = XGBClassifier(
            objective=self._config.training.objective,
            eval_metric=self._config.training.eval_metric,
            tree_method=self._config.training.tree_method,
            random_state=self._config.training.random_state,
            n_jobs=self._config.training.n_jobs,
            **best_params,
        )
        model.fit(x, y)
        return model

    async def _initialise_model(
        self,
        *,
        train_if_missing: bool = True,
    ) -> None:
        """Load existing model or train."""
        if self._model is not None:
            return

        self._model = self._load()

        if self._model is not None:
            return

        if not train_if_missing:
            raise RuntimeError("Model does not exist.")

        self._model = await self._train()

    def _find_best_threshold(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[float, float]:
        """Find threshold with best F1 score."""
        threshold_cfg = self._config.threshold

        best_threshold = threshold_cfg.default_threshold
        best_score = -1.0

        for threshold in np.linspace(
            threshold_cfg.min_threshold,
            threshold_cfg.max_threshold,
            threshold_cfg.num_thresholds,
        ):
            predictions = probabilities >= threshold

            score = f1_score(
                labels,
                predictions,
            )

            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        return (
            best_threshold,
            float(best_score),
        )

    def _save_threshold(
        self,
        threshold: float,
    ) -> None:
        """Save selected threshold."""
        self._cache.set(
            "threshold",
            threshold,
        )

    def _tune_hyperparameters(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float | int]:
        """Tune XGBoost hyperparameters using Optuna."""
        optuna_config = self._config.optuna
        training_config = self._config.training

        def objective(
            trial: optuna.Trial,
        ) -> float:
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    optuna_config.n_estimators_min,
                    optuna_config.n_estimators_max,
                    step=optuna_config.n_estimators_step,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    optuna_config.max_depth_min,
                    optuna_config.max_depth_max,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    optuna_config.learning_rate_min,
                    optuna_config.learning_rate_max,
                    log=True,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    optuna_config.subsample_min,
                    optuna_config.subsample_max,
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    optuna_config.colsample_bytree_min,
                    optuna_config.colsample_bytree_max,
                ),
            }

            model = XGBClassifier(
                objective=training_config.objective,
                eval_metric=training_config.eval_metric,
                tree_method=training_config.tree_method,
                random_state=training_config.random_state,
                n_jobs=training_config.n_jobs,
                **params,
            )

            scores = cross_val_score(
                model,
                x,
                y,
                cv=optuna_config.cv_folds,
                scoring="f1",
                n_jobs=optuna_config.n_jobs,
            )

            return float(scores.mean())

        optuna.logging.set_verbosity(
            optuna.logging.WARNING,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=optuna_config.seed),
        )
        study.optimize(
            objective,
            n_trials=optuna_config.n_trials,
        )

        return study.best_params
