"""Module with XGBoost for interpretative frame classification."""

import pickle
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Final, override

import optuna
from optuna.samplers import TPESampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from xgboost import XGBClassifier

from src.argument_framing.argument_framer import ArgumentFramer
from src.argument_framing.argument_framing_dataset import ArgumentFramingDataset
from src.argument_framing.configuration import XGBoostConfig
from src.configuration import config
from src.data_models.data_models import (
    LABEL_TO_FRAME,
    Argument,
    FramedArgument,
    SubsetName,
)
from src.utils.errors import ModelNotTrainedError, NotPreparedError


class XGBoostFrameClassifier(ArgumentFramer):
    """XGBoost classifier assigning interpretative frames to arguments."""

    PATH_TO_MODEL: Final = Path(
        config.data_directory
        / config.model_subdirectory
        / "xgboost_frame_classifier.pkl"
    )

    PATH_TO_VECTORIZER: Final = Path(
        config.data_directory / config.model_subdirectory / "tfidf_vectorizer.pkl"
    )

    NUM_CLASSES: Final = 15

    def __init__(
        self,
        dataset: ArgumentFramingDataset,
        *,
        proof_of_concept_mode: bool = False,
        configuration: XGBoostConfig | None = None,
    ) -> None:
        """
        Initialise prerequisites for frame classifier.

        Args:
            dataset (ArgumentFramingDataset): Preprocessed dataset containing arguments
                paired with their interpretative frame labels.
            proof_of_concept_mode (bool, optional): Whether the classifier should be
            used in a proof of concept mode where a tiny number of examples is used.
                Defaults to False (all samples are used).
            configuration (XGBoostConfig | None): Internal configuration of the XGBoost
                and TF-IDF vectoriser models. None means default values will be used.
        """
        self.PATH_TO_MODEL.parent.mkdir(parents=True, exist_ok=True)

        self._model = self._load()
        self._vectorizer = self._load_vectorizer()
        self._dataset = dataset
        self.config = configuration or XGBoostConfig()

        self._max_samples: Final = 1000 if proof_of_concept_mode else 0

    @override
    def _load(self) -> XGBClassifier | None:
        if not self.PATH_TO_MODEL.exists():
            return None

        with self.PATH_TO_MODEL.open("rb") as file:
            return pickle.load(file)  # noqa: S301, Trusted source.

    def _load_vectorizer(self) -> TfidfVectorizer | None:
        if not self.PATH_TO_VECTORIZER.exists():
            return None

        with self.PATH_TO_VECTORIZER.open("rb") as file:
            return pickle.load(file)  # noqa: S301, Trusted source.

    @override
    async def _train(
        self,
    ) -> XGBClassifier:

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=self.NUM_CLASSES,
            eval_metric="mlogloss",
            n_estimators=self.config.xgb_n_estimators,
            n_jobs=self.config.xgb_n_jobs,
        )
        vectorizer = TfidfVectorizer(
            max_features=self.config.tf_idf_max_features,
            ngram_range=(
                self.config.tf_idf_shortest_n_gram,
                self.config.tf_idf_longest_n_gram,
            ),
            lowercase=True,
        )

        await self._dataset.prepare()
        df = self._dataset.get_split(SubsetName.TRAINING, max_samples=self.max_samples)

        tqdm.pandas(desc="Creating TF-IDF features")
        x = vectorizer.fit_transform(df["text"])
        y = df["interpretative_frame"].astype("int32")

        # validation split
        validation_df = self._dataset.get_split(SubsetName.VALIDATION)
        x_val = vectorizer.transform(validation_df["text"])
        y_val = validation_df["interpretative_frame"].astype("int32")

        model.fit(
            x,
            y,
            eval_set=[(x_val, y_val)],
            verbose=True,
        )

        with self.PATH_TO_MODEL.open("wb") as file:
            pickle.dump(model, file)

        with self.PATH_TO_VECTORIZER.open("wb") as file:
            pickle.dump(vectorizer, file)

        self._vectorizer = vectorizer
        return model

    @override
    async def prepare(self) -> None:
        if self._model is None:
            self._model = await self._train()

    @override
    def frame(
        self,
        arguments: Iterable[Argument],
    ) -> Collection[FramedArgument]:
        if self._model is None:
            raise NotPreparedError("Cannot run the unprepared model.")
        if self._vectorizer is None:
            raise ModelNotTrainedError(
                "Vectorizer is not initialized. Train the model first."
            )

        arguments = list(arguments)
        texts = [str(argument) for argument in arguments]
        probabilities = self._model.predict_proba(self._vectorizer.transform(texts))

        framed_arguments: list[FramedArgument] = []
        for argument, row_probabilities in zip(arguments, probabilities, strict=True):
            frame_probabilities = {
                LABEL_TO_FRAME[label]: float(probability)
                for label, probability in enumerate(row_probabilities)
            }
            framed_arguments.append(
                FramedArgument(
                    claim=argument.claim,
                    evidence=argument.evidence,
                    primary_frame=LABEL_TO_FRAME[int(row_probabilities.argmax())],
                    frame_probabilities=frame_probabilities,
                )
            )

        return framed_arguments

    @property
    def max_samples(self) -> int:
        """Get the maximum number of samples allowed for training."""
        return self._max_samples

    async def perform_tuning(
        self,
    ) -> dict[str, int | float]:
        """Tune XGBoost hyperparameters using Optuna."""
        await self._dataset.prepare()

        if self._vectorizer is None:
            raise ModelNotTrainedError("Vectorizer must be trained before tuning.")

        df = self._dataset.get_split(
            SubsetName.TRAINING,
            max_samples=self._max_samples,
        )

        x = self._vectorizer.transform(df["text"])
        y = df["interpretative_frame"].astype("int32")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "multi:softprob",
                "num_class": self.NUM_CLASSES,
                "eval_metric": "mlogloss",
                "random_state": self.config.random_seed,
                "n_jobs": self.config.xgb_n_jobs,
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    10,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.3,
                    log=True,
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    100,
                    1000,
                    step=100,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    0.6,
                    1.0,
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.5,
                    1.0,
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    1,
                    10,
                ),
                "gamma": trial.suggest_float(
                    "gamma",
                    0.0,
                    5.0,
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    0.0,
                    5.0,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    0.0,
                    5.0,
                ),
            }

            model = XGBClassifier(**params)

            scores = cross_val_score(
                model,
                x,
                y,
                cv=self.config.cross_validation_folds,
                scoring=self.config.cross_validation_metric,
                n_jobs=self.config.xgb_n_jobs,
            )

            return float(scores.mean())

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.config.random_seed),
        )

        study.optimize(
            objective,
            n_trials=self.config.optuna_trials,
        )

        best_params = study.best_params

        self._model = XGBClassifier(
            objective="multi:softprob",
            num_class=self.NUM_CLASSES,
            eval_metric="mlogloss",
            random_state=self.config.random_seed,
            n_jobs=self.config.xgb_n_jobs,
            **best_params,
        )

        self._model.fit(x, y)

        with self.PATH_TO_MODEL.open("wb") as file:
            pickle.dump(self._model, file)

        return {
            "best_score": float(study.best_value),
            **best_params,
        }
