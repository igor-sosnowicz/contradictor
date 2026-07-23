"""Module with XGBoost for interpretative frame classification."""

import pickle
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
from src.configuration import config
from src.data_models.data_models import (
    LABEL_TO_FRAME,
    FramedArgument,
    SubsetName,
)
from src.utils.errors import ModelNotTrainedError


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
    ) -> None:
        """
        Initialise prerequisites for frame classifier.

        Args:
            dataset (ArgumentFramingDataset): Preprocessed dataset containing arguments
                paired with their interpretative frame labels.
            proof_of_concept_mode (bool, optional): Whether the classifier should be
            used in a proof of concept mode where a tiny number of examples is used.
                Defaults to False (all samples are used).
        """
        self.PATH_TO_MODEL.parent.mkdir(parents=True, exist_ok=True)

        self._model = self._load()
        self._vectorizer = self._load_vectorizer()
        self._dataset = dataset

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
        dataset: ArgumentFramingDataset,
    ) -> XGBClassifier:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=self.NUM_CLASSES,
            eval_metric="mlogloss",
            n_estimators=30,  # change to 150
            n_jobs=-1,
        )
        vectorizer = TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            lowercase=True,
        )

        await dataset.prepare()
        df = dataset.get_split(SubsetName.TRAINING, max_samples=self.max_samples)

        tqdm.pandas(desc="Creating TF-IDF features")
        x = vectorizer.fit_transform(df["text"])
        y = df["interpretative_frame"].astype("int32")

        # validation split
        validation_df = dataset.get_split(SubsetName.VALIDATION)
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
    async def frame(
        self,
        text: str,
    ) -> FramedArgument:
        if self._model is None:
            self._model = await self._train(self._dataset)

        if self._vectorizer is None:
            raise ModelNotTrainedError(
                "Vectorizer is not initialized. Train the model first."
            )

        x = self._vectorizer.transform([text])
        probabilities = self._model.predict_proba(x)[0]

        frame_probabilities = {
            LABEL_TO_FRAME[label]: float(probability)
            for label, probability in enumerate(probabilities)
        }
        primary_frame = max(
            frame_probabilities,
            key=lambda k: frame_probabilities[k],
        )

        return FramedArgument(
            text=text,
            primary_frame=primary_frame,
            frame_probabilities=frame_probabilities,
        )

    @property
    def max_samples(self) -> int:
        """Get the maximum number of samples allowed for training."""
        return self._max_samples

    async def perform_tuning(
        self,
        dataset: ArgumentFramingDataset,
    ) -> dict[str, int | float]:
        """Tune XGBoost hyperparameters using Optuna."""
        await dataset.prepare()

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
                "random_state": 42,
                "n_jobs": -1,
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
                cv=5,
                scoring="f1_macro",
                n_jobs=4,
            )

            return float(scores.mean())

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
        )

        study.optimize(
            objective,
            n_trials=50,
        )

        best_params = study.best_params

        self._model = XGBClassifier(
            objective="multi:softprob",
            num_class=self.NUM_CLASSES,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            **best_params,
        )

        self._model.fit(x, y)

        with self.PATH_TO_MODEL.open("wb") as file:
            pickle.dump(self._model, file)

        return {
            "best_score": float(study.best_value),
            **best_params,
        }
