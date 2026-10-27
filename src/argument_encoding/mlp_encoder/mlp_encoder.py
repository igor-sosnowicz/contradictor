"""Module with an encoder wrapper around Multi-Layer Perceptron (MLP)."""

import pickle
from pathlib import Path
from typing import cast, override

import numpy as np
import optuna
import pandas as pd
import torch
from loguru import logger
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.argument_encoding.argument_encoder import ArgumentEncoder
from src.argument_encoding.argument_encoding_dataset import ArgumentEncodingDataset
from src.argument_encoding.mlp_encoder.config import MLPConfig, TrainingConfig
from src.argument_encoding.mlp_encoder.neutral_network import (
    MLP,
    train_mlp,
)
from src.argument_encoding.mlp_encoder.utils import (
    cosine_distance,
    mlp_encoder_filename,
    row_to_input_tensor,
)
from src.configuration import config as global_config
from src.data_models.data_models import (
    InterpretativeFrame,
    StyledFramedArgument,
    SubsetName,
)
from src.utils.errors import NotPreparedError
from src.utils.timer import Timer


class MLPEncoder(ArgumentEncoder):
    """Implementation of an argument encoder as a Multi-Layer Perceptron."""

    MODEL_DIR: Path = global_config.data_directory / global_config.model_subdirectory

    def __init__(
        self,
        dataset: ArgumentEncodingDataset,
        output_dimensions: int,
        fine_tuning_config: TrainingConfig | None = None,
        training_config: TrainingConfig | None = None,
        n_optuna_trials: int = 20,
    ) -> None:
        """
        Initialise basic configuration of the encoder.

        Args:
            dataset (ArgumentEncodingDataset): Dataset used to train & validate the
                encoder on if missing.
            output_dimensions (int): A number of features the output of the encoder
                is supposed to have.
            fine_tuning_config (TrainingConfig | None): Configuration of training a
                MLP during phase of adjusting its hyperparameters. If None, a default
                configuration is used.
            training_config (TrainingConfig | None): Configuration of training a
                MLP during the final training. If None, a default configuration is used.
            n_optuna_trials (int): Number of Optuna hyperparameter search trials
                executed when no cached model exists. Lower this (e.g. to 1-2) in
                tests to keep them fast.

        Raises:
            ValueError: Raised if a non-positive number of output dimensions was given.
        """
        if output_dimensions < 1:
            raise ValueError(
                f"A number of output dimensions of {MLPEncoder.__name__} has to be "
                "greater than zero."
            )
        self._output_dimensions = output_dimensions
        self._input_dimensions: int | None = None
        self.MODEL_DIR.mkdir(exist_ok=True, parents=True)
        self._model: torch.nn.Module | None = None
        self._dataset = dataset
        self._training_config = training_config or TrainingConfig(
            epochs=50,
            batch_size=32,
            learning_rate=1e-3,
            patience=5,
        )
        self._fine_tuning_config = fine_tuning_config or TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            patience=10,
        )
        self._n_optuna_trials = n_optuna_trials

    @property
    def _model_path(self) -> Path | None:
        """Path to the saved encoder for the current input/output dimensionality."""
        if self._input_dimensions is None:
            return None
        return self.MODEL_DIR / mlp_encoder_filename(
            self._input_dimensions, self._output_dimensions
        )

    @override
    async def prepare(self) -> None:
        if self._model is not None:
            logger.debug("The model is cached and will not be retrained.")
            return
        dims = await self._compute_input_dimensions()
        logger.debug(f"Detected {dims} features as the encoder input.")
        loaded_model = self._load_model()
        if loaded_model is None:
            logger.info("The model will be trained now. It may take a long time.")
            with Timer("_train_best_model"):
                mlp, _ = await self._train_best_model(input_dimensions=dims)
        else:
            mlp, _ = loaded_model
        self._model = mlp

    def _load_model(self) -> tuple[torch.nn.Module, MLPConfig] | None:
        model_path = self._model_path
        if model_path is None or not model_path.exists() or not model_path.is_file():
            logger.info(
                f"There is no trained model for {self._input_dimensions} features and "
                f"{self._output_dimensions} outputs."
            )
            return None
        with model_path.open("rb") as file:
            saved = pickle.load(file)  # noqa: S301, Self-generated model file is safe.
        config = saved["config"]
        # Retrain if layer dimensions do not match target dimensionality.
        if (
            config.layers[0] != self._input_dimensions
            or config.layers[-1] != self._output_dimensions
        ):
            logger.warning(
                "A matching model was found, but after loading into memory, it turned "
                "out to be incompatible."
            )
            logger.debug(
                f"Expected a model with {self._input_dimensions} features and "
                f"{self._output_dimensions} outputs. Got {config.layers[0]} "
                f"features and {config.layers[-1]} outputs."
            )
            logger.warning("This model file is likely corrupted and will be deleted.")
            model_path.unlink()
            return None
        state_dict = saved["state_dict"]
        mlp = MLP(config=config)
        mlp.load_state_dict(state_dict)
        mlp.to(torch.device(global_config.device))
        mlp.eval()
        return mlp, config

    def _save_model(self, mlp: torch.nn.Module, config: MLPConfig) -> None:
        model_path = self._model_path
        if model_path is None:
            raise RuntimeError("Cannot save model before input dimensions are known.")
        state_dict = {
            key: value.detach().cpu() for key, value in mlp.state_dict().items()
        }
        with model_path.open("wb") as file:
            pickle.dump({"config": config, "state_dict": state_dict}, file)

    def _build_config(
        self,
        params: dict[str, int | float],
        input_dimensions: int,
        output_dimensions: int,
    ) -> MLPConfig:
        return MLPConfig(
            layers=(
                input_dimensions,
                int(params["neurons_in_layer_1"]),
                int(params["neurons_in_layer_2"]),
                int(params["neurons_in_layer_3"]),
                output_dimensions,
            ),
            dropout_rate=float(params["dropout_rate"]),
        )

    async def _train_best_model(
        self, input_dimensions: int
    ) -> tuple[torch.nn.Module, MLPConfig]:
        with Timer("dataset.prepare"):
            await self._dataset.prepare()
        train_df = self._dataset.get_split(SubsetName.TRAINING)
        validation_df = self._dataset.get_split(SubsetName.VALIDATION)

        train_tensors = self._dataframe_to_tensors(train_df)
        validation_tensors = self._dataframe_to_tensors(validation_df)

        train_dataset = TensorDataset(*train_tensors)
        validation_dataset = TensorDataset(*validation_tensors)

        def objective(trial: optuna.trial.Trial) -> float:
            layer_1 = trial.suggest_int("neurons_in_layer_1", 32, 4 * input_dimensions)
            layer_2 = trial.suggest_int("neurons_in_layer_2", 16, 4 * layer_1)
            layer_3 = trial.suggest_int("neurons_in_layer_3", 8, 2 * layer_2)
            params = {
                "neurons_in_layer_1": layer_1,
                "neurons_in_layer_2": layer_2,
                "neurons_in_layer_3": layer_3,
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.95),
            }
            hyperparameters = self._build_config(
                params, input_dimensions, self._output_dimensions
            )
            mlp = MLP(config=hyperparameters)
            with Timer(f"Optuna trial {trial.number} train_mlp"):
                mlp = cast(
                    "MLP",
                    train_mlp(
                        mlp,
                        train_dataset,
                        validation_dataset,
                        config=self._training_config,
                    ),
                )
            return self._validation_loss(mlp, validation_dataset)

        logger.info(
            f"Starting Optuna hyperparameter search with {self._n_optuna_trials} trials"
            f" (this may take a long time on CPU)."
        )
        trial_bar = tqdm(total=self._n_optuna_trials, desc="Optuna trials", leave=False)

        def _update_trial_bar(
            _study: optuna.Study, _trial: optuna.trial.FrozenTrial
        ) -> None:
            trial_bar.n = len(_study.trials)
            trial_bar.refresh()

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self._training_config.random_seed),
        )
        study.optimize(
            objective, n_trials=self._n_optuna_trials, callbacks=[_update_trial_bar]
        )
        trial_bar.close()

        best_config = self._build_config(
            study.best_params, input_dimensions, self._output_dimensions
        )
        best_mlp = MLP(config=best_config)
        with Timer("fine-tuning train_mlp"):
            best_mlp = cast(
                "MLP",
                train_mlp(
                    best_mlp,
                    train_dataset,
                    validation_dataset,
                    config=self._fine_tuning_config,
                ),
            )
        self._save_model(best_mlp, best_config)
        return best_mlp, best_config

    def _validation_loss(
        self, mlp: torch.nn.Module, validation_dataset: TensorDataset
    ) -> float:
        """
        Compute the average validation loss of a model.

        Args:
            mlp (torch.nn.Module): The model to evaluate.
            validation_dataset (TensorDataset): The validation dataset.

        Returns:
            float: The mean mean-squared error between predicted and expected
                cosine distances.
        """
        val_loader = DataLoader(
            validation_dataset,
            batch_size=self._training_config.batch_size,
            shuffle=False,
        )
        mlp.eval()
        val_losses = []
        with torch.no_grad():
            for batch_first, batch_second, batch_distance in val_loader:
                first_vector = mlp(batch_first)
                second_vector = mlp(batch_second)
                distances = cosine_distance(first_vector, second_vector)
                loss = torch.nn.functional.mse_loss(
                    distances, batch_distance.squeeze(-1)
                )
                val_losses.append(loss.item())
        return sum(val_losses) / len(val_losses)

    async def _compute_input_dimensions(self) -> int:
        """
        Compute input dimensions dynamically from the dataset's style vectors.

        Style vector length varies, so we derive it from the actual data
        rather than hardcoding it.
        """
        if self._input_dimensions is not None:
            return self._input_dimensions

        await self._dataset.prepare()
        train_df = self._dataset.get_split(SubsetName.TRAINING)
        style_length = len(train_df.iloc[0]["style_vector_1"])

        self._input_dimensions = (
            style_length
            + len(InterpretativeFrame)  # frame one-hot
            + len(InterpretativeFrame)  # frame probabilities
        )
        return self._input_dimensions

    def _dataframe_to_tensors(
        self, df: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        first_inputs_list: list[torch.Tensor] = []
        second_inputs_list: list[torch.Tensor] = []
        expected_distances_list: list[torch.Tensor] = []

        skipped = 0
        for _, row in df.iterrows():
            try:
                first_vector = row_to_input_tensor(row.to_dict(), "1")
                second_vector = row_to_input_tensor(row.to_dict(), "2")
            except ValueError as exc:
                skipped += 1
                logger.warning("Skipping row with invalid frame: {}", exc)
                continue
            distance = torch.tensor([row["expected_distance"]], dtype=torch.float32)
            first_inputs_list.append(first_vector)
            second_inputs_list.append(second_vector)
            expected_distances_list.append(distance)

        if skipped:
            logger.info("Skipped {} row(s) with invalid frames.", skipped)
        if not first_inputs_list:
            raise ValueError("No valid rows left after skipping invalid frames.")

        first_inputs = torch.stack(first_inputs_list)
        second_inputs = torch.stack(second_inputs_list)
        expected_distances = torch.stack(expected_distances_list)
        return first_inputs, second_inputs, expected_distances

    @override
    async def __call__(self, arguments: list[StyledFramedArgument]) -> list[np.ndarray]:
        if self._model is None:
            raise NotPreparedError(
                "Cannot infer with the unprepared model. Call `prepare()` first."
            )
        if not arguments:
            return []

        self._model.eval()
        device = next(self._model.parameters()).device
        batch_size = self._training_config.batch_size

        results: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(arguments), batch_size):
                chunk = arguments[start : start + batch_size]
                input_batch = torch.stack(
                    [argument.to_tensor() for argument in chunk]
                ).to(device)
                encoded_batch = self._model(input_batch)
                results.extend(row.cpu().numpy() for row in encoded_batch)

        return results

    @override
    def reset(self) -> None:
        self._model = None
        self._input_dimensions = None
        if self._model_path is not None:
            self._model_path.unlink(missing_ok=True)


__all__ = ["MLPConfig", "MLPEncoder", "TrainingConfig"]
