"""Module with an encoder implemented as Multi-Layer Perceptron (MLP)."""

import ast
import pickle
import time
from pathlib import Path
from typing import NamedTuple, cast, override

import numpy as np
import optuna
import pandas as pd
import torch
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.argument_encoding.argument_encoder import ArgumentEncoder
from src.argument_encoding.argument_encoding_dataset import ArgumentEncodingDataset
from src.configuration import config as global_config
from src.data_models.data_models import (
    InterpretativeFrame,
    StyledFramedArgument,
    SubsetName,
)
from src.utils.errors import NotPreparedError

# --- Configuration section


class TrainingConfig(BaseModel):
    """Hyperparameters for training a neural network."""

    epochs: int = Field(default=100, gt=0, description="A maximum number of epochs.")
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    patience: int = Field(default=10, gt=0)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MLPConfig(BaseModel):
    """Hyperparameters defining the architecture of a neural network."""

    layers: tuple[int, ...]
    dropout_rate: float = Field(0.3, ge=0.0, le=1.0)

    @field_validator("layers", mode="after")
    @classmethod
    def validate_layers(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """
        Validate that each layers is expected to have a positive number of neurons.

        Args:
            value (tuple[int, ...]): Each value represents a layers, first - an input
                layer, last - an output layer. Intermediate are hidden layers.
                Each integers represent a number of neurons per layers.

        Returns:
            tuple[int, ...]: A validated tuple of a number of neurons per layer.

        Raises:
            ValueError: A non-positive number of neurons in a layer of a neural network
                was provided.
        """
        for integer in value:
            if integer <= 0:
                raise ValueError(
                    "A number of neurons for each layer must be strictly positive. "
                    f"Provided: {value}"
                )
        return value


# --- A neural network


class MLP(torch.nn.Module):
    """A Multi-Layer Preceptron with dense layers decreasing in dimensionality."""

    def __init__(self, config: MLPConfig) -> None:
        """
        Initialise an MLP according to the configuration.

        Args:
            config (MLPConfig): Hyperparameters of architecture of the neural network.
        """
        super().__init__()

        layers: list[torch.nn.Module] = []
        for in_features, out_features in zip(
            config.layers[:-1], config.layers[1:], strict=True
        ):
            layers.extend(
                self._create_block(
                    in_features, out_features, dropout_rate=config.dropout_rate
                )
            )
        self._layers = torch.nn.Sequential(*layers)

    def _create_block(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float,
    ) -> tuple[torch.nn.Module, ...]:
        return (
            torch.nn.Linear(in_features=in_features, out_features=out_features),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.BatchNorm1d(out_features),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Pass the data through the neural network.

        Args:
            input_tensor (torch.Tensor): Tensor being an output of a neural network.
                Representation of a styled argument.

        Returns:
            torch.Tensor: Description.
        """
        return self._layers(input_tensor)


def _argument_to_input_tensor(argument: StyledFramedArgument) -> torch.Tensor:
    """
    Convert a StyledFramedArgument to a fixed-size input tensor.

    Features:
        - style_vector: dynamic length from SpacyStyleExtractor
        - frame one-hot: len(InterpretativeFrame) dimensions
        - frame probabilities: len(InterpretativeFrame) dimensions
    """
    style = torch.tensor(argument.style_vector, dtype=torch.float32).reshape(-1)

    frame_one_hot = torch.zeros(len(InterpretativeFrame), dtype=torch.float32)
    frame_index = list(InterpretativeFrame).index(argument.primary_frame)
    frame_one_hot[frame_index] = 1.0

    frame_probs = torch.tensor(
        [argument.frame_probabilities.get(frame, 0.0) for frame in InterpretativeFrame],
        dtype=torch.float32,
    )

    return torch.cat([style, frame_one_hot, frame_probs])


def _row_to_input_tensor(row: dict, suffix: str) -> torch.Tensor:
    """Convert a DataFrame row (one argument's columns) to an input tensor."""
    first_vector = torch.tensor(
        row[f"style_vector_{suffix}"], dtype=torch.float32
    ).reshape(-1)

    frame_name = row[f"frame_{suffix}"]
    frame_one_hot = torch.zeros(len(InterpretativeFrame), dtype=torch.float32)
    try:
        frame_index = list(InterpretativeFrame).index(InterpretativeFrame(frame_name))
    except ValueError:
        frame_index = 0
    frame_one_hot[frame_index] = 1.0

    probs_dict = row.get(f"frame_probabilities_{suffix}", {})
    if isinstance(probs_dict, str):
        probs_dict = ast.literal_eval(probs_dict)
    elif probs_dict is None or (isinstance(probs_dict, float) and pd.isna(probs_dict)):
        probs_dict = {}
    frame_probs = torch.tensor(
        [float(probs_dict.get(frame, 0.0)) for frame in InterpretativeFrame],
        dtype=torch.float32,
    )

    return torch.cat([first_vector, frame_one_hot, frame_probs])


def _cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Calculate cosine distances between two batches of vectors."""
    first_norm = torch.nn.functional.normalize(first, p=2, dim=1)
    second_norm = torch.nn.functional.normalize(second, p=2, dim=1)
    cosine_sim = torch.sum(first_norm * second_norm, dim=1)
    return 1.0 - cosine_sim


def _train_epoch(
    mlp: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    mlp.train()
    losses: list[float] = []
    for batch_first, batch_second, batch_distance in train_loader:
        first_input = batch_first.to(device)
        second_input = batch_second.to(device)
        expected_distance = batch_distance.to(device)
        optimizer.zero_grad()
        distances = _cosine_distance(mlp(first_input), mlp(second_input))
        loss = criterion(distances, expected_distance.squeeze(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def _validate_epoch(
    mlp: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Run one validation epoch and return average loss."""
    mlp.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_first, batch_second, batch_distance in val_loader:
            first_input = batch_first.to(device)
            second_input = batch_second.to(device)
            expected_distance = batch_distance.to(device)
            distances = _cosine_distance(mlp(first_input), mlp(second_input))
            loss = criterion(distances, expected_distance.squeeze(-1))
            losses.append(loss.item())
    return sum(losses) / len(losses)


class DataLoaders(NamedTuple):
    """Tuple with loader used for a neural network training."""

    train: DataLoader
    validation: DataLoader


def train_mlp(
    mlp: torch.nn.Module,
    train: TensorDataset,
    validation: TensorDataset,
    *,
    config: TrainingConfig | None = None,
) -> torch.nn.Module:
    """
    Train an MLP encoder with a contrastive distance loss.

    Each TensorDataset contains (input_tensor_1, input_tensor_2, expected_distance).
    The loss is MSE between cosine distance of encoded vectors and expected_distance.
    """
    config = config or TrainingConfig()
    device = torch.device(config.device)
    mlp.to(device)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    loaders = DataLoaders(
        train=DataLoader(train, batch_size=config.batch_size, shuffle=True),
        validation=DataLoader(validation, batch_size=config.batch_size, shuffle=False),
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict[str, torch.Tensor] | None = None

    with tqdm(range(config.epochs), desc="Training epochs", leave=False) as epoch_bar:
        for epoch in epoch_bar:
            avg_train_loss = _train_epoch(
                mlp, loaders.train, optimizer, criterion, device
            )
            avg_val_loss = _validate_epoch(mlp, loaders.validation, criterion, device)

            epoch_bar.set_postfix(
                train_loss=f"{avg_train_loss:.4f}",
                val_loss=f"{avg_val_loss:.4f}",
            )
            logger.debug(
                f"Epoch {epoch + 1}/{config.epochs}: "
                f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in mlp.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break

    if best_state is not None:
        mlp.load_state_dict(best_state)

    return mlp


def mlp_encoder_filename(input_dim: int, output_dim: int) -> str:
    """
    Build the filename for an encoder saved with input/output dimensionality.

    Args:
        input_dim (int): A number of features.
        output_dim (int): A number of outputs.

    Returns:
        str: Filename for a model of given dimensionality.
    """
    return f"mlp_encoder_in{input_dim}_out{output_dim}.pkl"


# --- Argument encoder wrapper


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
            t0 = time.time()
            mlp, _config = await self._train_best_model(input_dimensions=dims)
            logger.info(f"[PERF] _train_best_model: {time.time() - t0:.1f}s")
        else:
            mlp, _config = loaded_model
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
        mlp.eval()
        return mlp, config

    def _save_model(self, mlp: torch.nn.Module, config: MLPConfig) -> None:
        model_path = self._model_path
        if model_path is None:
            raise RuntimeError("Cannot save model before input dimensions are known.")
        with model_path.open("wb") as file:
            pickle.dump({"config": config, "state_dict": mlp.state_dict()}, file)

    async def _train_best_model(
        self, input_dimensions: int
    ) -> tuple[torch.nn.Module, MLPConfig]:
        t0 = time.time()
        await self._dataset.prepare()
        logger.info(f"[PERF] dataset.prepare: {time.time() - t0:.1f}s")
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
            hyperparameters = MLPConfig(
                layers=(
                    input_dimensions,
                    layer_1,
                    layer_2,
                    layer_3,
                    self._output_dimensions,
                ),
                dropout_rate=trial.suggest_float("dropout_rate", 0.0, 0.95),
            )
            mlp = MLP(config=hyperparameters)
            trial_t0 = time.time()
            mlp = cast(
                "MLP",
                train_mlp(
                    mlp, train_dataset, validation_dataset, config=self._training_config
                ),
            )
            logger.debug(
                f"[PERF] Optuna trial {trial.number} train_mlp: "
                f"{time.time() - trial_t0:.1f}s"
            )
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
                    distances = _cosine_distance(first_vector, second_vector)
                    loss = torch.nn.functional.mse_loss(
                        distances, batch_distance.squeeze(-1)
                    )
                    val_losses.append(loss.item())
            return sum(val_losses) / len(val_losses)

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
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10, interval_steps=1
            ),
        )
        study.optimize(
            objective, n_trials=self._n_optuna_trials, callbacks=[_update_trial_bar]
        )
        trial_bar.close()

        best_config = MLPConfig(
            layers=(
                input_dimensions,
                study.best_params["neurons_in_layer_1"],
                study.best_params["neurons_in_layer_2"],
                study.best_params["neurons_in_layer_3"],
                self._output_dimensions,
            ),
            dropout_rate=study.best_params["dropout_rate"],
        )
        best_mlp = MLP(config=best_config)
        fine_tune_t0 = time.time()
        best_mlp = cast(
            "MLP",
            train_mlp(
                best_mlp,
                train_dataset,
                validation_dataset,
                config=self._fine_tuning_config,
            ),
        )
        logger.info(f"[PERF] fine-tuning train_mlp: {time.time() - fine_tune_t0:.1f}s")
        self._save_model(best_mlp, best_config)
        return best_mlp, best_config

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

        for _, row in df.iterrows():
            first_vector = _row_to_input_tensor(row.to_dict(), "1")
            second_vector = _row_to_input_tensor(row.to_dict(), "2")
            distance = torch.tensor([row["expected_distance"]], dtype=torch.float32)
            first_inputs_list.append(first_vector)
            second_inputs_list.append(second_vector)
            expected_distances_list.append(distance)

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
        self._model.eval()
        device = next(self._model.parameters()).device

        results = []
        with torch.no_grad():
            for argument in arguments:
                input_tensor = (
                    _argument_to_input_tensor(argument).unsqueeze(0).to(device)
                )
                encoded_vector = self._model(input_tensor)
                results.append(encoded_vector.squeeze(0).cpu().numpy())
        return results

    @override
    def clear(self) -> None:
        if self._model_path is not None:
            self._model_path.unlink(missing_ok=True)


__all__ = ["MLPConfig", "MLPEncoder", "TrainingConfig"]
