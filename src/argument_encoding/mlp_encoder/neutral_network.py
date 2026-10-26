"""Module with an MLP implementation of the argument encoder."""

from typing import NamedTuple

import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.argument_encoding.mlp_encoder.config import MLPConfig, TrainingConfig
from src.argument_encoding.mlp_encoder.utils import cosine_distance
from src.utils.errors import DatasetError


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
        for i, (in_features, out_features) in enumerate(
            zip(config.layers[:-1], config.layers[1:], strict=True)
        ):
            is_last_layer = i == len(config.layers) - 2

            if is_last_layer:
                layers.extend(
                    self._create_block(
                        in_features,
                        out_features,
                        dropout_rate=config.dropout_rate,
                        is_final=True,
                    )
                )
            else:
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
        *,
        is_final: bool = False,
    ) -> tuple[torch.nn.Module, ...]:
        if not is_final:
            return (
                torch.nn.Linear(in_features=in_features, out_features=out_features),
                torch.nn.BatchNorm1d(out_features),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=dropout_rate),
            )
        return (
            torch.nn.Linear(in_features=in_features, out_features=out_features),
            torch.nn.Sigmoid(),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Pass the data through the neural network.

        Args:
            input_tensor (torch.Tensor): Tensor being an input for
                a neural network.

        Returns:
            torch.Tensor: Tensor being an output of a neural network.
                Representation of a styled argument.
        """
        return self._layers(input_tensor)


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
        distances = cosine_distance(mlp(first_input), mlp(second_input))
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
            distances = cosine_distance(mlp(first_input), mlp(second_input))
            loss = criterion(distances, expected_distance.squeeze(-1))
            losses.append(loss.item())
    return sum(losses) / len(losses)


class DataLoaders(NamedTuple):
    """Tuple with loader used for a neural network training."""

    train: DataLoader
    validation: DataLoader


class TrainResult(NamedTuple):
    """Result of training an MLP encoder."""

    model: MLP
    best_validation_loss: float
    best_epoch: int
    epochs_run: int


def _run_training_loop(
    mlp: MLP,
    loaders: DataLoaders,
    config: TrainingConfig,
) -> tuple[float, int, int, dict[str, torch.Tensor] | None]:
    """
    Run training epochs with early stopping and track the best validation state.

    Args:
        mlp (MLP): The model being trained.
        loaders (DataLoaders): Training and validation data loaders.
        config (TrainingConfig): The training configuration.

    Returns:
        A tuple of (best validation loss, best epoch, epochs run, best state dict).
    """
    device = torch.device(config.device)
    mlp.to(device)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_run = 0

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
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in mlp.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    epochs_run = epoch + 1
                    break
        else:
            epochs_run = config.epochs

    return best_val_loss, best_epoch, epochs_run, best_state


def train_mlp(
    mlp: MLP,
    train: TensorDataset,
    validation: TensorDataset,
    *,
    config: TrainingConfig | None = None,
) -> TrainResult:
    """
    Train an MLP encoder with a contrastive distance loss.

    The loss is MSE between cosine distance of encoded vectors and `expected_distance`.

    Args:
        mlp (MLP): An untrained instance of an MLP.
        train (TensorDataset): A training split of a TensorDataset containing the
            following items:(input_tensor_1, input_tensor_2, expected_distance).
        validation (TensorDataset): A validation split of a TensorDataset containing
            the following items:(input_tensor_1, input_tensor_2, expected_distance).
        config (TrainingConfig | None): Configuration regarding the training process
            of a MLP. It uses pre-defined values if `None` passed. Defaults to None.

    Returns:
        TrainResult: A named tuple with the trained MLP encoder, the best
            validation loss, the epoch at which it was achieved, and the number of
            epochs run.

    Raises:
        DatasetError: Raised if either a training or validation split are empty.
    """
    if len(train) == 0:
        raise DatasetError(
            "The dataset is too small resulting in an empty training split. "
            "Use a dataset with a larger number of examples."
        )
    if len(validation) == 0:
        raise DatasetError(
            "The dataset is too small resulting in an empty validation split. "
            "Use a dataset with a larger number of examples."
        )
    config = config or TrainingConfig()
    if config.random_seed is not None:
        torch.manual_seed(config.random_seed)

    loaders = DataLoaders(
        train=DataLoader(train, batch_size=config.batch_size, shuffle=True),
        validation=DataLoader(validation, batch_size=config.batch_size, shuffle=False),
    )

    best_val_loss, best_epoch, epochs_run, best_state = _run_training_loop(
        mlp, loaders, config
    )

    if best_state is not None:
        mlp.load_state_dict(best_state)

    return TrainResult(
        mlp,
        best_val_loss,
        max(0, best_epoch),
        epochs_run,
    )
