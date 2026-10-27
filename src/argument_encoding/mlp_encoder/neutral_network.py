"""Module with an MLP implementation of the argument encoder."""

from typing import NamedTuple

import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.argument_encoding.mlp_encoder.config import MLPConfig, TrainingConfig
from src.argument_encoding.mlp_encoder.utils import cosine_distance


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
    if config.random_seed is not None:
        torch.manual_seed(config.random_seed)

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
