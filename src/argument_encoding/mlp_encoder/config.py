"""Module with configuration data models for the MLP encoder."""

from pydantic import BaseModel, Field, field_validator

from src.configuration import config


class TrainingConfig(BaseModel):
    """Hyperparameters for training a neural network."""

    epochs: int = Field(default=100, gt=0, description="A maximum number of epochs.")
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    patience: int = Field(default=10, gt=0)
    device: str = config.device
    random_seed: int | None = None


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
        minimal_layer_count = 2
        if len(value) < minimal_layer_count:
            raise ValueError(
                f"An MLP has to have at least two layers: input & output, got: {value})"
            )
        for integer in value:
            if integer <= 0:
                raise ValueError(
                    "A number of neurons for each layer must be strictly positive. "
                    f"Provided: {value}"
                )
        return value
