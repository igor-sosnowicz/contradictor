"""Configuration data structures for the ModernBERT framer."""

import hashlib
import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.configuration import config

TrainingStrategy = Literal[
    # LoRA always uses fused Q, K, V matrices for efficiency.
    "full-lora-and-head",  # All layers and a classification head.
    "extended-lora-and-head",  # Every second layer and a classification head.
    "minimal-lora-and-head",  # Every fourth layer and a classification head.
    "head-only",  # Only a classification head.
]


class RetrainIfChanged:
    """A change to this field means existing weights no longer match the config."""


class TrainingConfig(BaseModel):
    """Training configuration for ModernBERT."""

    # Below parameters require re-training after changing.
    max_training_epochs: Annotated[int, RetrainIfChanged] = Field(3, ge=1)
    random_seed: Annotated[int | None, RetrainIfChanged] = config.random_seed
    strategy: Annotated[TrainingStrategy, RetrainIfChanged] = "extended-lora-and-head"
    learning_rate: Annotated[float, RetrainIfChanged] = Field(
        default=2e-5,
        gt=0.0,
        lt=10.0,
    )
    max_samples_per_split: Annotated[int, RetrainIfChanged] = Field(
        default=0,
        ge=0,
        description=(
            "Maximum number of samples per dataset split. "
            "0 means all samples. Affects the trained model."
        ),
    )
    # Below parameters do NOT require re-training after changing.
    batch_size: int = Field(2, ge=1)
    gradient_accumulation_steps: int = Field(
        8,
        ge=1,
        description=(
            "Increases effective batch size without holding the whole batch in memory, "
            "but can add overhead compared with fitting that batch directly."
        ),
    )
    bfloat16: bool = Field(
        default=True,
        description=(
            "A 16-bit number format. It uses less memory than standard 32-bit floats "
            "and runs faster but can run only on supported hardware."
        ),
    )

    @classmethod
    def retrain_if_changed_field_names(cls) -> frozenset[str]:
        """
        Return fields whose change requires retraining.

        Returns:
            frozenset[str]: Names of fields annotated with `RetrainIfChanged`.
        """
        return frozenset(
            name
            for name, field in cls.model_fields.items()
            if RetrainIfChanged in field.metadata
        )

    def retrain_if_changed_fields(self) -> dict[str, object]:
        """
        Return training fields whose change requires retraining.

        Returns:
            dict[str, object]: Values of fields that require retraining when changed.
        """
        return self.model_dump(include=set(type(self).retrain_if_changed_field_names()))

    def retrain_if_changed_fingerprint(self) -> str:
        """
        Hash the fields whose change requires retraining.

        Returns:
            str: A stable hash of the retraining-relevant training configuration.
        """
        canonical = json.dumps(
            self.retrain_if_changed_fields(),
            indent=2,
            sort_keys=True,
        )
        return hashlib.sha1(  # noqa: S324, No sensitive information is hashed.
            canonical.encode()
        ).hexdigest()


class RemoteTrainingMetadata(BaseModel):
    """Training metadata published alongside a remote ModernBERT model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    training_config: TrainingConfig
    dataset_version: str


def retrain_if_changed_differences(
    requested: TrainingConfig,
    remote: TrainingConfig,
) -> dict[str, tuple[object, object]]:
    """
    Find retraining-relevant fields that differ between two training configurations.

    Args:
        requested (TrainingConfig): A configuration requested for training.
        remote (TrainingConfig): A configuration stored alongside a remote model.

    Returns:
        dict[str, tuple[object, object]]: A mapping from field name to a tuple of
            requested and remote values for every field that differs.
    """
    requested_fields = requested.retrain_if_changed_fields()
    remote_fields = remote.retrain_if_changed_fields()
    differences: dict[str, tuple[object, object]] = {}

    for name, requested_value in requested_fields.items():
        remote_value = remote_fields[name]

        if isinstance(requested_value, float) and isinstance(remote_value, float):
            matches = math.isclose(
                requested_value,
                remote_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        else:
            matches = requested_value == remote_value

        if not matches:
            differences[name] = (requested_value, remote_value)

    return differences


class InferenceConfig(BaseModel):
    """Inference-time configuration for ModernBERT."""

    batch_size: int = Field(4, ge=1)
    bfloat16: bool = Field(
        default=True,
        description=(
            "A 16-bit number format. It uses less memory than standard 32-bit floats "
            "and runs faster but can run only on supported hardware."
        ),
    )
