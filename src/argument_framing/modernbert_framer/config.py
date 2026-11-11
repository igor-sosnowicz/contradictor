"""Configuration data structures for the ModernBERT framer."""

from typing import Literal

from pydantic import BaseModel, Field

from src.configuration import config

TrainingStrategy = Literal[
    # LoRA always uses fused Q, K, V matrices for efficiency.
    "full-lora-and-head",  # All layers and a classification head.
    "extended-lora-and-head",  # Every second layer and a classification head.
    "minimal-lora-and-head",  # Every fourth layer and a classification head.
    "head-only",  # Only a classification head.
]


class TrainingConfig(BaseModel):
    """Training configuration for ModernBERT."""

    # Below parameters require re-training after changing.
    max_training_epochs: int = Field(3, ge=1)
    random_seed: int | None = config.random_seed
    strategy: TrainingStrategy = "extended-lora-and-head"
    learning_rate: float = Field(default=2e-5, gt=0.0, lt=10.0)
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
