"""Module with a encoder-only transformer implementation of argument framing."""

from collections.abc import Collection, Iterable
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, override

import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    ModernBertForSequenceClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.argument_framing.argument_framer import ArgumentFramer
from src.argument_framing.argument_framing_dataset import ArgumentFramingDataset
from src.configuration import config
from src.data_models.data_models import (
    LABEL_TO_FRAME,
    Argument,
    FramedArgument,
    InterpretativeFrame,
    SubsetName,
)
from src.utils.errors import NotPreparedError

if TYPE_CHECKING:
    from typing import Any


TrainingStrategy = Literal[
    # LoRA always uses fused Q, K, V matrices for efficiency.
    "full-lora-and-head",  # All layers and a classification head.
    "extended-lora-and-head",  # Every second layer and a classification head.
    "minimal-lora-and-head",  # Every fourth layer and a classification head.
    "head-only",  # Only a classification head.
]


class ModernBERTConfig(BaseModel):
    """Training configuration for ModernBERT."""

    batch_size: int = Field(2, ge=1)
    max_training_epochs: int = Field(3, ge=1)
    random_seed: int | None = config.random_seed
    strategy: TrainingStrategy = "extended-lora-and-head"
    learning_rate: float = Field(default=2e-5, gt=0.0, lt=10.0)
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


class ModernBertTrainer:
    """Trainer for underlying ModernBERT encoder-only transformer."""

    MODEL_NAME = "answerdotai/ModernBERT-base"

    def __init__(
        self,
        classes_count: int,
        dataset: ArgumentFramingDataset,
        training_directory: Path,
        training_config: ModernBERTConfig,
        max_samples_per_split: int,
    ) -> None:
        if classes_count <= 1:
            raise ValueError(
                "A number of classes for classification should be larger than one, "
                f"got: {classes_count}"
            )

        self._tokeniser = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._pretrained_model = ModernBertForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=classes_count,
        )
        self._adjust_model_to_training_strategy(training_config.strategy)

        # Store the index-to-frame contract with the model so inference can map
        # logit indices back to interpretative frames authoritatively.
        self._pretrained_model.config.id2label = {
            str(index): frame.value for index, frame in enumerate(InterpretativeFrame)
        }

        if training_directory.exists() and not training_directory.is_dir():
            raise NotADirectoryError(
                f"The provided path to a training directory ({training_directory}) "
                "should lead to an existent or non-existent directory but leads to "
                "a non-directory (possibly a file)."
            )
        training_directory.mkdir(parents=True, exist_ok=True)
        self._training_directory = training_directory

        additional_kwargs: dict[str, Any] = {}
        if training_config.random_seed is not None:
            additional_kwargs["seed"] = training_config.random_seed

        if training_config.bfloat16 and config.device == "cpu":
            logger.warning(
                "bfloat16 is not supported on CPU; disabling it "
                "even though it was requested in the training config."
            )

        self._training_configuration = TrainingArguments(
            output_dir=str(self._training_directory),
            num_train_epochs=training_config.max_training_epochs,
            per_device_train_batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"every_n_layers": 4},
            bf16=training_config.bfloat16 and config.device != "cpu",
            learning_rate=training_config.learning_rate,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            use_cache=config.device == "cpu",
            **additional_kwargs,
        )

        self._dataset_wrapper = dataset
        self._config = training_config
        self._max_samples = max_samples_per_split

    def _adjust_model_to_training_strategy(self, strategy: TrainingStrategy) -> None:
        n_layers = self._pretrained_model.config.num_hidden_layers

        common_lora = {
            "task_type": TaskType.SEQ_CLS,
            "r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.05,
            "target_modules": ["Wqkv"],
            "modules_to_save": ["head", "classifier"],
            "bias": "none",
        }

        lora_config = None

        match strategy.lower():
            case "full-lora-and-head":
                # LoRA on fused QKV in every layer.
                lora_config = LoraConfig(**cast("dict[str, Any]", common_lora))

            case "extended-lora-and-head":
                # LoRA on fused QKV in every second layer.
                lora_config = LoraConfig(
                    **cast("dict[str, Any]", common_lora),
                    layers_to_transform=list(range(0, n_layers, 2)),
                    layers_pattern="layers",
                )

            case "minimal-lora-and-head":
                # LoRA on fused QKV in every fourth layer.
                lora_config = LoraConfig(
                    **cast("dict[str, Any]", common_lora),
                    layers_to_transform=list(range(0, n_layers, 4)),
                    layers_pattern="layers",
                )

            case "head-only":
                # Freeze the backbone and train the classification head.
                for param in self._pretrained_model.model.parameters():
                    param.requires_grad = False

                for module in (
                    self._pretrained_model.head,
                    self._pretrained_model.classifier,
                ):
                    for param in module.parameters():
                        param.requires_grad = True

            case _:
                raise ValueError(f"Invalid training strategy: {strategy}")

        if lora_config is not None:
            self._pretrained_model = cast(
                "ModernBertForSequenceClassification",
                get_peft_model(self._pretrained_model, lora_config),
            )
            cast("Any", self._pretrained_model).print_trainable_parameters()

    async def _prepare_dataset(
        self, dataset: ArgumentFramingDataset, max_samples: int
    ) -> DatasetDict:
        await dataset.prepare()
        features = Features(
            {
                "text": Value("string"),
                "interpretative_frame": ClassLabel(names=list(InterpretativeFrame)),
            }
        )

        return DatasetDict(
            {
                "train": Dataset.from_pandas(
                    dataset.get_split(SubsetName.TRAINING, max_samples=max_samples),
                    preserve_index=False,
                    features=features,
                ),
                "validation": Dataset.from_pandas(
                    dataset.get_split(SubsetName.VALIDATION, max_samples=max_samples),
                    preserve_index=False,
                    features=features,
                ),
                "test": Dataset.from_pandas(
                    dataset.get_split(SubsetName.TESTING, max_samples=max_samples),
                    preserve_index=False,
                    features=features,
                ),
            }
        )

    async def __call__(
        self,
    ) -> tuple[AutoTokenizer, ModernBertForSequenceClassification]:
        """Fine-tune ModernBERT on the framer classification task."""
        dataset = cast(
            "DatasetDict",
            await self._prepare_dataset(
                dataset=self._dataset_wrapper, max_samples=self._max_samples
            ),
        )

        trainer = Trainer(
            model=self._pretrained_model,
            args=self._training_configuration,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=self._tokeniser,
            # It pads to longest in batch instead of in the dataset, saving processing
            # by skipping unnecessary padding.
            data_collator=DataCollatorForLanguageModeling(self._tokeniser, mlm=False),
        )

        training_output = trainer.train()

        logger.info(
            f"Fine-tuning {self.MODEL_NAME} has succeeded. Training output:\n\n"
            f"{training_output}"
        )
        trainer.save_model(str(self._training_directory))
        self._tokeniser.save_pretrained(self._training_directory)

        model = ModernBertForSequenceClassification.from_pretrained(
            str(self._training_directory), local_files_only=True
        )
        tokeniser = AutoTokenizer.from_pretrained(
            str(self._training_directory), local_files_only=True
        )
        return cast("AutoTokenizer", tokeniser), model


class ModernBertFramer(ArgumentFramer):
    """Encoder-only ModernBERT based implementation of argument framing."""

    def __init__(
        self,
        training_config: ModernBERTConfig | None = None,
        *,
        proof_of_concept_mode: bool = False,
        dataset: ArgumentFramingDataset,
    ) -> None:
        """
        Initialise the transformer framer.

        Args:
            training_config (ModernBERTConfig | None): Configuration of the training
                of a model. If None sane defaults are ensured depending on whether
                `proof_of_concept_mode` is enabled or not.
            proof_of_concept_mode (bool): Whether the model should use the smallest
                possible amount of computation and samples to solely prove it works.
            dataset (ArgumentFramingDataset): Dataset used for training, validation,
                and testing the model.
        """
        self._max_samples = 100 if proof_of_concept_mode else 0
        if proof_of_concept_mode:
            self._config = training_config or ModernBERTConfig(
                batch_size=1,
                max_training_epochs=1,
                gradient_accumulation_steps=1,
                random_seed=0,
                strategy="head-only",
            )
        else:
            self._config = training_config or ModernBERTConfig(
                batch_size=2,
                max_training_epochs=3,
                gradient_accumulation_steps=8,
            )
        self._model: ModernBertForSequenceClassification | None = None
        self._tokeniser: AutoTokenizer | None = None
        self._dataset = dataset

    @property
    def _training_directory(self) -> Path:
        subdirectory_name = f"modernbert-{self._config.strategy}"
        return (
            config.data_directory
            / config.model_subdirectory
            / ModernBertFramer.__name__
            / subdirectory_name
        )

    @override
    def frame(self, arguments: Iterable[Argument]) -> Collection[FramedArgument]:
        if self._model is None or self._tokeniser is None:
            raise NotPreparedError(
                "Cannot frame arguments with the unprepared framer. "
                f"Call `await {ModernBertFramer.__name__}.prepare(...)` first."
            )

        id2label = self._model.config.id2label
        if id2label is None:
            raise ValueError(
                "The model config has no `id2label` mapping. Train the model with "
                f"`{ModernBertFramer.__name__}.prepare()` first, or set "
                "`model.config.id2label` manually."
            )

        # Interpretative frames ordered by their logit index.
        index_to_frame = [
            InterpretativeFrame(frame_name)
            for _, frame_name in sorted(id2label.items(), key=lambda kv: int(kv[0]))
        ]

        self._model.eval()

        # Cast to a tokenizer base that is callable and exposes `model_max_length`.
        tokeniser = cast("PreTrainedTokenizerBase", self._tokeniser)
        max_length = getattr(tokeniser, "model_max_length", 8192)

        return self._infer(
            model=self._model,
            tokeniser=tokeniser,
            arguments=arguments,
            max_length=max_length,
            index_to_frame=index_to_frame,
        )

    def _infer(
        self,
        model: ModernBertForSequenceClassification,
        tokeniser: PreTrainedTokenizerBase,
        arguments: Iterable[Argument],
        max_length: int,
        index_to_frame: list[InterpretativeFrame],
    ) -> list[FramedArgument]:
        device = next(model.parameters()).device
        framed_arguments: list[FramedArgument] = []
        with torch.no_grad():
            # The last batch can be smaller than previous batches.
            for argument_batch in batched(
                arguments, self._config.batch_size, strict=False
            ):
                logits = model(
                    **tokeniser(
                        [str(argument) for argument in argument_batch],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(device)
                ).logits
                predicted_indices = logits.argmax(dim=-1).tolist()
                probability_rows = torch.nn.functional.softmax(logits, dim=-1).tolist()

                for argument, predicted_index, probability_row in zip(
                    argument_batch,
                    predicted_indices,
                    probability_rows,
                    strict=True,
                ):
                    framed_arguments.append(
                        FramedArgument(
                            claim=argument.claim,
                            evidence=argument.evidence,
                            primary_frame=index_to_frame[predicted_index],
                            frame_probabilities={
                                frame: float(probability)
                                for frame, probability in zip(
                                    index_to_frame, probability_row, strict=True
                                )
                            },
                        )
                    )
        return framed_arguments

    @override
    async def prepare(self) -> None:
        await self._dataset.prepare()

        tokeniser, model = self._load()
        if model is None:
            tokeniser, model = await self._train()

        self._model = model
        self._tokeniser = tokeniser

    @override
    async def _train(self) -> tuple[AutoTokenizer, ModernBertForSequenceClassification]:
        labels_count = len(LABEL_TO_FRAME)
        trainer = ModernBertTrainer(
            training_config=self._config,
            classes_count=labels_count,
            dataset=self._dataset,
            training_directory=self._training_directory,
            max_samples_per_split=self._max_samples,
        )
        return await trainer()

    @override
    def _load(
        self,
    ) -> tuple[AutoTokenizer | None, ModernBertForSequenceClassification | None]:
        try:
            directory = str(self._training_directory)
            tokeniser = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            model = ModernBertForSequenceClassification.from_pretrained(
                directory, local_files_only=True
            )
        except (OSError, ValueError) as e:
            logger.debug(
                f"A fine-tuned version ModernBERT is not available for loading: {e}"
            )
            return None, None

        return cast("AutoTokenizer", tokeniser), cast(
            "ModernBertForSequenceClassification", model
        )

    @override
    async def perform_tuning(self) -> dict[str, float]:
        raise NotImplementedError("perform_tuning")


__all__ = ["ModernBertFramer"]
