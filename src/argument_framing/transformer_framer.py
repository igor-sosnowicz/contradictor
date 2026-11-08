"""Module with a encoder-only transformer implementation of argument framing."""

from collections.abc import Collection, Iterable
from itertools import batched
from pathlib import Path
from typing import Any, Literal, cast, override

import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
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

TrainingStrategy = Literal[
    # LoRA always uses fused Q, K, V matrices for efficiency.
    "full-lora-and-head",  # All layers and a classification head.
    "extended-lora-and-head",  # Every second layer and a classification head.
    "minimal-lora-and-head",  # Every fourth layer and a classification head.
    "head-only",  # Only a classification head.
]


class ModernBertTrainer:
    def __init__(
        self,
        classes_count: int,
        dataset: ArgumentFramingDataset,
        training_directory: Path,
        strategy: TrainingStrategy,
        batch_size: int,
    ) -> None:
        if classes_count <= 1:
            raise ValueError(
                "A number of classes for classification should be larger than one, "
                f"got: {classes_count}"
            )
        if batch_size < 1:
            raise ValueError("Batch size has to larger than 0.")

        self._model_name = "answerdotai/ModernBERT-base"
        self._tokeniser = AutoTokenizer.from_pretrained(self._model_name)
        self._pretrained_model = ModernBertForSequenceClassification.from_pretrained(
            self._model_name,
            num_labels=classes_count,
        )
        self._adjust_model_to_training_strategy(strategy)

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

        self._training_configuration = TrainingArguments(
            output_dir=str(self._training_directory),
            num_train_epochs=3,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"every_n_layers": 4},
            bf16=True,
            learning_rate=2e-5,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        self._dataset_wrapper = dataset
        self._dataset = None

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

    async def _prepare_dataset(self, dataset: ArgumentFramingDataset) -> DatasetDict:
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
                    dataset.get_split(SubsetName.TRAINING),
                    preserve_index=False,
                    features=features,
                ),
                "validation": Dataset.from_pandas(
                    dataset.get_split(SubsetName.VALIDATION),
                    preserve_index=False,
                    features=features,
                ),
                "test": Dataset.from_pandas(
                    dataset.get_split(SubsetName.TESTING),
                    preserve_index=False,
                    features=features,
                ),
            }
        )

    async def __call__(
        self,
    ) -> tuple[AutoTokenizer, ModernBertForSequenceClassification]:
        """Fine-tune ModernBERT on the framer classification task."""
        if self._dataset is None:
            self._dataset = await self._prepare_dataset(dataset=self._dataset_wrapper)
        dataset = cast("DatasetDict", self._dataset)

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
            f"Fine-tuning {self._model_name} has succeeded. Training output:\n\n"
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


class TransformerFramer(ArgumentFramer):
    """Encoder-only transformer based implementation of argument framing."""

    def __init__(
        self,
        training_strategy_if_model_missing: TrainingStrategy = "extended-lora-and-head",
        batch_size: int = 2,
    ) -> None:
        """
        Initialise the transformer framer.

        Args:
            training_strategy_if_model_missing (TrainingStrategy): LoRA strategy used
                when no fine-tuned model is available and training is triggered.
            batch_size (int): Number of arguments processed per inference batch.

        Raises:
            ValueError: If `batch_size` is smaller than 1.
        """
        if batch_size < 1:
            raise ValueError("Batch size has to larger than 0.")
        self._batch_size = batch_size
        self._model: ModernBertForSequenceClassification | None = None
        self._tokeniser: AutoTokenizer | None = None
        self._dataset = ArgumentFramingDataset()
        self._training_strategy = training_strategy_if_model_missing
        self._training_directory = self._to_training_directory()

    def _to_training_directory(self) -> Path:
        subdirectory_name = f"modernbert-{self._training_strategy}"
        return (
            config.data_directory
            / config.model_subdirectory
            / TransformerFramer.__name__
            / subdirectory_name
        )

    @override
    def frame(self, arguments: Iterable[Argument]) -> Collection[FramedArgument]:
        if self._model is None or self._tokeniser is None:
            raise NotPreparedError(
                "Cannot frame arguments with the unprepared framer. "
                f"Call `await {TransformerFramer.__name__}.prepare(...)` first."
            )

        id2label = self._model.config.id2label
        if id2label is None:
            raise ValueError(
                "The model config has no `id2label` mapping. Train the model with "
                f"`{TransformerFramer.__name__}.prepare()` first, or set "
                "`model.config.id2label` manually."
            )

        # Interpretative frames ordered by their logit index.
        index_to_frame = [
            InterpretativeFrame(frame_name)
            for _, frame_name in sorted(id2label.items(), key=lambda kv: int(kv[0]))
        ]

        self._model.eval()
        device = next(self._model.parameters()).device

        # AutoTokenizer is not typed as callable by mypy; cast to a tokenizer
        # base that is callable and exposes `model_max_length`.
        tokeniser = cast("PreTrainedTokenizerBase", self._tokeniser)
        max_length = getattr(tokeniser, "model_max_length", 8192)

        framed_arguments: list[FramedArgument] = []
        with torch.no_grad():
            # The last batch can be smaller than previous batches.
            for argument_batch in batched(arguments, self._batch_size, strict=False):
                texts = [str(argument) for argument in argument_batch]

                inputs = tokeniser(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                outputs = self._model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

                predicted_indices = logits.argmax(dim=-1).tolist()
                probability_rows = probabilities.tolist()

                for argument, predicted_index, probability_row in zip(
                    argument_batch,
                    predicted_indices,
                    probability_rows,
                    strict=True,
                ):
                    primary_frame = index_to_frame[predicted_index]
                    frame_probabilities = {
                        frame: float(probability)
                        for frame, probability in zip(
                            index_to_frame, probability_row, strict=True
                        )
                    }
                    framed_arguments.append(
                        FramedArgument(
                            claim=argument.claim,
                            evidence=argument.evidence,
                            primary_frame=primary_frame,
                            frame_probabilities=frame_probabilities,
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
            classes_count=labels_count,
            dataset=self._dataset,
            strategy=cast("TrainingStrategy", self._training_strategy),
            training_directory=self._training_directory,
            batch_size=self._batch_size,
        )
        return await trainer()

    @override
    def _load(
        self,
    ) -> tuple[AutoTokenizer | None, ModernBertForSequenceClassification | None]:
        try:
            directory = str(self._to_training_directory())
            tokeniser = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            model = ModernBertForSequenceClassification.from_pretrained(
                directory, local_files_only=True
            )
        except (OSError, ValueError) as e:
            logger.debug(
                f"A fine-tuned version ModernBERT is not available for loading: {e}"
            )
            return None, None
        else:
            return cast("AutoTokenizer", tokeniser), cast(
                "ModernBertForSequenceClassification", model
            )

    @override
    async def perform_tuning(self) -> dict[str, float]:
        raise NotImplementedError("perform_tuning")


__all__ = ["TransformerFramer"]
