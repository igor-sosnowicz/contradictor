"""Module with a training utilities for ModernBERT framing detection."""

import json
from pathlib import Path
from typing import Any, cast

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from loguru import logger
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    ModernBertForSequenceClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.argument_framing.argument_framing_dataset import ArgumentFramingDataset
from src.argument_framing.modernbert_framer.config import (
    RemoteTrainingMetadata,
    TrainingConfig,
    TrainingStrategy,
)
from src.argument_framing.modernbert_framer.utils import (
    EvaluationMetrics,
    calculate_evaluation_metrics,
)
from src.configuration import config
from src.data_models.data_models import (
    InterpretativeFrame,
    SubsetName,
)


class ModernBertTrainer:
    """Trainer for underlying ModernBERT encoder-only transformer."""

    MODEL_NAME = "answerdotai/ModernBERT-base"

    def __init__(
        self,
        classes_count: int,
        dataset: ArgumentFramingDataset,
        training_directory: Path,
        training_config: TrainingConfig,
    ) -> None:
        """
        Initialise ModernBERT training configuration.

        Args:
            classes_count (int): A number of classes that will be predicted.
            dataset (ArgumentFramingDataset): A dataset for training, validation, and
                testing.
            training_directory (Path): A root directory, where sub-directories with
                trained models and other training artifacts should be stored,
                and - later on - loaded from.
            training_config (TrainingConfig): Configuration for fine-tuning of
                ModernBERT.

        Raises:
            ValueError: Raised if a number of classes was smaller or equal to one.
            NotADirectoryError: Raised if a training directory does not exist or
                is not a directory.
        """
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
            gradient_checkpointing=config.device != "cpu",
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
        self._max_samples = training_config.max_samples_per_split

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
            # PEFT wraps the model but proxies its public API, which a type
            # checker cannot see through.
            self._pretrained_model = cast(
                "ModernBertForSequenceClassification",
                cast("object", get_peft_model(self._pretrained_model, lora_config)),
            )
            cast("Any", self._pretrained_model).print_trainable_parameters()

    @staticmethod
    def _tokenise_batch(
        examples: dict[str, list[str]], tokeniser: PreTrainedTokenizerBase
    ) -> BatchEncoding:
        return tokeniser(
            examples["text"],
            truncation=True,
            max_length=getattr(tokeniser, "model_max_length", 8192),
        )

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
    ) -> tuple[
        PreTrainedTokenizerBase, ModernBertForSequenceClassification, EvaluationMetrics
    ]:
        """Fine-tune ModernBERT on the framer classification task."""
        dataset = await self._prepare_dataset(
            dataset=self._dataset_wrapper, max_samples=self._max_samples
        )

        # Tokenise raw text so the splits keep columns matching the model's forward
        # signature: input_ids, attention_mask, and labels.
        dataset = dataset.map(
            ModernBertTrainer._tokenise_batch,
            batched=True,
            remove_columns=["text"],
            fn_kwargs={"tokeniser": self._tokeniser},
        )
        dataset = dataset.rename_column("interpretative_frame", "labels")

        trainer = Trainer(
            model=self._pretrained_model,
            args=self._training_configuration,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=self._tokeniser,
            # Pads to longest in batch instead of in the dataset, saving processing
            # by skipping unnecessary padding.
            compute_metrics=calculate_evaluation_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=self._tokeniser),
        )

        training_output = trainer.train()
        evaluation = cast(
            "EvaluationMetrics",
            trainer.evaluate(eval_dataset=cast("Any", dataset["test"])),
        )

        metrics_path = self._training_directory / "evaluation_metrics.json"
        metrics_path.write_text(json.dumps(evaluation, indent=2))

        metadata_path = self._training_directory / "training_config.json"
        metadata_path.write_text(
            RemoteTrainingMetadata(
                training_config=self._config,
                dataset_version=self._dataset_wrapper.version,
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"Fine-tuning {self.MODEL_NAME} has succeeded. Training output:\n\n"
            f"{training_output}"
        )

        model = cast("ModernBertForSequenceClassification | PeftModel", trainer.model)
        if isinstance(trainer.model, PeftModel):
            logger.info(
                "Merging LoRA adapters with the pretrained base ModernBERT model."
            )
            model = cast(
                "ModernBertForSequenceClassification", trainer.model.merge_and_unload()
            )

        model.save_pretrained(str(self._training_directory))
        self._tokeniser.save_pretrained(self._training_directory)

        model = ModernBertForSequenceClassification.from_pretrained(
            str(self._training_directory), local_files_only=True
        )
        tokeniser = AutoTokenizer.from_pretrained(
            str(self._training_directory), local_files_only=True
        )
        return cast("PreTrainedTokenizerBase", tokeniser), model, evaluation
