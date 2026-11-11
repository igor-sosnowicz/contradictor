"""Module with a encoder-only transformer implementation of argument framing."""

import hashlib
import json
from collections.abc import Collection, Iterable
from itertools import batched
from pathlib import Path
from typing import Final, cast, override

import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    PreTrainedTokenizerBase,
)

from src.argument_framing.argument_framer import ArgumentFramer
from src.argument_framing.argument_framing_dataset import ArgumentFramingDataset
from src.argument_framing.modernbert_framer.config import (
    InferenceConfig,
    TrainingConfig,
)
from src.argument_framing.modernbert_framer.trainer import ModernBertTrainer
from src.argument_framing.modernbert_framer.utils import EvaluationMetrics
from src.configuration import config
from src.data_models.abstract.configuration import ConfigurationValue
from src.data_models.data_models import (
    LABEL_TO_FRAME,
    Argument,
    FramedArgument,
    InterpretativeFrame,
)
from src.utils.errors import NotPreparedError


class ModernBertFramer(ArgumentFramer):
    """Encoder-only ModernBERT based implementation of argument framing."""

    HUGGING_FACE_MODEL_ID: Final[str] = "contradictor-team/modernbert-framer"

    def __init__(
        self,
        inference_config: InferenceConfig | None = None,
        training_config: TrainingConfig | None = None,
        *,
        proof_of_concept_mode: bool = False,
        dataset: ArgumentFramingDataset,
    ) -> None:
        """
        Initialise the transformer framer.

        Args:
            inference_config (InferenceConfig | None): Configuration of inference of
                the model. Defaults to None, which means sane defaults will be used.
            training_config (TrainingConfig | None): Configuration of the training
                of a model. If None sane defaults are ensured depending on whether
                `proof_of_concept_mode` is enabled or not.
            proof_of_concept_mode (bool): Whether the model should use the smallest
                possible amount of computation and samples to solely prove it works.
            dataset (ArgumentFramingDataset): Dataset used for training, validation,
                and testing the model.
        """
        self._proof_of_concept_mode = proof_of_concept_mode
        if proof_of_concept_mode:
            self._training_config = training_config or TrainingConfig(
                batch_size=1,
                max_training_epochs=1,
                gradient_accumulation_steps=1,
                random_seed=0,
                strategy="head-only",
            )
        else:
            self._training_config = training_config or TrainingConfig(
                batch_size=2,
                max_training_epochs=3,
                gradient_accumulation_steps=8,
            )
        self._inference_config = inference_config or InferenceConfig(
            batch_size=4 + 12 * int(config.device != "cpu"),  # 4 - CPU, 16 - GPU
            bfloat16=config.device != "cpu",
        )
        self._model: ModernBertForSequenceClassification | None = None
        self._tokeniser: PreTrainedTokenizerBase | None = None
        self._dataset = dataset
        self._evaluation: EvaluationMetrics | None = None

    @property
    def _training_directory(self) -> Path:
        training_config_hash = hashlib.sha1(  # noqa: S324, No sensitive information is hashed.
            self._training_config.model_dump_json(
                indent=2,
                exclude={  # First two - can be different at training vs inference time.
                    "batch_size",
                    "bfloat16",
                    "gradient_accumulation_steps",  # It does not matter after training.
                },
            ).encode()
        ).hexdigest()

        subdirectory_name = f"modernbert-{training_config_hash}-{self._dataset.version}"
        return (
            config.data_directory
            / config.model_subdirectory
            / ModernBertFramer.__name__
            / subdirectory_name
        )

    @property
    def _max_sequence_length(self) -> int:
        return getattr(self._tokeniser, "model_max_length", 8192)

    @property
    def _max_samples_per_split(self) -> int:
        return 100 if self._proof_of_concept_mode else 0

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

        tokeniser = self._tokeniser

        return self._infer(
            model=self._model,
            tokeniser=tokeniser,
            arguments=arguments,
            index_to_frame=index_to_frame,
        )

    def _infer(
        self,
        model: ModernBertForSequenceClassification,
        tokeniser: PreTrainedTokenizerBase,
        arguments: Iterable[Argument],
        index_to_frame: list[InterpretativeFrame],
    ) -> list[FramedArgument]:
        device = next(model.parameters()).device
        framed_arguments: list[FramedArgument] = []
        with torch.no_grad():
            # The last batch can be smaller than previous batches.
            for argument_batch in batched(
                arguments, self._inference_config.batch_size, strict=False
            ):
                logits = model(
                    **tokeniser(
                        [str(argument) for argument in argument_batch],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self._max_sequence_length,
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

        tokeniser, model, evaluation_metrics = self._load()
        if model is None:
            tokeniser, model, evaluation_metrics = await self._train()

        self._model = model
        self._tokeniser = tokeniser
        self._evaluation = evaluation_metrics

        if self._inference_config.bfloat16:
            if config.device == "cpu":
                logger.warning(
                    "bfloat16 is not supported on CPU; using the loaded model dtype "
                    "even though it was requested in the inference config."
                )
            else:
                # `PreTrainedModel.to` is wrapped with `functools.wraps`, which
                # confuses type checkers; call the plain module method instead.
                cast("torch.nn.Module", self._model).to(dtype=torch.bfloat16)

    @override
    async def _train(
        self,
    ) -> tuple[
        PreTrainedTokenizerBase, ModernBertForSequenceClassification, EvaluationMetrics
    ]:
        labels_count = len(LABEL_TO_FRAME)
        trainer = ModernBertTrainer(
            training_config=self._training_config,
            classes_count=labels_count,
            dataset=self._dataset,
            training_directory=self._training_directory,
            max_samples_per_split=self._max_samples_per_split,
        )
        return await trainer()

    def _load_from_disk(
        self,
    ) -> tuple[
        PreTrainedTokenizerBase | None,
        ModernBertForSequenceClassification | None,
        EvaluationMetrics | None,
    ]:
        try:
            directory = str(self._training_directory)
            tokeniser = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            model = ModernBertForSequenceClassification.from_pretrained(
                directory, local_files_only=True
            )
        except (OSError, ValueError) as e:
            logger.debug(
                f"A fine-tuned version of ModernBERT is not available for loading: {e}"
            )
            return None, None, None

        metrics_path = self._training_directory / "evaluation_metrics.json"
        try:
            evaluation_metrics = json.loads(metrics_path.read_text())
        except FileNotFoundError:
            logger.warning(
                f"Evaluation metrics file not found at {metrics_path}; "
                "returning None for evaluation metrics."
            )
            evaluation_metrics = None

        return (
            cast("PreTrainedTokenizerBase", tokeniser),
            model,
            evaluation_metrics,
        )

    def _load_from_hugging_face(
        self,
    ) -> tuple[
        PreTrainedTokenizerBase | None,
        ModernBertForSequenceClassification | None,
        EvaluationMetrics | None,
    ]:
        tokeniser = AutoTokenizer.from_pretrained(self.HUGGING_FACE_MODEL_ID)
        model = ModernBertForSequenceClassification.from_pretrained(
            self.HUGGING_FACE_MODEL_ID
        )
        metrics_path = Path(
            hf_hub_download(
                repo_id=self.HUGGING_FACE_MODEL_ID,
                filename="evaluation_metrics.json",
            )
        )
        try:
            evaluation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning(
                f"Evaluation metrics file not found at {metrics_path}; "
                "returning None for evaluation metrics."
            )
            evaluation_metrics = None

        return cast("PreTrainedTokenizerBase", tokeniser), model, evaluation_metrics

    @override
    def _load(
        self,
    ) -> tuple[
        PreTrainedTokenizerBase | None,
        ModernBertForSequenceClassification | None,
        EvaluationMetrics | None,
    ]:
        artifacts = self._load_from_disk()
        if all(artifact is not None for artifact in artifacts):
            return artifacts

        artifacts = self._load_from_hugging_face()
        if all(artifact is not None for artifact in artifacts):
            return artifacts

        return None, None, None

    @override
    async def perform_tuning(self) -> dict[str, float]:
        raise NotImplementedError("perform_tuning")

    @override
    def get_configuration(self) -> dict[str, ConfigurationValue]:
        return {
            "proof_of_concept_mode": self._proof_of_concept_mode,
            "max_sequence_length": self._max_sequence_length,
            "max_samples_per_split": self._max_samples_per_split,
            "metrics_on_test_set": cast("ConfigurationValue", self._evaluation),
            **self._training_config.model_dump(),
        }


__all__ = ["ModernBertFramer"]
