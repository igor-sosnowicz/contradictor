"""Module with HuggingFace transformers implementation of NLI."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Self, override

from transformers import pipeline

from src.configuration import config
from src.data_models.data_models import NLIPrediction
from src.nli.nli import NLI
from src.nli.utils import to_nli_enum
from src.utils.errors import NotPreparedError


class TransformersNLI(NLI):
    """HuggingFace's transformers-based implementation of NLI."""

    def __init__(
        self,
        model: str = config.transformer_nli_model,
        batch_size: int = 16,
        max_workers: int = 1,
    ) -> None:
        """
        Initialise a transformer NLI pipeline.

        Args:
            model (str): A transformer model to be used. Defaults to
               the model from configuration.
            batch_size (int): A number of samples in a batch processed simultaneously.
                Defaults to 16.
            max_workers (int): A maximum number of workers allowed to run in separate
                threads. Defaults to 1 not to prevent CPU-bound tasks from getting
                executed.

        Raises:
            ValueError: Raised if a negative batch size was provided.
        """
        if batch_size <= 0:
            raise ValueError("Batch size has to be a positive integer.")
        if max_workers <= 0:
            raise ValueError(
                "A maximum number of workers (`max_workers`) has to be a positive "
                "integer."
            )
        self._nli_model = pipeline(
            "text-classification", model=model, batch_size=batch_size
        )
        self.max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self._prepared = False

    def _reshape_to_nested(
        self, flat: list[NLIPrediction], other_texts: list[str]
    ) -> list[list[NLIPrediction]]:
        chunk_size = len(other_texts)
        return [flat[i : i + chunk_size] for i in range(0, len(flat), chunk_size)]

    @override
    async def __call__(
        self, reference_texts: list[str], other_texts: list[str]
    ) -> list[list[NLIPrediction]]:
        if not self._prepared:
            raise NotPreparedError(
                f"{TransformersNLI.__name__} needs to be used with a context manager. "
                f"Use: `with {TransformersNLI.__name__} as nli:`"
            )

        if not reference_texts or not other_texts:
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._perform_inference,
            reference_texts,
            other_texts,
        )

    def _perform_inference(
        self, reference_texts: list[str], other_texts: list[str]
    ) -> list[list[NLIPrediction]]:
        pairs = [
            {"text": reference_text, "text_pair": hypothesis}
            for reference_text in reference_texts
            for hypothesis in other_texts
        ]
        inferences = self._nli_model(pairs)

        flat_output = [
            NLIPrediction(
                result=to_nli_enum(result["label"]), confidence=(result["score"])
            )
            for result in inferences
        ]
        return self._reshape_to_nested(flat_output, other_texts)

    @override
    def __enter__(self) -> Self:
        # The first usage triggers download of the model files.
        self._nli_model(
            [
                {
                    "text": "Download the model.",  # Premise/reference
                    "text_pair": "He is from France.",  # Hypothesis
                }
            ]
        )
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._prepared = True
        return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=exc_type is not None)
