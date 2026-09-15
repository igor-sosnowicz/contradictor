"""
NLI scoring for generated premises.

Cross-encoder role:
Comparing premise vs claim is not a logical entailment, so
NLI is used as a *negative* filter only.

    premise "Reactors run above 90% capacity factor year round."
    claim   "Nuclear is the only realistic way to decarbonise a grid."
                            |
                            v
    | contradiction 0.000 | entailment 0.000 | neutral 1.000 |
      ^ this is the gate                       ^ a good premise lives here

A premise that contradicts its own claim is fabricated ground truth and is
rejected.
"""

from functools import lru_cache
from typing import Protocol

import numpy as np
from loguru import logger

from src.configuration import config as global_config

ENTAILMENT_LABEL = "entailment"
CONTRADICTION_LABEL = "contradiction"


class EntailmentScorer(Protocol):
    """Scores the NLI relation between a premise and a conclusion."""

    def score(self, premise: str, conclusion: str) -> float:
        """Return the entailment probability of conclusion given premise."""
        ...

    def score_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return entailment probabilities for several (premise, conclusion) pairs."""
        ...

    def contradiction(self, premise: str, conclusion: str) -> float:
        """Return the probability that the premise contradicts the conclusion."""
        ...

    def contradiction_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return contradiction probabilities for several pairs."""
        ...


class CrossEncoderEntailmentScorer:
    """
    NLI scorer backed by a sentence-transformers cross-encoder.

    The model is loaded lazily on first use, so importing the pipelines stays
    cheap and test runs that stub the scorer never download weights.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Store the model name; loading is deferred to the first score call."""
        self.model_name: str = model_name or global_config.nli_model_name
        self._model: object | None = None
        self._label_index: dict[str, int] = {}

    def _ensure_model(self) -> object:
        """Load the cross-encoder and locate its label indices."""
        if self._model is not None:
            return self._model

        from sentence_transformers import CrossEncoder

        logger.info(f"Loading NLI model {self.model_name!r}")
        model = CrossEncoder(self.model_name)
        id2label: dict[int, str] = dict(model.config.id2label)  # type: ignore

        for wanted in (ENTAILMENT_LABEL, CONTRADICTION_LABEL):
            index = next(
                (i for i, label in id2label.items() if wanted in str(label).lower()),
                None,
            )
            if index is None:
                raise ValueError(
                    f"NLI model {self.model_name!r} has no {wanted!r} label "
                    f"among {sorted(id2label.values())}."
                )
            self._label_index[wanted] = index

        self._model = model
        return model

    def _probabilities(self, pairs: list[tuple[str, str]], label: str) -> list[float]:
        """Return one label's probability for each (premise, conclusion) pair."""
        if not pairs:
            return []

        model = self._ensure_model()
        logits = np.asarray(model.predict(pairs), dtype=np.float64)  # type: ignore[attr-defined]
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        index = self._label_index[label]
        return [float(row[index]) for row in _softmax(logits)]

    def score(self, premise: str, conclusion: str) -> float:
        """Return the entailment probability of conclusion given premise."""
        return self.score_many([(premise, conclusion)])[0]

    def score_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return entailment probabilities for several (premise, conclusion) pairs."""
        return self._probabilities(pairs, ENTAILMENT_LABEL)

    def contradiction(self, premise: str, conclusion: str) -> float:
        """Return the probability that the premise contradicts the conclusion."""
        return self.contradiction_many([(premise, conclusion)])[0]

    def contradiction_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return contradiction probabilities for several pairs."""
        return self._probabilities(pairs, CONTRADICTION_LABEL)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax that is stable for large logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


@lru_cache(maxsize=1)
def default_scorer() -> EntailmentScorer:
    """Return the process-wide cross-encoder scorer, creating it on first use."""
    return CrossEncoderEntailmentScorer()
