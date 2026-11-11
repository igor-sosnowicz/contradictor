"""Module with utilities for ModernBERT framing detection."""

from collections.abc import Callable
from typing import Final, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import EvalPrediction

from src.data_models.data_models import InterpretativeFrame

type MetricValue = float | int | dict[str, "MetricValue"]
type EvaluationMetrics = dict[str, MetricValue]

MODERNBERT_MAX_CONTEXT_LENGTH: Final[int] = 8192


def calculate_evaluation_metrics(
    evaluation_prediction: EvalPrediction,
) -> EvaluationMetrics:
    """
    Calculate evaluation metrics from raw ModernBERT classification outputs.

    Args:
        evaluation_prediction (EvalPrediction): The Hugging Face `Trainer`
            gathers model logits and ground-truth label IDs into this object.
            Logits are float32 arrays of shape `(n_samples, num_labels)` from
            ModernBERT's `SequenceClassifierOutput`. Labels are integer IDs of
            shape `(n_samples,)`.

    Returns:
        EvaluationMetrics: Accuracy, micro and macro F1 scores, macro precision
            and recall, per-class precision, recall, F1 score, and support
            keyed by frame names, and the confusion matrix keyed first by the
            true frame and then by the predicted frame.
    """
    # ModernBERT's `SequenceClassifierOutput` yields only logits, so the Trainer
    # gathers a single array.
    predictions = cast("np.ndarray", evaluation_prediction.predictions)
    labels = cast("np.ndarray", evaluation_prediction.label_ids)

    # index of the highest logit
    predicted_ids = np.argmax(predictions, axis=-1)

    # Report all label indices, so frames absent from the
    # ground truth or the predictions receive zero-valued metrics.
    class_ids = np.arange(predictions.shape[-1])
    supports = np.bincount(labels, minlength=predictions.shape[-1])

    # Training assigns label indices in the order of the interpretative frame
    # enum, so the same order maps indices back to frame names here.
    frame_names = {
        index: frame.value for index, frame in enumerate(InterpretativeFrame)
    }

    # `average=None` returns one score per class, but scikit-learn does not
    # type that return as an array.
    # scikit-learn annotates `average` and `zero_division` more narrowly than
    # the runtime API, so silence the argument type errors on those keywords.
    precisions = cast(
        "np.ndarray",
        precision_score(
            labels,
            predicted_ids,
            labels=class_ids,
            average=None,  # pyright: ignore[reportArgumentType]
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
    )
    recalls = cast(
        "np.ndarray",
        recall_score(
            labels,
            predicted_ids,
            labels=class_ids,
            average=None,  # pyright: ignore[reportArgumentType]
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
    )
    per_class_f1_scores = cast(
        "np.ndarray",
        f1_score(
            labels,
            predicted_ids,
            labels=class_ids,
            average=None,  # pyright: ignore[reportArgumentType]
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
    )

    # Rows are true frames, columns are predicted frames.
    confusion = cast(
        "np.ndarray",
        confusion_matrix(labels, predicted_ids, labels=class_ids),
    )

    def as_metric(metric: Callable[..., float], **kwargs: str) -> float:
        return float(metric(labels, predicted_ids, zero_division=0, **kwargs))

    return {
        "accuracy": float(accuracy_score(labels, predicted_ids)),
        "micro_f1_score": as_metric(f1_score, average="micro"),
        "macro_f1_score": as_metric(f1_score, average="macro"),
        "precision": as_metric(precision_score, average="macro"),
        "recall": as_metric(recall_score, average="macro"),
        "per_class": {
            frame_names[int(class_id)]: {
                "precision": float(precisions[class_id]),
                "recall": float(recalls[class_id]),
                "f1_score": float(per_class_f1_scores[class_id]),
                "support": int(supports[class_id]),
            }
            for class_id in class_ids
        },
        # actual in rows, predicted in columns
        "confusion_matrix": {
            frame_names[int(true_id)]: {
                frame_names[int(predicted_id)]: int(confusion[true_id, predicted_id])
                for predicted_id in class_ids
            }
            for true_id in class_ids
        },
    }
