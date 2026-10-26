"""Module for benchmarking Contradictor on the Golden Dataset."""

from pydantic import BaseModel, Field

from src.benchmark.golden_dataset import GoldenDataset
from src.data_models.data_models import Argument, FramedArgument


class BenchmarkMetrics(BaseModel):
    """Metrics achieved by Contradictor on benchmark against the Golden Dataset."""

    IoU_for_argument_extraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Intersection over union of the unique predicted and golden arguments. "
            "Arguments match when their evidence and claim text are identical."
        ),
    )
    argument_level_precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of predicted arguments that exactly match a golden argument."
        ),
    )
    argument_level_recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of golden arguments that were predicted.",
    )
    argument_level_f1: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Harmonic mean of argument-level precision and recall.",
    )
    pair_level_precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of predicted argument pairs that exactly match a golden pair."
        ),
    )
    pair_level_recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of golden argument pairs that were predicted.",
    )
    pair_level_f1: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Harmonic mean of pair-level precision and recall.",
    )


class Benchmark:
    """Benchmark calculating metrics on the Golden Dataset."""

    def __init__(self, golden_dataset: GoldenDataset) -> None:
        """
        Initialise a benchmark.

        Args:
            golden_dataset (GoldenDataset): A reference dataset to use for benchmarking.
        """
        self._golden_dataset = golden_dataset

    def calculate_metrics(
        self, predicted: list[tuple[FramedArgument, FramedArgument]]
    ) -> BenchmarkMetrics:
        """
        Calculate performance metrics based on the provided predictions.

        Args:
            predicted (list[tuple[FramedArgument, FramedArgument]]): List of pairs of
                framed arguments and counter-arguments.

        Returns:
            BenchmarkMetrics: Metrics achieved by the system that provided predictions.
        """
        predicted_arguments = {
            self._argument_key(argument) for pair in predicted for argument in pair
        }
        golden_arguments = {
            self._argument_key(argument) for argument in self._golden_dataset.arguments
        }

        argument_precision, argument_recall, argument_f1 = self._precision_recall_f1(
            predicted_arguments, golden_arguments
        )

        predicted_pairs = {self._pair_key(pair) for pair in predicted}
        golden_pairs = {
            self._pair_key(pair) for pair in self._golden_dataset.paired_arguments
        }

        pair_precision, pair_recall, pair_f1 = self._precision_recall_f1(
            predicted_pairs, golden_pairs
        )

        union = predicted_arguments | golden_arguments
        iou = len(predicted_arguments & golden_arguments) / len(union) if union else 1.0

        return BenchmarkMetrics(
            IoU_for_argument_extraction=iou,
            argument_level_precision=argument_precision,
            argument_level_recall=argument_recall,
            argument_level_f1=argument_f1,
            pair_level_precision=pair_precision,
            pair_level_recall=pair_recall,
            pair_level_f1=pair_f1,
        )

    def _argument_key(self, argument: Argument) -> tuple[str, str]:
        return (argument.evidence, argument.claim)

    def _pair_key(
        self, pair: tuple[Argument, Argument]
    ) -> tuple[tuple[str, str], tuple[str, str]]:
        left, right = sorted((self._argument_key(pair[0]), self._argument_key(pair[1])))
        return (left, right)

    def _precision_recall_f1(
        self, predicted_set: set, gold_set: set
    ) -> tuple[float, float, float]:
        true_positives = len(predicted_set & gold_set)
        precision = true_positives / len(predicted_set) if predicted_set else 0.0
        recall = true_positives / len(gold_set) if gold_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        return precision, recall, f1
