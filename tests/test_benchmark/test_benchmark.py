"""Tests for the benchmarking mechanism."""

import pytest

from src.benchmark.benchmark import Benchmark, BenchmarkMetrics
from src.benchmark.golden_dataset import GoldenDataset
from src.data_models.data_models import FramedArgument, InterpretativeFrame

type ContradictorOutput = list[tuple[FramedArgument, FramedArgument]]


def _framed_argument(claim: str, evidence: str) -> FramedArgument:
    return FramedArgument(
        claim=claim,
        evidence=evidence,
        primary_frame=InterpretativeFrame.ECONOMIC,
        frame_probabilities=dict.fromkeys(InterpretativeFrame, 0.0),
    )


_ARG_A = _framed_argument("Claim A", "Evidence A")
_ARG_B = _framed_argument("Claim B", "Evidence B")
_ARG_C = _framed_argument("Claim C", "Evidence C")
_ARG_D = _framed_argument("Claim D", "Evidence D")
_ARG_E = _framed_argument("Claim E", "Evidence E")
_ARG_F = _framed_argument("Claim F", "Evidence F")
_ARG_G = _framed_argument("Claim G", "Evidence G")
_ARG_H = _framed_argument("Claim H", "Evidence H")


@pytest.fixture
def perfect_case() -> tuple[GoldenDataset, BenchmarkMetrics, ContradictorOutput]:
    """Perfect prediction — every metric is exactly 1.0."""
    dataset = GoldenDataset(
        raw_texts={},
        version="0.1",
        arguments=[_ARG_A, _ARG_B],
        paired_arguments=[(_ARG_A, _ARG_B)],
    )
    predicted: ContradictorOutput = [(_ARG_A, _ARG_B)]
    metrics = BenchmarkMetrics(
        IoU_for_argument_extraction=1.0,
        argument_level_precision=1.0,
        argument_level_recall=1.0,
        argument_level_f1=1.0,
        pair_level_precision=1.0,
        pair_level_recall=1.0,
        pair_level_f1=1.0,
    )
    return dataset, metrics, predicted


@pytest.fixture
def partial_case() -> tuple[GoldenDataset, BenchmarkMetrics, ContradictorOutput]:
    """
    Partial prediction — every metric lands on a non-trivial value.

    Golden args   = {A, B, C, D, E}   Golden pairs   = {AB, DE}
    Predicted     = [(A, B), (C, F)]
    Pred args         = {A, B, C, F}
    Pred pairs        = {AB, CF}

    IoU                              = 3 / 6             = 0.5
    argument_level_precision         = 3 / 4             = 0.75
    argument_level_recall            = 3 / 5             = 0.6
    argument_level_f1                = 2 * 0.75 * 0.6 / (0.75 + 0.6) = 2 / 3
    pair_level_precision             = 1 / 2             = 0.5
    pair_level_recall                = 1 / 2             = 0.5
    pair_level_f1                    = 0.5
    """
    dataset = GoldenDataset(
        raw_texts={},
        version="0.1",
        arguments=[_ARG_A, _ARG_B, _ARG_C, _ARG_D, _ARG_E],
        paired_arguments=[(_ARG_A, _ARG_B), (_ARG_D, _ARG_E)],
    )
    predicted: ContradictorOutput = [
        (_ARG_A, _ARG_B),
        (_ARG_C, _ARG_F),
    ]
    metrics = BenchmarkMetrics(
        IoU_for_argument_extraction=0.5,
        argument_level_precision=0.75,
        argument_level_recall=0.6,
        argument_level_f1=2 / 3,
        pair_level_precision=0.5,
        pair_level_recall=0.5,
        pair_level_f1=0.5,
    )
    return dataset, metrics, predicted


@pytest.fixture
def mixed_case() -> tuple[GoldenDataset, BenchmarkMetrics, ContradictorOutput]:
    """
    Mixed prediction — precision != recall, all metrics non-trivial.

    Golden args   = {A, B, C, D, E, F} Golden pairs   = {AB, CD, EF}
    Predicted     = [(A, B), (G, H)]
    Pred args         = {A, B, G, H}
    Pred pairs        = {AB, GH}

    IoU                              = 2 / 8             = 0.25
    argument_level_precision         = 2 / 4             = 0.5
    argument_level_recall            = 2 / 6             = 1 / 3
    argument_level_f1                = 2 * 0.5 * (1/3) / (0.5 + 1/3) = 0.4
    pair_level_precision             = 1 / 2             = 0.5
    pair_level_recall                = 1 / 3             = 1 / 3
    pair_level_f1                    = 2 * 0.5 * (1/3) / (0.5 + 1/3) = 0.4
    """
    dataset = GoldenDataset(
        raw_texts={},
        version="0.1",
        arguments=[_ARG_A, _ARG_B, _ARG_C, _ARG_D, _ARG_E, _ARG_F],
        paired_arguments=[(_ARG_A, _ARG_B), (_ARG_C, _ARG_D), (_ARG_E, _ARG_F)],
    )
    predicted: ContradictorOutput = [
        (_ARG_A, _ARG_B),
        (_ARG_G, _ARG_H),
    ]
    metrics = BenchmarkMetrics(
        IoU_for_argument_extraction=0.25,
        argument_level_precision=0.5,
        argument_level_recall=1 / 3,
        argument_level_f1=0.4,
        pair_level_precision=0.5,
        pair_level_recall=1 / 3,
        pair_level_f1=0.4,
    )
    return dataset, metrics, predicted


@pytest.fixture(
    params=["perfect_case", "partial_case", "mixed_case"],
    ids=["perfect_case", "partial_case", "mixed_case"],
)
def golden_dataset_case(
    request: pytest.FixtureRequest,
) -> tuple[GoldenDataset, BenchmarkMetrics, ContradictorOutput]:
    """Parametrises the test over each separate case fixture."""
    return request.getfixturevalue(request.param)


def test_calculating_metrics(
    golden_dataset_case: tuple[GoldenDataset, BenchmarkMetrics, ContradictorOutput],
) -> None:
    """Test calculating benchmark metrics against hand-computed expectations."""
    golden_dataset, expected_metrics, predicted = golden_dataset_case
    benchmark = Benchmark(golden_dataset)
    calculated_metrics = benchmark.calculate_metrics(predicted=predicted)

    assert calculated_metrics.model_dump() == pytest.approx(
        expected_metrics.model_dump()
    )
