"""Tests for the `benchmark` command."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import Popen
from unittest.mock import AsyncMock, MagicMock

import pytest
from anyio import Path as AsyncPath

from src.benchmark.benchmark import BenchmarkMetrics
from tests.test_benchmark.utils import load_latest_benchmark_result, validate_result


def _make_metrics() -> BenchmarkMetrics:
    return BenchmarkMetrics(
        IoU_for_argument_extraction=1.0,
        argument_level_precision=1.0,
        argument_level_recall=1.0,
        argument_level_f1=1.0,
        pair_level_precision=1.0,
        pair_level_recall=1.0,
        pair_level_f1=1.0,
    )


_TEST_DATASET_VERSION = "0.1"


def test_save_results_writes_expected_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that `_save_results` writes a result file with the expected structure."""
    import src.benchmark.command as command_module

    monkeypatch.setattr(command_module, "BENCHMARK_RESULTS_DIR", tmp_path)

    pipeline = MagicMock()
    pipeline.get_configuration.return_value = {"search_pipeline": "LOCAL_DATASET"}

    metrics = _make_metrics()

    command_module._save_results(
        contradictor=pipeline,
        metrics=metrics,
        golden_dataset_version=_TEST_DATASET_VERSION,
        start_datetime=datetime.now().astimezone(),
    )

    files = list(tmp_path.iterdir())
    assert len(files) == 1

    result = json.loads(files[0].read_text())
    validate_result(result)
    assert result["configuration"] == {"search_pipeline": "LOCAL_DATASET"}
    assert result["metadata"]["golden_dataset_version"] == _TEST_DATASET_VERSION


async def test_perform_benchmark_returns_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test `perform_benchmark` orchestrates pipeline and metrics without heavy ML."""
    import src.benchmark.command as command_module

    monkeypatch.setattr(command_module, "BENCHMARK_RESULTS_DIR", tmp_path)

    expected_metrics = _make_metrics()

    mock_pipeline = MagicMock()
    mock_pipeline.prepare = AsyncMock()
    mock_pipeline.__call__ = AsyncMock(return_value=[])
    mock_pipeline.get_configuration = MagicMock(
        return_value={"search_pipeline": "LOCAL_DATASET"}
    )

    mock_dataset = MagicMock()
    mock_dataset.raw_texts = {}
    mock_dataset.version = _TEST_DATASET_VERSION

    mock_benchmark = MagicMock()
    mock_benchmark.calculate_metrics.return_value = expected_metrics

    monkeypatch.setattr(
        command_module, "ContradictorPipeline", lambda pipes: mock_pipeline
    )
    monkeypatch.setattr(
        command_module.GoldenDatasetLoader, "load", lambda self: mock_dataset
    )
    monkeypatch.setattr(
        command_module, "Benchmark", lambda golden_dataset: mock_benchmark
    )

    metrics = await command_module.perform_benchmark()

    assert metrics == expected_metrics

    files = [f async for f in AsyncPath(tmp_path).iterdir()]
    assert len(files) == 1
    result = json.loads(await files[0].read_text())
    validate_result(result)
    assert result["metadata"]["golden_dataset_version"] == _TEST_DATASET_VERSION
    assert isinstance(result["metadata"]["duration"], (int, float))
    assert result["metadata"]["duration"] >= 0


def test_save_results_requires_all_metadata_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that `_save_results` writes all required metadata fields."""
    import src.benchmark.command as command_module

    monkeypatch.setattr(command_module, "BENCHMARK_RESULTS_DIR", tmp_path)

    pipeline = MagicMock()
    pipeline.get_configuration.return_value = {"search_pipeline": "LOCAL_DATASET"}

    metrics = _make_metrics()
    start_datetime = datetime.now().astimezone()

    command_module._save_results(
        contradictor=pipeline,
        metrics=metrics,
        golden_dataset_version="0.2",
        start_datetime=start_datetime,
    )

    files = list(tmp_path.iterdir())
    assert len(files) == 1

    result = json.loads(files[0].read_text())
    validate_result(result)
    assert result["metadata"]["golden_dataset_version"] == "0.2"
    assert result["metadata"]["start_datetime"] == start_datetime.isoformat()
    assert isinstance(result["metadata"]["finish_datetime"], str)
    assert isinstance(result["metadata"]["duration"], (int, float))
    assert result["metadata"]["duration"] >= 0


@pytest.mark.cli
def test_calling_benchmark_command() -> None:
    """Test running a benchmark command."""
    timeout_in_seconds = timedelta(minutes=30).total_seconds()
    args = ["uv", "run", "contradictor", "benchmark"]

    process = Popen(args)  # noqa: S603, The command implemented by authors.
    return_code = process.wait(timeout_in_seconds)

    assert return_code == 0

    result = load_latest_benchmark_result()
    validate_result(result)
