"""Module with a command for benchmarking Contradictor."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from src.benchmark.benchmark import Benchmark, BenchmarkMetrics
from src.benchmark.golden_dataset import GoldenDatasetLoader
from src.configuration import config
from src.data_models.data_models import (
    ContradictorPipelinePreset,
    FramedArgument,
)
from src.pipeline.contradictor_pipeline import ContradictorPipeline
from src.pipeline.factories.contradictor_pipeline_factory import (
    get_contradictor_pipeline,
)
from src.utils.logger import logger

BENCHMARK_RESULTS_DIR: Path = config.data_directory / "benchmark_results"


async def perform_benchmark() -> BenchmarkMetrics:
    """
    Perform a benchmark of Contradictor against the Golden Dataset.

    Returns:
        BenchmarkMetrics: Metrics achieved by Contradictor.
    """
    start_datetime = datetime.now().astimezone()

    pipeline = get_contradictor_pipeline(
        pipeline_preset=ContradictorPipelinePreset.FOR_BENCHMARK
    )
    await pipeline.prepare()

    golden_dataset = GoldenDatasetLoader().load()
    benchmark = Benchmark(golden_dataset)

    predictions: list[tuple[FramedArgument, FramedArgument]] = []
    for text in golden_dataset.raw_texts.values():
        results = await pipeline(text)
        for result in results.pairs:
            predictions.extend(
                (result.argument, counterargument)
                for counterargument in result.counterarguments
            )

    metrics = benchmark.calculate_metrics(predictions)
    _save_results(
        contradictor=pipeline,
        metrics=metrics,
        golden_dataset_version=golden_dataset.version,
        start_datetime=start_datetime,
    )
    return metrics


def _save_results(
    contradictor: ContradictorPipeline,
    metrics: BenchmarkMetrics,
    golden_dataset_version: str,
    start_datetime: datetime,
) -> None:
    """
    Save results of the benchmark run.

    Args:
        contradictor (ContradictorPipeline): Pipeline used to achieve provided metrics.
        metrics (BenchmarkMetrics): Metrics achieved during the benchmark on given
            a dataset.
        golden_dataset_version (str): The version of a golden dataset.
        start_datetime (datetime): Moment when the benchmark run started.
    """
    finish_datetime = datetime.now().astimezone()
    duration_seconds = (finish_datetime - start_datetime).total_seconds()
    datetime_string = finish_datetime.strftime("%H_%M_%S_%d_%m_%Y")
    _uuid = str(uuid.uuid4())
    filename = f"{datetime_string}_{_uuid}.json"
    path = (BENCHMARK_RESULTS_DIR / filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metrics": metrics.model_dump(),
        "configuration": contradictor.get_configuration(),
        "metadata": {
            "start_datetime": start_datetime.isoformat(),
            "finish_datetime": finish_datetime.isoformat(),
            "duration": duration_seconds,
            "golden_dataset_version": golden_dataset_version,
        },
    }

    with path.open("wt") as file:
        json.dump(
            results,
            fp=file,
            indent=2,
            default=str,
        )

    logger.info(f"Saved the results of the benchmark run to {path}")
