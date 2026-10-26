"""Module with a command for benchmarking Contradictor."""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.benchmark.benchmark import Benchmark, BenchmarkMetrics
from src.benchmark.golden_dataset import GoldenDatasetLoader
from src.configuration import config
from src.data_models.data_models import FramedArgument, SearchPipelineImplementation
from src.pipeline.config import PipelineConfiguration
from src.pipeline.contradictor_pipeline import ContradictorPipeline
from src.pipeline.factories.argument_extractor_factory import build_argument_extractor
from src.pipeline.factories.argument_framer_factory import build_argument_framer
from src.pipeline.factories.encoder_factory import build_encoder
from src.pipeline.factories.nli_factory import build_nli
from src.pipeline.factories.search_pipeline_factory import build_search_pipeline
from src.pipeline.factories.style_extractor_factory import build_style_extractor
from src.pipeline.factories.vector_search_factory import build_vector_search

BENCHMARK_RESULTS_DIR: Path = config.data_directory / "benchmark_results"


async def perform_benchmark() -> BenchmarkMetrics:
    """
    Perform a benchmark of Contradictor against the Golden Dataset.

    Returns:
        BenchmarkMetrics: Metrics achieved by Contradictor.
    """
    pipeline = ContradictorPipeline(
        pipes=PipelineConfiguration(
            search_pipeline=build_search_pipeline(
                SearchPipelineImplementation.LOCAL_DATASET
            ),
            argument_extractor=build_argument_extractor(),
            argument_framer=build_argument_framer(),
            encoder=build_encoder(),
            nli=build_nli(),
            style_extractor=build_style_extractor(),
            vector_search=build_vector_search(),
        ),
    )

    await pipeline.prepare()

    golden_dataset = GoldenDatasetLoader().load()
    benchmark = Benchmark(golden_dataset)

    predictions: list[tuple[FramedArgument, FramedArgument]] = []
    for text in golden_dataset.raw_texts.values():
        results = await pipeline(text)
        for result in results:
            predictions.extend(
                (result.argument, counterargument)
                for counterargument in result.counterarguments
            )

    metrics = benchmark.calculate_metrics(predictions)
    _save_results(contradictor=pipeline, metrics=metrics)
    return metrics


def _save_results(
    contradictor: ContradictorPipeline, metrics: BenchmarkMetrics
) -> None:
    """
    Save results of the benchmark run.

    Args:
        contradictor (ContradictorPipeline): Pipeline used to achieve provided metrics.
        metrics (BenchmarkMetrics): Metrics achieved during the benchmark on given
            a dataset.
    """
    filename = datetime.now().astimezone().strftime("%H_%M_%S_%d_%m_%Y") + ".json"
    path = (BENCHMARK_RESULTS_DIR / filename).expanduser().resolve()

    results = {
        "metrics": metrics.model_dump_json(),
        "configuration": contradictor.get_configuration(),
    }

    with path.open("rt") as file:
        json.dump(results, fp=file, indent=2)

    logger.info(f"Saved the results of the benchmark run to {path}")
