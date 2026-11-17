"""Module providing the main Contradictor pipeline factory function."""

from dataclasses import replace

from src.data_models.data_models import (
    ContradictorPipelinePreset,
    SearchPipelineImplementation,
)
from src.pipeline.config import PipelineConfiguration
from src.pipeline.contradictor_pipeline import ContradictorPipeline


def get_contradictor_pipeline(
    pipeline_preset: ContradictorPipelinePreset,
) -> ContradictorPipeline:
    """
    Get an unprepared instance of the main Contradictor pipeline.

    Args:
        pipeline_preset (ContradictorPipelinePreset): Preset of the main Contradictor
            pipeline. Different presents are used for different use cases.

    Returns:
        ContradictorPipeline: An unprepared instance of Contradictor pipeline.
    """
    # Local imports for lazy loading prevent importing heavy implementation
    # modules when they are not requested.
    # pylint: disable=import-outside-toplevel
    from src.pipeline.factories.argument_extractor_factory import (
        build_argument_extractor,
    )
    from src.pipeline.factories.argument_framer_factory import build_argument_framer
    from src.pipeline.factories.encoder_factory import build_encoder
    from src.pipeline.factories.nli_factory import build_nli
    from src.pipeline.factories.search_pipeline_factory import build_search_pipeline
    from src.pipeline.factories.style_extractor_factory import build_style_extractor
    from src.pipeline.factories.vector_search_factory import build_vector_search
    # pylint: enable=import-outside-toplevel

    config = PipelineConfiguration(
        search_pipeline=build_search_pipeline(),
        argument_extractor=build_argument_extractor(),
        argument_framer=build_argument_framer(),
        encoder=build_encoder(),
        nli=build_nli(),
        style_extractor=build_style_extractor(),
        vector_search=build_vector_search(),
    )

    match pipeline_preset:
        case ContradictorPipelinePreset.FOR_API:
            pass
        case ContradictorPipelinePreset.FOR_BENCHMARK:
            config = replace(
                config,
                search_pipeline=build_search_pipeline(
                    SearchPipelineImplementation.LOCAL_DATASET
                ),
            )

    return ContradictorPipeline(config)
