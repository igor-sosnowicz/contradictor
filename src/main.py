"""A module being an entrypoint to the entire application."""

import asyncio

from src.pipeline.config import PipelineConfiguration
from src.pipeline.contradictor_pipeline import ContradictorPipeline
from src.pipeline.factories.argument_extractor_factory import build_argument_extractor
from src.pipeline.factories.argument_framer_factory import build_argument_framer
from src.pipeline.factories.encoder_factory import build_encoder
from src.pipeline.factories.nli_factory import build_nli
from src.pipeline.factories.search_pipeline_factory import build_search_pipeline
from src.pipeline.factories.style_extractor_factory import build_style_extractor
from src.pipeline.factories.vector_search_factory import build_vector_search


async def main() -> None:
    """Run the application."""
    text = """
    The for statement in Python differs a bit from what you may be used to in
    C or Pascal. Rather than always iterating over an arithmetic progression
    of numbers (like in Pascal), or giving the user the ability to define both
    the iteration step and halting condition (as C), Python's for statement
    iterates over the items of any sequence (a list or a string),
    in the order that they appear in the sequence.
    """.strip()

    pipeline = ContradictorPipeline(
        pipes=PipelineConfiguration(
            search_pipeline=build_search_pipeline(),
            argument_extractor=build_argument_extractor(),
            argument_framer=build_argument_framer(),
            encoder=build_encoder(),
            nli=build_nli(),
            style_extractor=build_style_extractor(),
            vector_search=build_vector_search(),
        ),
    )
    await pipeline.prepare()

    results = await pipeline(text)
    print(results)  # noqa: T201, Temporary to show off the results.


if __name__ == "__main__":
    asyncio.run(main())
