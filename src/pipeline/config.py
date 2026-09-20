"""Module with the Contradictor pipeline configuration."""

from dataclasses import dataclass

from src.argument_framing.argument_framer import ArgumentFramer
from src.nli.nli import NLI
from src.pipeline.argument_extractor import ArgumentExtractor
from src.pipeline.encoder import Encoder
from src.pipeline.vector_search import VectorSearch
from src.search_module.pipeline import SearchPipeline
from src.style_extraction.style_extractor import StyleExtractor


@dataclass(frozen=True)
class PipelineConfiguration:
    """Pipes of the main pipeline."""

    search_pipeline: SearchPipeline
    argument_extractor: ArgumentExtractor
    argument_framer: ArgumentFramer
    nli: NLI
    style_extractor: StyleExtractor
    encoder: Encoder
    vector_search: VectorSearch
