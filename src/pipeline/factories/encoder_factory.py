"""Module with an encoder factory function."""

from src.configuration import config
from src.data_models.data_models import EncoderImplementation
from src.pipeline.encoder import Encoder


def build_encoder(
    encoder_implementation: EncoderImplementation | None = None,
) -> Encoder:
    """
    Construct a ready-made encoder from pre-configured implementations.

    Args:
        encoder_implementation (EncoderImplementation): Choice of a
            pre-configured implementation.

    Returns:
        Encoder: An initialised instance of the encoder.

    Raises:
        NotImplementedError: Raised if not implemented variant was requested.
    """
    encoder_implementation = encoder_implementation or config.encoder

    match encoder_implementation:
        case EncoderImplementation.NOT_IMPLEMENTED:
            raise NotImplementedError("No encoder is implemented yet.")
