"""Module with an encoder factory function."""

from src.argument_encoding.argument_encoder import ArgumentEncoder
from src.argument_encoding.argument_encoding_dataset import ArgumentEncodingDataset
from src.configuration import config
from src.data_models.data_models import EncoderImplementation


def build_encoder(
    encoder_implementation: EncoderImplementation | None = None,
    output_dimensions: int = config.encoder_output_dimensions,
    **kwargs,
) -> ArgumentEncoder:
    """
    Construct a ready-made encoder from pre-configured implementations.

    Args:
        encoder_implementation (EncoderImplementation): Choice of a
            pre-configured implementation.
        output_dimensions (int): A number of dimensions of the representation vector
        kwargs: Additional parameters passed on directly to the constructor
            of an encoder.

    Returns:
        ArgumentEncoder: An initialised instance of the encoder.

    Raises:
        NotImplementedError: Raised if not implemented variant was requested.
    """
    encoder_implementation = encoder_implementation or config.encoder
    dataset = ArgumentEncodingDataset()

    match encoder_implementation:
        case EncoderImplementation.MLP:
            # pylint: disable=import-outside-toplevel
            from src.argument_encoding.mlp_encoder import MLPEncoder
            # pylint: enable=import-outside-toplevel

            return MLPEncoder(
                dataset=dataset,
                output_dimensions=output_dimensions,
                **kwargs,
            )
