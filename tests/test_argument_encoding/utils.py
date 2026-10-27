"""Module with utilities for testing argument encoders."""

import faker

from src.argument_encoding.argument_encoder import ArgumentEncoder
from src.argument_encoding.mlp_encoder.config import TrainingConfig
from src.data_models.data_models import (
    Argument,
    EncoderImplementation,
    StyledFramedArgument,
)
from src.pipeline.factories.argument_framer_factory import build_argument_framer
from src.pipeline.factories.encoder_factory import build_encoder
from src.pipeline.factories.style_extractor_factory import build_style_extractor
from src.utils.timer import Timer

FAST_TRAINING_CONFIG = TrainingConfig(
    epochs=2,
    batch_size=32,
    learning_rate=1e-3,
    patience=1,
)


def get_argument_encoders(
    output_dimensions: int | None = None,
) -> list[ArgumentEncoder]:
    """
    Get an instance of every implementation of an argument encoder.

    Args:
        output_dimensions (int | None): A number of outputs of the encoder.

    Returns:
        list[ArgumentEncoder]: A list of instances of argument encoders.
    """
    encoders: list[ArgumentEncoder] = []
    for encoder in EncoderImplementation:
        kwargs = {
            "training_config": FAST_TRAINING_CONFIG,
            "fine_tuning_config": FAST_TRAINING_CONFIG,
            "n_optuna_trials": 2,
            "proof_of_concept_mode": True,
        }
        if output_dimensions is None:
            encoders.append(build_encoder(encoder, **kwargs))  # type: ignore[arg-type]
        else:
            encoders.append(
                build_encoder(encoder, output_dimensions=output_dimensions, **kwargs)
            )
    return encoders


async def get_random_styled_and_framed_texts(n: int) -> list[StyledFramedArgument]:
    """
    Get `n` random styled arguments with a frame assigned to them.

    These are not arguments because their content is random text.
    However, they occupy a DTO called `Argument`.

    Args:
        n (int): A number of "arguments".

    Returns:
        list[StyledFramedArgument]: List of `n` styled and framed "arguments".
    """
    fake = faker.Faker()
    fake.seed_instance(1)

    with Timer("style_extractor created"):
        style_extractor = build_style_extractor()

    with Timer("style_extractor.prepare"):
        await style_extractor.prepare()

    with Timer("framer created"):
        framer = build_argument_framer(proof_of_concept_mode=True)

    with Timer("framer.prepare"):
        await framer.prepare()

    with Timer(f"{n} fake arguments created"):
        arguments = [
            Argument(claim=fake.paragraph(), evidence=fake.paragraph())
            for _ in range(n)
        ]

    with Timer(f"framer.frame({n})", log_level="INFO"):
        framed_arguments = framer.frame(arguments)

    with Timer(
        f"style_extractor.extract({len(framed_arguments)} texts)",
        log_level="INFO",
    ):
        style_vectors = await style_extractor.extract(
            [str(argument) for argument in framed_arguments]
        )
    return [
        StyledFramedArgument(
            claim=argument.claim,
            evidence=argument.evidence,
            primary_frame=argument.primary_frame,
            frame_probabilities=argument.frame_probabilities,
            style_vector=style_vector,
        )
        for argument, style_vector in zip(framed_arguments, style_vectors, strict=True)
    ]
