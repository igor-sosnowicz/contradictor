"""Module with utilities for testing argument encoders."""

import time

import faker
from loguru import logger

from src.argument_encoding.argument_encoder import ArgumentEncoder
from src.argument_encoding.argument_encoding_dataset import ArgumentEncodingDataset
from src.argument_encoding.mlp_encoder import TrainingConfig
from src.data_models.data_models import (
    Argument,
    EncoderImplementation,
    StyledFramedArgument,
)
from src.pipeline.factories.argument_framer_factory import build_argument_framer
from src.pipeline.factories.encoder_factory import build_encoder
from src.pipeline.factories.style_extractor_factory import build_style_extractor

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
    tiny_dataset = ArgumentEncodingDataset(max_samples=10)
    for encoder in EncoderImplementation:
        kwargs = {
            "training_config": FAST_TRAINING_CONFIG,
            "fine_tuning_config": FAST_TRAINING_CONFIG,
            "n_optuna_trials": 2,
            "dataset": tiny_dataset,
        }
        if output_dimensions is None:
            encoders.append(build_encoder(encoder, **kwargs))
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

    t0 = time.time()
    style_extractor = build_style_extractor()
    logger.debug(f"[PERF] style_extractor created: {time.time() - t0:.1f}s")

    t0 = time.time()
    await style_extractor.prepare()
    logger.debug(f"[PERF] style_extractor.prepare: {time.time() - t0:.1f}s")

    t0 = time.time()
    framer = build_argument_framer()
    logger.debug(f"[PERF] framer created: {time.time() - t0:.1f}s")

    t0 = time.time()
    await framer.prepare()
    logger.debug(f"[PERF] framer.prepare: {time.time() - t0:.1f}s")

    arguments = [
        Argument(claim=fake.paragraph(), evidence=fake.paragraph()) for _ in range(n)
    ]
    logger.debug(f"[PERF] {n} fake arguments created: {time.time() - t0:.1f}s")

    t0 = time.time()
    framed_arguments = framer.frame(arguments)
    logger.info(
        f"[PERF] framer.frame({len(framed_arguments)}): {time.time() - t0:.1f}s"
    )

    t0 = time.time()
    style_vectors = await style_extractor.extract(
        [str(argument) for argument in framed_arguments]
    )
    logger.info(
        f"[PERF] style_extractor.extract({len(framed_arguments)} texts): "
        f"{time.time() - t0:.1f}s"
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
