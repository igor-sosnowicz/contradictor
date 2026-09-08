"""Module with a high-level Contradictor pipeline implementation."""

import asyncio
from collections.abc import Collection, Iterable

import numpy as np
from loguru import logger

from src.configuration import config
from src.data_models.data_models import (
    Argument,
    ArgumentWithCounterarguments,
    ComputingBackend,
    FramedArgument,
    InterpretativeFrame,
    NLIResult,
)
from src.pipeline.config import PipelineConfiguration
from src.utils.errors import UnsupportedError


class ContradictorPipeline:
    """Full Contradictor pipeline defining data flow."""

    def __init__(
        self,
        pipes: PipelineConfiguration,
        backend: ComputingBackend | None = None,
    ) -> None:
        """
        Initialise pipeline elements and set the computing backend.

        Args:
            pipes (PipelineConfiguration): Stages (pipes) of the pipeline.
            backend (ComputingBackend | None): Computing backend for performing heavy
                calculation. Defaults to the default backend from the configuration.
        """
        self._pipes = pipes
        self._backend = backend or config.computing_backend

    async def prepare(self) -> None:
        """Prepare all components of the Contradictor pipeline."""
        await self._pipes.argument_framer.prepare()
        await self._pipes.style_extractor.prepare()

    def _get_arguments(self, reference_text: str) -> list[Argument]:
        arguments = self._pipes.argument_extractor([reference_text])[0]
        logger.debug(f"Extracted {len(arguments)} arguments from the reference text.")
        return arguments

    def _get_potential_counterarguments(self, reference_text: str) -> list[Argument]:
        other_texts = [
            document.text
            for document in self._pipes.search_pipeline.search(reference_text)
        ]

        # Flatten the list of counterarguments.
        counterarguments = [
            counterargument
            for document_arguments in self._pipes.argument_extractor(other_texts)
            for counterargument in document_arguments
        ]
        logger.debug(f"Extracted {len(counterarguments)} potential counterarguments.")
        return counterarguments

    async def _merge_claims(
        self, arguments: list[Argument], counterarguments: list[Argument]
    ) -> list[Argument]:
        if not arguments:
            return []

        claims_of_counterarguments: list[str] = [
            claim.claim for claim in counterarguments
        ]

        # A single batched call lets forward pass over all pairs.
        nli_results_by_argument = await self._pipes.nli(
            reference_texts=[argument.claim for argument in arguments],
            other_texts=claims_of_counterarguments,
        )

        merged_potential_counterarguments: list[Argument] = []
        for argument, nli_results in zip(
            arguments, nli_results_by_argument, strict=True
        ):
            merged = [
                Argument(
                    # Merge if entails.
                    claim=argument.claim
                    if nli_result == NLIResult.ENTAILMENT
                    # No merge in case of neutral or contradiction
                    else potential_counterargument.claim,
                    evidence=potential_counterargument.evidence,
                )
                for nli_result, potential_counterargument in zip(
                    nli_results, counterarguments, strict=True
                )
            ]
            merged_potential_counterarguments.extend(merged)

        counterargument_claims = {claim.claim for claim in counterarguments}
        merged_claims = {
            merged_argument.claim
            for merged_argument in merged_potential_counterarguments
        }
        # Distinct counterargument claims subsumed by an argument claim
        # disappear from the merged output.
        absorbed_claim_count = len(counterargument_claims - merged_claims)
        reduction_percentage = (
            (absorbed_claim_count / len(counterargument_claims) * 100)
            if counterargument_claims
            else 0.0
        )
        logger.debug(
            "Merging claims absorbed "
            f"{absorbed_claim_count} of {len(counterargument_claims)} "
            "distinct counterargument claims "
            f"({reduction_percentage:.2f}% reduction)."
        )

        return merged_potential_counterarguments

    def _filter_arguments_by_frame(
        self, frame: InterpretativeFrame, framed_arguments: Collection[FramedArgument]
    ) -> list[FramedArgument]:
        return [
            framed_argument
            for framed_argument in framed_arguments
            if framed_argument.primary_frame == frame
        ]

    async def _extract_style(
        self, framed_arguments: list[FramedArgument]
    ) -> list[np.ndarray]:
        return await self._pipes.style_extractor.extract(
            [str(argument) for argument in framed_arguments]
        )

    async def _match_counterarguments_to_arguments_by_frame(
        self,
        arguments: Collection[FramedArgument],
        counterarguments: Collection[FramedArgument],
        frame: InterpretativeFrame,
    ) -> list[ArgumentWithCounterarguments]:
        shared_frame_counterarguments = self._filter_arguments_by_frame(
            framed_arguments=counterarguments,
            frame=frame,
        )
        shared_frame_arguments = self._filter_arguments_by_frame(
            framed_arguments=arguments,
            frame=frame,
        )
        logger.debug(
            f"After filtering, there are {len(shared_frame_arguments)} arguments and "
            f"{len(shared_frame_counterarguments)} potential counterarguments."
        )

        if not shared_frame_arguments:
            return []

        logger.debug("Style vectors extraction started.")
        counterarguments_style = await self._extract_style(
            shared_frame_counterarguments
        )
        arguments_style = await self._extract_style(shared_frame_arguments)
        logger.debug("Style vectors extracted.")

        logger.debug("Representation encoding has started.")
        arguments_representation = await self._pipes.encoder(
            shared_frame_arguments, arguments_style
        )
        counterarguments_representation = await self._pipes.encoder(
            shared_frame_counterarguments, counterarguments_style
        )
        logger.debug("Representation encoding has finished.")

        logger.debug("Matching counterarguments with arguments has started.")
        indices_by_argument = await self._pipes.vector_search.max_distance(
            references=arguments_representation,
            other=counterarguments_representation,
        )
        output = [
            ArgumentWithCounterarguments(
                argument=argument,
                counterarguments=tuple(
                    shared_frame_counterarguments[index] for index in indices
                ),
            )
            for argument, indices in zip(
                shared_frame_arguments, indices_by_argument, strict=True
            )
        ]
        logger.debug("Matching counterarguments with arguments has finished.")

        return output

    async def _process_all_frames(
        self,
        frames: Iterable[InterpretativeFrame],
        framed_arguments: Collection[FramedArgument],
        framed_counterarguments: Collection[FramedArgument],
    ) -> list[ArgumentWithCounterarguments]:
        # One coroutine for each frame
        tasks = [
            self._match_counterarguments_to_arguments_by_frame(
                arguments=framed_arguments,
                counterarguments=framed_counterarguments,
                frame=frame,
            )
            for frame in frames
        ]
        results = await asyncio.gather(*tasks)

        # Flatten the list of lists into a single output list.
        output: list[ArgumentWithCounterarguments] = []
        for result in results:
            output.extend(result)

        return output

    def sanitise_input(self, text: str) -> str:
        """
        Sanitise the input text.

        Args:
            text (str): Text to be sanitised.

        Returns:
            str: Sanitised version of the text.

        Raises:
            ValueError: Raised if the text is empty or made of whitespaces only.
        """
        reference_text = text.strip()
        if not reference_text:
            raise ValueError("Reference text cannot be empty.")
        if len(reference_text) > config.reference_text_max_length:
            logger.warning(
                "The reference text is too long. Truncating it to "
                f"{config.reference_text_max_length} characters."
            )
            reference_text = reference_text[: config.reference_text_max_length]

        return reference_text

    async def __call__(
        self,
        reference_text: str,
    ) -> list[ArgumentWithCounterarguments]:
        """
        Pass a reference text through the pipeline.

        Args:
            reference_text (str): A text from the user to match counterarguments to
                arguments found in the text.

        Returns:
            list[ArgumentWithCounterarguments]: A list of arguments and
                matching counterarguments.

        Raises:
            ValueError: Raised if reference text is empty.
            UnsupportedError: Raised if a currently unsupported
                computing backend was used.
        """
        reference_text = self.sanitise_input(reference_text)

        logger.info("Starting the pipeline...")

        arguments = self._get_arguments(reference_text)
        potential_counterarguments = self._get_potential_counterarguments(
            reference_text
        )

        potential_counterarguments = await self._merge_claims(
            arguments=arguments, counterarguments=potential_counterarguments
        )

        match self._backend:
            case ComputingBackend.CPU:
                framed_arguments = self._pipes.argument_framer.frame(arguments)
                framed_potential_counterarguments = self._pipes.argument_framer.frame(
                    potential_counterarguments
                )
            case _:
                raise UnsupportedError(
                    "Framing does not support computing backends other than CPU."
                )

        matched_arguments = await self._process_all_frames(
            frames=tuple(InterpretativeFrame),
            framed_arguments=framed_arguments,
            framed_counterarguments=framed_potential_counterarguments,
        )
        logger.info("Matching counterarguments to arguments completed.")
        return matched_arguments
