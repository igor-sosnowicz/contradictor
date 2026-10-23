"""Module with an argument encoding dataset."""

import shutil
import time
from pathlib import Path
from typing import cast, override

import numpy as np
import pandas as pd
from datasets import load_dataset
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from src.configuration import config
from src.data_models.abstract.dataset import Dataset
from src.data_models.data_models import (
    Argument,
    InterpretativeFrame,
    NLIPrediction,
    NLIResult,
    StyledFramedArgument,
    SubsetName,
)
from src.nli.nli import NLI
from src.nli.transformers_nli import TransformersNLI
from src.pipeline.factories.argument_framer_factory import build_argument_framer
from src.style_extraction.spacy_style_extractor import SpacyStyleExtractor
from src.utils.errors import NotPreparedError


class ArgumentEncodingDataset(Dataset):
    """The dataset for training an argument encoder."""

    MAX_SAMPLES_DEFAULT: int = 100

    _sbert_model: SentenceTransformer | None = None

    def __init__(
        self,
        max_samples: int = MAX_SAMPLES_DEFAULT,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialise the argument encoding dataset.

        Args:
            max_samples (int): Maximum number of FEVER arguments to load.
                In every split of the dataset, arguments create N^2 rows.
                For example, training split with 10 samples results in 100 rows.
            random_seed (int | None): Random seed for argument splitting.
        """
        super().__init__(random_seed=random_seed)
        self.max_samples = max_samples

    @classmethod
    def _sbert(cls) -> SentenceTransformer:
        """Lazy-load the sentence-transformers model for semantic similarity."""
        if cls._sbert_model is None:
            cls._sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._sbert_model

    @staticmethod
    def _calculate_distance(
        style_vector_1: np.ndarray,
        style_vector_2: np.ndarray,
        argument_agreement: NLIPrediction,
        text_1: str,
        text_2: str,
    ) -> float:
        """
        Calculate the expected distance between two arguments.

        Semantics: contradiction = far (1.0), entailment = close (0.0),
        neutral = different claim / frame: high semantic similarity means
        same topic but opposing view (counter-argument → far), low similarity
        means different topics → distance tracks style.
        """
        epsilon = 10e-7

        # Calculate a style-based cosine distance.
        denominator = (
            np.linalg.norm(style_vector_1) * np.linalg.norm(style_vector_2)
        ) or epsilon
        numerator = np.dot(style_vector_1, style_vector_2)
        style_based_distance = 1.0 - numerator / denominator

        # Semantics: contradiction = far (1.0), entailment = close (0.0),
        # neutral = different claim / frame: high semantic similarity means
        # same topic but opposing view (counter-argument → far), low similarity
        # means different topics → distance tracks style.
        if argument_agreement.result == NLIResult.ENTAILMENT:
            semantics_based_distance = 0.0
        elif argument_agreement.result == NLIResult.CONTRADICTION:
            semantics_based_distance = 1.0
        else:  # NEUTRAL
            t0 = time.time()
            embeddings = ArgumentEncodingDataset._sbert().encode(
                [text_1, text_2], normalize_embeddings=True
            )
            logger.debug(f"[PERF] _sbert().encode (2 texts): {time.time() - t0:.1f}s")
            semantics_based_distance = float(np.dot(embeddings[0], embeddings[1]))

        semantics_weight = 0.7
        style_weight = 0.3
        return round(
            style_weight * style_based_distance
            + semantics_weight * semantics_based_distance,
            ndigits=5,
        )

    @staticmethod
    async def _generate_argument_encoding_dataset(
        arguments: list[StyledFramedArgument],
        nli: NLI,
    ) -> pd.DataFrame:
        """
        Generate a pairwise argument encoding dataset.

        Args:
            arguments (list[StyledFramedArgument]): Framed arguments with style vectors.
            nli (NLI): Natural language inference model.

        Returns:
            pd.DataFrame: Pairwise dataset with expected distances.
        """
        data: list[dict] = []
        reference_texts = [str(argument) for argument in arguments]
        other_texts = reference_texts
        t0 = time.time()
        agreements = await nli(reference_texts, other_texts)
        logger.info(
            f"[PERF] nli({len(reference_texts)}x{len(other_texts)} pairs): "
            f"{time.time() - t0:.1f}s"
        )

        for i, argument_1 in enumerate(arguments):
            for j, argument_2 in enumerate(arguments):
                argument_agreement = agreements[i][j]

                data.append(
                    {
                        # argument 1
                        "claim_1": argument_1.claim,
                        "evidence_1": argument_1.evidence,
                        "style_vector_1": argument_1.style_vector,
                        "frame_1": argument_1.primary_frame,
                        # argument 2
                        "claim_2": argument_2.claim,
                        "evidence_2": argument_2.evidence,
                        "style_vector_2": argument_2.style_vector,
                        "frame_2": argument_2.primary_frame,
                        # pairwise label
                        "expected_distance": (
                            ArgumentEncodingDataset._calculate_distance(
                                argument_agreement=argument_agreement,
                                style_vector_1=argument_1.style_vector,
                                style_vector_2=argument_2.style_vector,
                                text_1=reference_texts[i],
                                text_2=other_texts[j],
                            )
                        ),
                    }
                )

        return pd.DataFrame(data)

    @staticmethod
    def load_fever_supports_pairs(
        max_samples: int = MAX_SAMPLES_DEFAULT,
    ) -> list[Argument]:
        """Load FEVER dataset and return claim-evidence pairs labeled SUPPORTS."""
        ds = load_dataset("pietrolesci/nli_fever", split="train")
        pairs: list[Argument] = []
        for row in ds:
            if row["fever_gold_label"] != "SUPPORTS":
                continue
            pairs.append(
                Argument(
                    claim=row["hypothesis"],
                    evidence=row["premise"],
                )
            )
            if len(pairs) >= max_samples:
                break
        return pairs

    @property
    @override
    def processed_folder_name(self) -> str:
        return "argument_encoding"

    def _save_style_vectors(
        self,
        texts: list[str],
        style_vectors: list[np.ndarray],
        path: Path,
    ) -> None:
        style_vector_store_path = path / "style_vectors.npz"
        np.savez(
            style_vector_store_path,
            texts=np.array(texts),
            vectors=np.stack(style_vectors),
        )

    def _save_frame_probabilities(
        self,
        styled_arguments: list[StyledFramedArgument],
        path: Path,
    ) -> None:
        """Index frame probabilities by argument text and save to npz."""
        frame_prob_store_path = path / "frame_probabilities.npz"
        frame_order = list(InterpretativeFrame)
        frame_prob_texts = []
        frame_prob_matrix = []
        for argument in styled_arguments:
            probs = argument.frame_probabilities
            frame_prob_texts.append(str(argument))
            frame_prob_matrix.append(
                [float(probs.get(frame, 0.0)) for frame in frame_order]
            )

        np.savez(
            frame_prob_store_path,
            frame_prob_texts=np.array(frame_prob_texts),
            frame_prob_matrix=np.array(frame_prob_matrix, dtype=np.float32),
        )

    async def _get_styled_arguments_from_fever_dataset(
        self,
    ) -> tuple[
        list[StyledFramedArgument],
        list[str],
        list[np.ndarray],
    ]:
        t0 = time.time()
        arguments_data = self.load_fever_supports_pairs(self.max_samples)
        logger.info(
            f"[PERF] load_fever_supports_pairs({self.max_samples}): "
            f"{time.time() - t0:.1f}s"
        )
        texts = [str(argument) for argument in arguments_data]

        t0 = time.time()
        style_extractor = SpacyStyleExtractor()
        await style_extractor.prepare()
        logger.info(
            f"[PERF] style_extractor.prepare (dataset): {time.time() - t0:.1f}s"
        )

        t0 = time.time()
        style_vectors = await style_extractor.extract(texts)
        logger.info(
            f"[PERF] style_extractor.extract ({len(texts)} texts, dataset): "
            f"{time.time() - t0:.1f}s"
        )

        t0 = time.time()
        framer = build_argument_framer()
        logger.info(f"[PERF] framer created (dataset): {time.time() - t0:.1f}s")

        t0 = time.time()
        await framer.prepare()
        logger.info(f"[PERF] framer.prepare (dataset): {time.time() - t0:.1f}s")

        t0 = time.time()
        framed_arguments = list(framer.frame(arguments_data))
        logger.info(
            f"[PERF] framer.frame ({len(arguments_data)} args, dataset): "
            f"{time.time() - t0:.1f}s"
        )

        total = time.time() - t0
        logger.info(
            f"[PERF] _get_styled_arguments_from_fever_dataset total: {total:.1f}s"
        )

        return (
            [
                StyledFramedArgument(
                    claim=framed.claim,
                    evidence=framed.evidence,
                    primary_frame=framed.primary_frame,
                    frame_probabilities=framed.frame_probabilities,
                    style_vector=vector,
                )
                for framed, vector in zip(framed_arguments, style_vectors, strict=True)
            ],
            texts,
            style_vectors,
        )

    async def _load_from_disk(self) -> bool:
        """
        Load a previously prepared dataset from disk instead of regenerating it.

        Returns:
            bool: True if the dataset was loaded from disk.
        """
        candidate_path = (
            config.data_directory
            / config.preprocessed_dataset_subdirectory
            / self.processed_folder_name
        )
        training_csv = candidate_path / f"{SubsetName.TRAINING.value}.csv"
        if not training_csv.exists():
            return False
        self._splits_path = candidate_path
        logger.debug(f"Loaded a dataset from disk at: {candidate_path}")
        return True

    @override
    async def prepare(self) -> Path:
        if self._splits_path is not None:
            logger.debug(
                "The dataset is already prepared. Its cached version will be used."
            )
            return self._splits_path

        if await self._load_from_disk():
            return cast("Path", self._splits_path)

        self._splits_path = (
            config.data_directory
            / config.preprocessed_dataset_subdirectory
            / self.processed_folder_name
        )
        self._splits_path.mkdir(parents=True, exist_ok=True)

        (
            styled_arguments,
            texts,
            style_vectors,
        ) = await self._get_styled_arguments_from_fever_dataset()
        self._save_style_vectors(
            texts=texts, style_vectors=style_vectors, path=self._splits_path
        )
        train_args, val_args, test_args = self._split_arguments(styled_arguments)

        t0 = time.time()
        with TransformersNLI() as nli:
            logger.info(
                f"[PERF] TransformersNLI context entered: {time.time() - t0:.1f}s"
            )

            t0 = time.time()
            train_df = await self._generate_argument_encoding_dataset(train_args, nli)
            logger.info(
                f"[PERF] _generate_argument_encoding_dataset (train, "
                f"{len(train_args)} args, {len(train_df)} rows): "
                f"{time.time() - t0:.1f}s"
            )

            t0 = time.time()
            val_df = await self._generate_argument_encoding_dataset(val_args, nli)
            logger.info(
                f"[PERF] _generate_argument_encoding_dataset (val, "
                f"{len(val_args)} args, {len(val_df)} rows): "
                f"{time.time() - t0:.1f}s"
            )

            t0 = time.time()
            test_df = await self._generate_argument_encoding_dataset(test_args, nli)
            logger.info(
                f"[PERF] _generate_argument_encoding_dataset (test, "
                f"{len(test_args)} args, {len(test_df)} rows): "
                f"{time.time() - t0:.1f}s"
            )

        self._save_frame_probabilities(styled_arguments, self._splits_path)

        train_df.drop(columns=["style_vector_1", "style_vector_2"]).to_csv(
            self._splits_path / f"{SubsetName.TRAINING.value}.csv",
            index=False,
        )
        val_df.drop(columns=["style_vector_1", "style_vector_2"]).to_csv(
            self._splits_path / f"{SubsetName.VALIDATION.value}.csv",
            index=False,
        )
        test_df.drop(columns=["style_vector_1", "style_vector_2"]).to_csv(
            self._splits_path / f"{SubsetName.TESTING.value}.csv",
            index=False,
        )

        return self._splits_path

    def _split_arguments(
        self,
        arguments: list[StyledFramedArgument],
    ) -> tuple[
        list[StyledFramedArgument],
        list[StyledFramedArgument],
        list[StyledFramedArgument],
    ]:
        """Split arguments into train/validation/test subsets to avoid row leakage."""
        splits = 3
        if len(arguments) < splits:
            return arguments, [], []

        train_val_args, test_args = train_test_split(
            arguments,
            test_size=0.1,
            random_state=self.random_seed,
        )
        train_args, val_args = train_test_split(
            train_val_args,
            test_size=0.1 / 0.9,
            random_state=self.random_seed,
        )
        return train_args, val_args, test_args

    @override
    def get_split(self, split: SubsetName, max_samples: int = 0) -> pd.DataFrame:
        """
        Get data for a given split.

        The dataset has to be prepared before getting a given split.

        Args:
            split (SubsetName): Split for which data should be provided.
            max_samples (int, optional): Maximum number of samples to return.
                If set to 0, all samples are returned. Defaults to 0.

        Raises:
            NotPreparedError: Raised if the dataset was not prepared (`prepare()`)
                before getting the split.

        Returns:
            pd.DataFrame: Data frame with examples.
        """
        if self._splits_path is None:
            raise NotPreparedError(
                "Cannot get a dataset split from an unprepared dataset. "
                f"Call `{ArgumentEncodingDataset.__name__}.prepare()` first."
            )

        df = super().get_split(split=split, max_samples=max_samples)

        # Restore style vectors indexed by argument text.
        style_vector_store_path = self._splits_path / "style_vectors.npz"
        if style_vector_store_path.exists():
            with np.load(style_vector_store_path) as data:
                # pylint: disable=no-member
                # It's a NumPy array and has `tolist()`.
                stored_texts = data["texts"].tolist()
                # pylint: enable=no-member
                stored_vectors = data["vectors"]
            style_store = dict(zip(stored_texts, stored_vectors, strict=True))
            df["style_vector_1"] = pd.Series(
                [
                    style_store[f"{row['claim_1']} {row['evidence_1']}"]
                    for _, row in df.iterrows()
                ],
                index=df.index,
            )
            df["style_vector_2"] = pd.Series(
                [
                    style_store[f"{row['claim_2']} {row['evidence_2']}"]
                    for _, row in df.iterrows()
                ],
                index=df.index,
            )

        # Restore frame probabilities indexed by argument text.
        frame_prob_store_path = self._splits_path / "frame_probabilities.npz"
        if frame_prob_store_path.exists():
            frame_order = list(InterpretativeFrame)
            with np.load(frame_prob_store_path) as data:
                # pylint: disable=no-member
                # It's a NumPy array and has `tolist()`.
                stored_frame_texts = data["frame_prob_texts"].tolist()
                # pylint: enable=no-member
                stored_frame_matrix = data["frame_prob_matrix"]
            frame_store = {
                text: {
                    frame: float(prob)
                    for frame, prob in zip(frame_order, row, strict=True)
                }
                for text, row in zip(
                    stored_frame_texts, stored_frame_matrix, strict=True
                )
            }
            df["frame_probabilities_1"] = pd.Series(
                [
                    frame_store[f"{row['claim_1']} {row['evidence_1']}"]
                    for _, row in df.iterrows()
                ],
                index=df.index,
            )
            df["frame_probabilities_2"] = pd.Series(
                [
                    frame_store[f"{row['claim_2']} {row['evidence_2']}"]
                    for _, row in df.iterrows()
                ],
                index=df.index,
            )
        return df

    @override
    def clear(self) -> None:
        """Remove both raw and the processed data."""
        if self._splits_path is not None and self._splits_path.exists():
            shutil.rmtree(self._splits_path)
            self._splits_path = None


__all__ = ["ArgumentEncodingDataset"]
