"""Module with SpaCy-based style extractor."""

import asyncio
import shutil
from collections import Counter
from typing import Final, Literal, cast, get_args, override

import numpy as np
import spacy
import textstat_rs
import transformers
from loguru import logger
from transformers import logging as transformers_logging
from transformers import pipeline

from src.style_extraction.style_extractor import StyleExtractor
from src.utils.errors import NotPreparedError

type SpacyModel = Literal[
    "en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"
]


class SpacyStyleExtractor(StyleExtractor):
    """SpaCy-based implementation of style extractor."""

    SUFFICIENT_TEXT_LENGTH: int = 100

    # Source: https://arxiv.org/pdf/2502.04321, see figure 3.3, page 14
    # Representative dataset with over a million sentence in the non-fiction category.
    # 95th percentile of sentence length is around 50.
    UPPER_SENTENCE_LENGTH = 50.0

    # Source: https://arxiv.org/pdf/1207.2334, figure 1, page 2
    # Calculated on words from a dictionary.
    # 95th percentile of word length is around 16.0
    UPPER_WORD_LENGTH = 16.0

    # Readability metrics output values from 6 (sixth grade) to 17 (graduate level).
    MIN_GRADE: Final[int] = 6
    MAX_GRADE: Final[int] = 17

    # Flesch Reading-Ease - range of values.
    MIN_EASE: Final[int] = 100
    MAX_EASE: Final[int] = 10

    # Dale-Chall Readability Score - range of values.
    DALE_CHALL_MIN: Final[float] = 4.9
    DALE_CHALL_MAX: Final[float] = 9.9

    def __init__(self, spacy_model: SpacyModel = "en_core_web_lg") -> None:
        """
        Validate SpaCy model name.

        Args:
            spacy_model (SpacyModel, optional): SpaCy pre-built model.
                Defaults to "en_core_web_lg".

        Raises:
            ValueError: Raised if non-existent SpaCy model was requested.
        """
        self._spacy_model = spacy_model

        # Suppress transformers library warnings.
        transformers_logging.set_verbosity_error()

        # Pylint false positive for PEP 695 TypeAliasType.__value__.
        available_spacy_models = get_args(SpacyModel.__value__)  # pylint: disable=no-member
        if self._spacy_model not in available_spacy_models:
            raise ValueError(
                f"Invalid SpaCy model: {self._spacy_model}. "
                f"Allowed models: {', '.join(available_spacy_models)}"
            )
        self._nlp: spacy.language.Language | None = None
        self._sentiment_pipeline: transformers.Pipeline | None = None

    def _grade_to_unit(self, grade: float) -> float:
        return (grade - self.MIN_GRADE) / (self.MAX_GRADE - self.MIN_GRADE)

    # Flesch Reading-Ease
    def _flesch_reading_ease_to_unit(self, ease: float) -> float:
        return (ease - self.MIN_EASE) / (self.MAX_EASE - self.MIN_EASE)

    # Dale-Chall Readability Score
    def _dale_chall_to_unit(self, score: float) -> float:
        return (score - self.DALE_CHALL_MIN) / (
            self.DALE_CHALL_MAX - self.DALE_CHALL_MIN
        )

    async def _is_spacy_model_installed(
        self,
        uv_path: str,
    ) -> bool:
        completed_process = await asyncio.create_subprocess_exec(
            uv_path,
            "run",
            "spacy",
            "validate",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await completed_process.communicate()
        return self._spacy_model in stdout.decode()

    async def _download_spacy_model(self, uv_path: str) -> None:
        logger.debug(
            f"SpaCy model {self._spacy_model} is missing. The model will be downloaded."
        )
        download_command = [uv_path, "run", "spacy", "download", self._spacy_model]
        logger.debug(f"SpaCy model download starts: {' '.join(download_command)}")
        process = await asyncio.create_subprocess_exec(*download_command)
        _, stderr = await process.communicate()

        if process.returncode != 0:
            stderr_text = stderr.decode().strip() if stderr else ""
            raise RuntimeError(
                f"Failed to download SpaCy model {self._spacy_model}."
                + (f" Details: {stderr_text}" if stderr_text else "")
            )

        logger.info(f"Successfully downloaded the {self._spacy_model} SpaCy model.")

    def _prepare_spacy_nlp_pipeline(self) -> spacy.language.Language:
        nlp = spacy.load(
            self._spacy_model,
            disable=[
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
        )
        nlp.add_pipe("sentencizer")
        return nlp

    @override
    async def prepare(self) -> None:
        # Resolve to absolute path
        uv_path = shutil.which("uv")
        if uv_path is None:
            raise RuntimeError("`uv` executable not found in PATH.")

        if not await self._is_spacy_model_installed(uv_path):
            await self._download_spacy_model(uv_path)

        self._nlp = self._prepare_spacy_nlp_pipeline()
        self._sentiment_pipeline = self._get_sentiment_pipeline()

    def _get_sentiment_pipeline(self) -> transformers.Pipeline:
        return pipeline(
            task="text-classification",  # equivalent to "sentiment-analysis"
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )

    async def _calculate_positivity(
        self, texts: list[str], sentiment_pipeline: transformers.Pipeline
    ) -> list[float]:
        sentiment_output = sentiment_pipeline(texts)
        if sentiment_output is None:
            raise RuntimeError("Sentiment pipeline returned no output.")

        if not isinstance(sentiment_output, list):
            sentiment_output = list(sentiment_output)

        sentiments: list[dict[str, str | float]] = cast(
            "list[dict[str, str | float]]", sentiment_output
        )
        # An example of the format: [{"label": "positive", "score": 0.99}]
        sentiment_to_positivity_mapping = {
            "positive": 1.0,
            "neutral": 0.5,
            "negative": 0.0,
        }
        return [
            # Adjust strength of positivty score by the classifier's confidence.
            sentiment_to_positivity_mapping[cast("str", sentiment["label"])]
            * float(sentiment["score"])
            for sentiment in sentiments
        ]

    @override
    async def extract(self, texts: list[str]) -> list[np.ndarray]:
        if self._nlp is None or self._sentiment_pipeline is None:
            raise NotPreparedError(
                "SpacyStyleExtractor not prepared. "
                f"Call {self.prepare.__name__}() first."
            )

        sentiment_task = asyncio.create_task(
            self._calculate_positivity(
                texts=texts, sentiment_pipeline=self._sentiment_pipeline
            )
        )
        vectors = []
        for text, document in zip(texts, self._nlp.pipe(texts), strict=True):
            if not text:
                raise ValueError("Cannot extract style from an empty text.")

            sentences = len(list(document.sents))
            if sentences == 0:
                raise ValueError("Cannot extract style from a text without sentences.")

            characters = len(text)
            token_count = len(document)
            parts_of_speech = [token.pos_ for token in document]
            pos_counts = Counter(parts_of_speech)

            # Word count is equal to a number of tokens minus punctuation marks.
            words = token_count - pos_counts["PUNCT"]
            if words == 0:
                raise ValueError("Cannot extract style from a text without words.")
            if words < self.SUFFICIENT_TEXT_LENGTH:
                logger.warning(
                    f"Measuring style of the text made of {words} words "
                    "is not representative of its style. Please, use text longer than "
                    f"{self.SUFFICIENT_TEXT_LENGTH} words."
                )

            metrics = [
                # Normalised lengths.
                (words / sentences) / self.UPPER_SENTENCE_LENGTH,  # sentence length
                (characters / words) / self.UPPER_WORD_LENGTH,  # word length
                # Parts of speech.
                pos_counts["NOUN"] / token_count,
                pos_counts["VERB"] / token_count,
                pos_counts["ADJ"] / token_count,
                pos_counts["PUNCT"] / token_count,
                pos_counts["NUM"] / token_count,  # numeric tokens
                pos_counts["PROPN"] / token_count,  # proper nouns
                # Classical readability metrics.
                self._grade_to_unit(textstat_rs.gunning_fog(text)),
                self._flesch_reading_ease_to_unit(
                    textstat_rs.flesch_reading_ease(text)
                ),
                self._grade_to_unit(textstat_rs.flesch_kincaid_grade(text)),
                self._grade_to_unit(textstat_rs.smog_index(text)),
                self._grade_to_unit(textstat_rs.coleman_liau_index(text)),
                self._grade_to_unit(textstat_rs.automated_readability_index(text)),
                self._dale_chall_to_unit(
                    textstat_rs.dale_chall_readability_score(text)
                ),
                self._grade_to_unit(textstat_rs.linsear_write_formula(text)),
                textstat_rs.difficult_words(text) / words,  # share of difficult words
            ]

            vectors.append(
                np.clip(  # Some metrics may be under 0.0 or exceed 1.0. Winsorise.
                    np.array(metrics),
                    0.0,
                    1.0,
                )
            )

        positivity_scores = await sentiment_task
        # Add positivity score calculated in parallel to the end of each vector.
        return [
            np.append(vector, positivity_score)
            for vector, positivity_score in zip(vectors, positivity_scores, strict=True)
        ]
