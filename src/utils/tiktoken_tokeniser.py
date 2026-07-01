"""Module with a text tokeniser."""

from typing import override

import tiktoken

from src.data_models.abstract.tokeniser import Tokeniser


class TiktokenTokeniser(Tokeniser):
    """Text tokeniser."""

    def __init__(self) -> None:
        """Initialise a pre-trained tokeniser."""
        self.tokeniser = tiktoken.get_encoding("o200k_base")

    @override
    def encode(self, text: list[str]) -> list[list[int]]:
        return self.tokeniser.encode_batch(text, allowed_special={"<|endoftext|>"})

    @override
    def decode(self, tokens: list[list[int]]) -> list[str]:
        return self.tokeniser.decode_batch(tokens)
