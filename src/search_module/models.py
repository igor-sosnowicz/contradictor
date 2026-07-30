"""Package with search module data models."""

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_extra_types.language_code import LanguageAlpha2


class SearchQuery(BaseModel):
    """Represents a processed search query with extracted keywords."""

    original_text: str
    keywords: tuple[str, ...]
    normalized: str


class SearchResult(BaseModel):
    """Represents a single search engine result with metadata."""

    url: str
    title: str
    snippet: str = ""


class Document(BaseModel):
    """Represents a cleaned document retrieved from a web page."""

    url: str
    text: str
    title: str = ""
    source: str = ""
    query: str = ""
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status_code: int = 200
    content_type: str = ""
    language: Annotated[str, LanguageAlpha2] = "en"
