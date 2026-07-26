"""Package for search module model configurations."""

from pydantic import BaseModel, Field


class SearchConfig(BaseModel):
    """Search engine configuration."""

    max_results: int = Field(default=5, ge=1)
    region: str = "wt-wt"  # aleternatively: "us-en"
    safesearch: str = "moderate"


class DownloadConfig(BaseModel):
    """Downloader configuration."""

    timeout: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    user_agent: str = (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/138.0 Safari/537.36"
    )
    max_content_size_mb: int = Field(
        default=10,
        ge=1,
    )


class CleaningConfig(BaseModel):
    """HTML cleaning configuration."""

    min_text_length: int = Field(default=300, ge=0)  # to be adjusted
    max_text_length: int = Field(default=10000, ge=0)  # to be adjusted
    min_word_count: int = Field(default=50, ge=0)  # to be adjusted
    min_sentence_count: int = Field(default=3, ge=0)  # to be adjusted
    remove_empty_lines: bool = True
    collapse_whitespace: bool = True
    remove_tags: tuple[str, ...] = (
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "noscript",
        "svg",
        "form",
        "iframe",
        "button",
    )
    unwanted_keywords: tuple[str, ...] = (
        "cookie",
        "privacy",
        "newsletter",
        "subscribe",
        "sign up",
        "advertisement",
        "share",
        "comments",
        "related",
        "login",
    )
    content_tags: tuple[str, ...] = (
        "article",
        "main",
        "section",
        "div",
    )
    keep_links: bool = False


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    ttl_seconds: int = Field(default=7 * 24 * 60 * 60, ge=1)
    directory: str = ".cache/search"
    version: str = "v1"


class KeywordConfig(BaseModel):
    """Keyword extraction configuration."""

    max_keywords: int = Field(default=5, ge=1)
    min_word_length: int = Field(default=3, ge=1)
    ngram_sizes: tuple[int, ...] = (1, 2)
    stop_words: set[str] = Field(
        default_factory=lambda: {
            "the",
            "and",
            "are",
            "for",
            "with",
            "that",
            "this",
            "from",
            "have",
            "has",
            "because",
            "while",
            "than",
            "more",
            "less",
            "into",
            "over",
            "under",
        }
    )


class SearchModuleConfig(BaseModel):
    """Complete configuration of the search module."""

    keyword: KeywordConfig = Field(default_factory=KeywordConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
