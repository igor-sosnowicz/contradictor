"""
Canonical schema for argument dataframes and the per-source adapter registry.

Each source is fetched in whatever shape its upstream provider uses, then
immediately normalised by its adapter into one flat table:

| id | argument | counter_argument_ids | premises | frame | language |

One row per argument. ``counter_argument_ids`` holds the source-local ids of
the arguments that counter this one, EMPTY when the source provides no such
mapping. Normalisation happens once, at fetch time (see ``fetch_datasets.py``)
- the parquet under ``arguments/compliant_datasets/<source>.parquet`` is
already schema-compliant, not the original per-source shape.

Adding a dataset means writing one adapter (see ``adapters.py``); nothing
constrains the shape the source's own fetch function downloads.

To add a new source:
- define a fetch function (`src/argument_dataset/datasets_handlers/fetch_datasets.py`)
- define an adapter function (`src/argument_dataset/datasets_handlers/adapters.py`)
"""

from collections.abc import Callable, Iterable

import pandas as pd
from pydantic import BaseModel, Field

from src.data_models.data_models import InterpretativeFrame

# --- Canonical schema ---

CANONICAL_COLUMNS: tuple[str, ...] = (
    "id",
    "argument",
    "counter_argument_ids",
    "premises",
    "frame",
    "language",
)


class RawArgumentRow(BaseModel):
    id: str = Field(..., description="Identifier unique within the source dataset")
    argument: str = Field(..., description="Conclusion / claim text of the argument")
    counter_argument_ids: list[str] = Field(
        default_factory=list,
        description="Source-local ids of countering arguments, EMPTY when none",
    )
    premises: list[str] = Field(
        default_factory=list,
        description="Native premise texts, EMPTY when the source has none",
    )
    frame: InterpretativeFrame | None = Field(
        default=None,
        description="Native interpretative frame, NONE when it must be classified",
    )
    language: str = Field(default="english", description="Language of the argument")

    @property
    def has_native_premises(self) -> bool:
        """Whether the source supplied real premises for this argument."""
        return bool(self.premises)

    @property
    def has_native_counters(self) -> bool:
        """Whether the source mapped this argument to its counter-arguments."""
        return bool(self.counter_argument_ids)


# --- Adapter registry ---

RawAdapter = Callable[[pd.DataFrame], Iterable[RawArgumentRow]]
_ADAPTERS: dict[str, RawAdapter] = {}


def registered_sources() -> list[str]:
    """List registered sources with their adapters."""
    return sorted(_ADAPTERS)

def get_source_adapter(source_name: str) -> RawAdapter:
    """Get source's adapter, falling back to the canonical reader."""
    return _ADAPTERS.get(source_name, read_canonical)

def register_adapter(source_name: str) -> Callable[[RawAdapter], RawAdapter]:
    """Register an adapter for a source."""

    def decorator(adapter: RawAdapter) -> RawAdapter:
        _ADAPTERS[source_name] = adapter
        return adapter

    return decorator

def read_canonical(frame: pd.DataFrame) -> list[RawArgumentRow]:
    """Read a dataframe that follows the canonical schema."""
    missing = {"id", "argument"} - set(frame.columns)

    if missing:
        raise KeyError(
            f"Dataframe is not canonical, missing column(s): {sorted(missing)}. "
            f"Register an adapter for this source instead."
        )

    return [
        RawArgumentRow(
            id=str(record["id"]),
            argument=str(record["argument"]),
            counter_argument_ids=_as_str_list(record.get("counter_argument_ids")),
            premises=_as_str_list(record.get("premises")),
            frame=_as_frame(record.get("frame")),
            language=str(record.get("language") or "english"),
        )
        for record in frame.to_dict(orient="records")
    ]

def normalize(source_name: str, frame: pd.DataFrame) -> list[RawArgumentRow]:
    """Convert one raw source dataframe into canonical argument rows."""
    return list(get_source_adapter(source_name)(frame))

def to_dataframe(rows: list[RawArgumentRow]) -> pd.DataFrame:
    """Serialise canonical rows back into a schema-compliant dataframe."""
    return pd.DataFrame(
        [row.model_dump() for row in rows], columns=list(CANONICAL_COLUMNS)
    )


# --- Reader Helpers ---

def _optional_str(value: object) -> str | None:
    """Coerce a cell into a non-empty string, or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None

def _as_str_list(value: object) -> list[str]:
    """Coerce a cell (list, string or blank) into a list of non-empty strings."""
    if value is None or isinstance(value, float):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return []

def _as_frame(value: object) -> InterpretativeFrame | None:
    """Coerce a frame cell into an InterpretativeFrame, or None when absent."""
    text = _optional_str(value)
    if text is None:
        return None
    return InterpretativeFrame(text)
