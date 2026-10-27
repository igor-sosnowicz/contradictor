"""Module for parsing and coercing dataset cells into schema models."""

from collections.abc import Iterable, Mapping
from typing import Literal, cast

import pandas as pd

from src.data_models.data_models import InterpretativeFrame, NLIPrediction
from src.golden_dataset.schema_models import (
    ArgumentRelation,
    Claim,
    Evidence,
    Premise,
    StanceType,
    TextBounds,
)


def optional_str(value: object) -> str | None:
    """Coerce a cell into a non-empty string, or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def as_arg_type(value: object) -> Literal["raw", "canonical"]:
    """Coerce a type cell into the canonical Literal, rejecting unknowns."""
    val_str = str(value).strip().lower()
    if val_str not in ("raw", "canonical"):
        raise ValueError(
            f"Invalid argument type: {val_str!r}. Expected 'raw' or 'canonical'."
        )
    return cast('Literal["raw", "canonical"]', val_str)


def as_frame(value: object) -> InterpretativeFrame | None:
    """Coerce a frame cell into an InterpretativeFrame, or None when absent."""
    text = optional_str(value)
    if text is None:
        return None
    return InterpretativeFrame(text)


def as_premises(value: object) -> list[Premise]:
    """Coerce a premises cell (list, string or blank) into Premise models."""
    if value is None or isinstance(value, float):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [Premise(text=text)] if text else []
    if isinstance(value, Iterable):
        return [Premise(text=str(item).strip()) for item in value if str(item).strip()]
    return []


def as_evidence(value: object) -> list[Evidence]:
    """Coerce an evidence cell (str, dict or iterable of those) into Evidence models."""
    if value is None or isinstance(value, float):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [Evidence(text=text)] if text else []
    if isinstance(value, dict):
        return [evidence_from_dict(value)]
    if isinstance(value, Iterable):
        return [
            evidence_from_dict(item)
            if isinstance(item, dict)
            else Evidence(text=str(item).strip())
            for item in value
            if str(item).strip()
        ]
    return []


def evidence_from_dict(value: dict) -> Evidence:
    """Build an Evidence from a dict, keeping only present allowed fields."""
    text = str(value.get("text", "")).strip()
    source_url: str | None = None
    bounds = as_bounds(value.get("bounds"))

    if value.get("source_url") is not None:
        source_url = str(value["source_url"])

    return Evidence(text=text, source_url=source_url, bounds=bounds)


def as_bounds(value: object) -> TextBounds | None:
    """Coerce a bounds cell (dict with int-able start/end) into TextBounds, or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, TextBounds):
        return value
    if (
        isinstance(value, Mapping)
        and value.get("start") is not None
        and value.get("end") is not None
    ):
        return TextBounds(start=int(value["start"]), end=int(value["end"]))
    return None


def as_relations(value: object) -> list[ArgumentRelation]:
    """Coerce a relations cell (dicts or models) into ArgumentRelation models."""
    if value is None or isinstance(value, float):
        return []
    if isinstance(value, ArgumentRelation):
        return [value]
    if isinstance(value, Iterable):
        relations: list[ArgumentRelation] = []
        for item in value:
            if isinstance(item, ArgumentRelation):
                relations.append(item)
                continue
            if isinstance(item, dict):
                target = item.get("target_argument_id")
                if target is None or (isinstance(target, float) and pd.isna(target)):
                    continue
                text = str(target).strip()
                if not text:
                    continue
                relations.append(
                    ArgumentRelation(
                        target_argument_id=text,
                        stance=StanceType(str(item["stance"])),
                        nli_prediction=as_nli(item.get("nli_prediction")),
                    )
                )
        return relations
    return []


def as_nli(value: object) -> NLIPrediction | None:
    """Coerce an NLI cell into an NLIPrediction, or None when absent."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, NLIPrediction):
        return value
    if isinstance(value, dict):
        kwargs = {
            key: val
            for key, val in value.items()
            if val is not None and not (isinstance(val, float) and pd.isna(val))
        }
        # Values originate from untyped dataset cells; let the model validate them.
        return NLIPrediction(**kwargs)  # type: ignore[arg-type]
    return None


# --- Detailed parsers ---
def gld_dataset_claims(spans: object) -> list[Claim]:
    """
    Coerce Igor's claim span dicts into Claim models, skipping blanks.

    Accepts a single span dict, an iterable of span dicts, or already-dumped
    Claim dicts ({text, is_main_claim, bounds}) from parquet round-trips.
    """
    if isinstance(spans, Mapping):
        spans = [spans]
    rows: list[Claim] = []
    if not isinstance(spans, Iterable) or isinstance(spans, (str, float)):
        return rows
    for span in spans:
        if isinstance(span, Claim):
            rows.append(span)
            continue
        if not isinstance(span, dict):
            text = str(span).strip()
            if text:
                rows.append(Claim(text=text))
            continue
        text = str(span.get("text", "")).strip()
        if not text:
            continue
        bounds = as_bounds(span.get("bounds"))
        if (
            bounds is None
            and span.get("start") is not None
            and span.get("end") is not None
        ):
            bounds = TextBounds(start=int(span["start"]), end=int(span["end"]))
        rows.append(
            Claim(
                text=text,
                bounds=bounds,
                is_main_claim=bool(span.get("is_main_claim", False)),
            )
        )
    return rows


def gld_dataset_evidence(spans: object) -> list[Evidence]:
    """Coerce Igor's evidence span dicts into Evidence models, skipping blanks."""
    rows: list[Evidence] = []
    if not isinstance(spans, Iterable) or isinstance(spans, (str, float)):
        return rows
    for span in spans:
        if not isinstance(span, dict):
            text = str(span).strip()
            if text:
                rows.append(Evidence(text=text))
            continue
        if span.get("start") is None or span.get("end") is None:
            continue
        text = str(span.get("text", "")).strip()
        if not text:
            continue
        rows.append(
            Evidence(
                text=text,
                bounds=TextBounds(start=int(span["start"]), end=int(span["end"])),
            )
        )
    return rows
