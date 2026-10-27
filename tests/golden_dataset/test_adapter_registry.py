"""Unit tests for AdapterRegistry, register_adapter and read_canonical."""

from collections.abc import Callable

import pandas as pd
import pytest

from src.golden_dataset.datasets_handlers.adapter_registry import (
    AdapterRegistry,
    register_adapter,
    registry,
)
from src.golden_dataset.schema_models import (
    ArgumentRelation,
    Claim,
    Evidence,
    SchemaArgument,
    StanceType,
    TextBounds,
)

NAN = float("nan")

COLUMNS = (
    "id",
    "type",
    "argument",
    "claim",
    "premises",
    "evidence",
    "frame",
    "relations",
    "provenance",
    "claim_bounds",
    "document_id",
)

RAW_ROW: dict = {"id": "S:1", "type": "raw", "argument": "hitz text"}

IGOR_ROW: dict = {
    "id": "igor:1",
    "type": "canonical",
    "argument": "full argument",
    "claim": [
        {"text": "main", "is_main_claim": True, "bounds": {"start": 0, "end": 4}}
    ],
    "evidence": [{"text": "ev", "bounds": {"start": 5, "end": 9}}],
    "relations": [{"target_argument_id": "igor:2", "stance": "rebuts"}],
    "provenance": "igor",
    "claim_bounds": {"start": 0, "end": 4},
    "document_id": "doc-7",
}


def _frame(*rows: dict) -> pd.DataFrame:
    """Build a dataframe over the full canonical column set, NaN-filled."""
    return pd.DataFrame([{**dict.fromkeys(COLUMNS, NAN), **row} for row in rows])


def _stub_adapter(frame: pd.DataFrame) -> list[SchemaArgument]:
    """Do-nothing adapter used as a registry sentinel."""
    return []


class TestInstanceIndependence:
    """AdapterRegistry is a plain class: instances do not share state."""

    def test_instances_have_independent_adapters(self) -> None:
        """Registering on one instance leaves the others untouched."""
        one, two = AdapterRegistry(), AdapterRegistry()
        one.register("hitz", _stub_adapter)
        assert one.registered_sources() == ["hitz"]
        assert two.registered_sources() == []
        assert "hitz" not in registry.registered_sources()


class TestRegisterAndGetAdapter:
    """register + get_adapter resolve names, else fall back to canonical."""

    def test_registered_adapter_returned(self) -> None:
        """The exact registered function is handed back."""
        reg = AdapterRegistry()
        reg.register("hitz", _stub_adapter)
        assert reg.get_adapter("hitz") is _stub_adapter

    def test_unknown_source_falls_back_to_read_canonical(self) -> None:
        """An unregistered source resolves to the canonical reader."""
        reg = AdapterRegistry()
        assert reg.get_adapter("no-such-source") == reg.read_canonical


class TestRegisteredSources:
    """registered_sources lists names sorted."""

    def test_empty_registry_is_empty(self) -> None:
        """A fresh registry has no sources."""
        assert AdapterRegistry().registered_sources() == []

    def test_sources_sorted(self) -> None:
        """Names come back alphabetically regardless of insert order."""
        reg = AdapterRegistry()
        for name in ("zeta", "alpha", "mid"):
            reg.register(name, _stub_adapter)
        assert reg.registered_sources() == ["alpha", "mid", "zeta"]


class TestRegisterAdapterDecorator:
    """register_adapter binds a decorator to one registry instance."""

    def test_registers_and_returns_adapter_unchanged(self) -> None:
        """The decorated function is returned as-is and registered."""
        reg = AdapterRegistry()

        @register_adapter(reg, "hitz")
        def hitz_adapter(frame: pd.DataFrame) -> list[SchemaArgument]:
            """Do-nothing adapter under decoration."""
            return []

        assert reg.registered_sources() == ["hitz"]
        assert reg.get_adapter("hitz") is hitz_adapter


class TestNormalize:
    """normalize dispatches to the source adapter or the canonical reader."""

    def test_registered_adapter_used(self) -> None:
        """A registered adapter receives the frame and its rows are returned."""
        reg = AdapterRegistry()
        seen: list[pd.DataFrame] = []
        sentinel = [SchemaArgument(id="S:1", type="raw", argument="text")]

        def adapter(frame: pd.DataFrame) -> list[SchemaArgument]:
            """Record the frame, return a fixed row."""
            seen.append(frame)
            return sentinel

        reg.register("hitz", adapter)
        frame = pd.DataFrame({"text": ["x"]})
        assert reg.normalize("hitz", frame) == sentinel
        assert seen == [frame]

    def test_unregistered_source_falls_back_to_canonical(self) -> None:
        """An unregistered source is parsed by read_canonical."""
        rows = AdapterRegistry().normalize("igor", _frame(IGOR_ROW))
        assert [row.id for row in rows] == ["igor:1"]


class TestReadCanonical:
    """read_canonical coerces canonical rows via the cell parsers."""

    @pytest.mark.parametrize(
        ("row", "position", "read", "expected"),
        [
            pytest.param(
                RAW_ROW,
                0,
                lambda row_: row_.claim,
                [],
                id="raw-empty-claim-list",
            ),
            pytest.param(
                RAW_ROW,
                0,
                lambda row_: (row_.claim_bounds, row_.document_id, row_.frame),
                (None, None, None),
                id="raw-nan-nullable-cells-to-none",
            ),
            pytest.param(
                IGOR_ROW,
                0,
                lambda row_: row_.claim,
                [
                    Claim(
                        text="main",
                        is_main_claim=True,
                        bounds=TextBounds(start=0, end=4),
                    )
                ],
                id="dumped-claim-dicts-parsed-with-bounds",
            ),
            pytest.param(
                IGOR_ROW,
                0,
                lambda row_: row_.evidence,
                [Evidence(text="ev", bounds=TextBounds(start=5, end=9))],
                id="evidence-dicts-parsed-with-bounds",
            ),
            pytest.param(
                IGOR_ROW,
                0,
                lambda row_: row_.relations,
                [
                    ArgumentRelation(
                        target_argument_id="igor:2", stance=StanceType.REBUTS
                    )
                ],
                id="relation-dicts-parsed",
            ),
            pytest.param(
                IGOR_ROW,
                0,
                lambda row_: (row_.provenance, row_.document_id, row_.frame),
                ("igor", "doc-7", None),
                id="nullable-strings-coerced",
            ),
        ],
    )
    def test_pieces(
        self,
        row: dict,
        position: int,
        read: Callable[[SchemaArgument], object],
        expected: object,
    ) -> None:
        """Each canonical cell ends up coerced on the built SchemaArgument."""
        rows = registry.read_canonical(_frame(row))
        assert read(rows[position]) == expected

    @pytest.mark.parametrize("missing", ["id", "type"])
    def test_missing_required_column_raises(self, missing: str) -> None:
        """Dropping a required column fails with the not-canonical error."""
        frame = _frame(RAW_ROW).drop(columns=[missing])
        with pytest.raises(KeyError, match="not canonical"):
            registry.read_canonical(frame)


class TestToDataFrameRoundTrip:
    """to_dataframe serialises rows back into a canonical dataframe."""

    def test_round_trip_preserves_rows(self) -> None:
        """Rows -> dataframe -> read_canonical keeps ids, types, relations."""
        rows = registry.read_canonical(_frame(RAW_ROW, IGOR_ROW))
        reparsed = registry.read_canonical(registry.to_dataframe(rows))
        assert [row.id for row in reparsed] == ["S:1", "igor:1"]
        assert [row.type for row in reparsed] == ["raw", "canonical"]
        assert [len(row.relations) for row in reparsed] == [0, 1]
        assert reparsed[1].claim == rows[1].claim
