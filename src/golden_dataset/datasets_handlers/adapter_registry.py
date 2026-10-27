"""Canonical schema for argument dataframes and the per-source adapter registry."""

from collections.abc import Callable, Iterable

import pandas as pd

from src.golden_dataset.datasets_handlers.parsers import (
    as_arg_type,
    as_bounds,
    as_evidence,
    as_frame,
    as_premises,
    as_relations,
    gld_dataset_claims,
    optional_str,
)
from src.golden_dataset.schema_models import SchemaArgument

DatasetAdapter = Callable[[pd.DataFrame], Iterable[SchemaArgument]]


# --- Adapter Registry ---


class AdapterRegistry:
    """Mapping of source names to their dataset adapters."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._adapters: dict[str, DatasetAdapter] = {}

    def register(self, source_name: str, adapter: DatasetAdapter) -> None:
        """Attach an adapter to a source name."""
        self._adapters[source_name] = adapter

    def get_adapter(self, source_name: str) -> DatasetAdapter:
        """Get source's adapter, falling back to the canonical reader."""
        return self._adapters.get(source_name, self.read_canonical)

    def registered_sources(self) -> list[str]:
        """List registered sources with their adapters."""
        return sorted(self._adapters)

    def normalize(self, source_name: str, frame: pd.DataFrame) -> list[SchemaArgument]:
        """Convert one raw source dataframe into canonical argument rows."""
        return list(self.get_adapter(source_name)(frame))

    def to_dataframe(self, rows: list[SchemaArgument]) -> pd.DataFrame:
        """Serialise canonical rows back into a schema-compliant dataframe."""
        return pd.DataFrame([row.model_dump() for row in rows])

    def read_canonical(self, frame: pd.DataFrame) -> list[SchemaArgument]:
        """Read a dataframe that follows the canonical schema."""
        schema_arguments: list[SchemaArgument] = []

        try:
            schema_arguments.extend(
                _canonical_row(record) for record in frame.to_dict(orient="records")
            )
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(
                f"Dataframe is not canonical, missing column(s): {sorted(missing)}. "
                f"Register an adapter for this source instead."
            ) from e

        return schema_arguments


def _canonical_row(record: dict) -> SchemaArgument:
    """Build one SchemaArgument from a canonical dataframe record."""
    return SchemaArgument(
        id=str(record["id"]),
        type=as_arg_type(record["type"]),
        argument=str(record["argument"]),
        claim=gld_dataset_claims(record.get("claim")),
        premises=as_premises(record.get("premises")),
        evidence=as_evidence(record.get("evidence")),
        frame=as_frame(record.get("frame")),
        relations=as_relations(record.get("relations")),
        provenance=optional_str(record.get("provenance")),
        claim_bounds=as_bounds(record.get("claim_bounds")),
        document_id=optional_str(record.get("document_id")),
    )


def register_adapter(
    adapter_registry: AdapterRegistry, source_name: str
) -> Callable[[DatasetAdapter], DatasetAdapter]:
    """
    Register an adapter for a source.

    Args:
        adapter_registry: The adapter registry instance to register the adapter with.
        source_name: The name of the source the adapter is registered for.

    Returns:
        Callable[[DatasetAdapter], DatasetAdapter]: A decorator function that registers
        the adapter with the specified source name.
    """

    def decorator(adapter: DatasetAdapter) -> DatasetAdapter:
        adapter_registry.register(source_name, adapter)
        return adapter

    return decorator


registry = AdapterRegistry()
