"""Per-source adapters (parsers) normalising each dataset's own dataframe shape."""

from enum import StrEnum

import pandas as pd

from src.golden_dataset.datasets_handlers.adapter_registry import (
    register_adapter,
    registry,
)
from src.golden_dataset.datasets_handlers.parsers import (
    gld_dataset_claims,
    gld_dataset_evidence,
)
from src.golden_dataset.schema_models import (
    ArgumentRelation,
    SchemaArgument,
    StanceType,
)


class DataSource(StrEnum):
    """Known sources of argument datasets."""

    HITZ_COUNTER_ARGUMENT = "HITZ_COUNTER_ARGUMENT"
    IGOR_GLD_DATASET = (
        "igor_gld_dataset"  # Hand-annotated corpus, fetched by copying the in-repo JSON
    )


@register_adapter(registry, DataSource.HITZ_COUNTER_ARGUMENT)
def hitz_counter_argument(frame: pd.DataFrame) -> list[SchemaArgument]:
    """
    HITZ Dataset: https://huggingface.co/datasets/HiTZ/counter-argument.

    From:
    | argument | counter-argument |

    To (two SchemaArgument rows, type="raw", linked by REBUTS relations):
    | HITZ_COUNTER_ARGUMENT:0:arg     | type=raw | relations=[...0:counter REBUTS] | ...
    | HITZ_COUNTER_ARGUMENT:0:counter | type=raw | relations=[...0:arg REBUTS]     | ...
    """
    rows: list[SchemaArgument] = []
    for index, record in enumerate(frame.to_dict(orient="records")):
        argument = str(record.get("argument") or "").strip()
        counter = str(record.get("counter-argument") or "").strip()
        if not argument or not counter:
            continue

        argument_id = f"{DataSource.HITZ_COUNTER_ARGUMENT}:{index}:arg"
        counter_id = f"{DataSource.HITZ_COUNTER_ARGUMENT}:{index}:counter"
        rows.append(
            SchemaArgument(
                id=argument_id,
                type="raw",
                argument=argument,
                relations=[
                    ArgumentRelation(
                        target_argument_id=counter_id, stance=StanceType.REBUTS
                    )
                ],
                provenance=DataSource.HITZ_COUNTER_ARGUMENT,
            )
        )
        rows.append(
            SchemaArgument(
                id=counter_id,
                type="raw",
                argument=counter,
                relations=[
                    ArgumentRelation(
                        target_argument_id=argument_id, stance=StanceType.REBUTS
                    )
                ],
                provenance=DataSource.HITZ_COUNTER_ARGUMENT,
            )
        )
    return rows


@register_adapter(registry, DataSource.IGOR_GLD_DATASET)
def igor_gld_dataset(frame: pd.DataFrame) -> list[SchemaArgument]:
    """
    Hand-annotated corpus (golden_dataset/argument-annotations.json)
    copied into the data root by fetch_igor_annotations.

    From (one row per annotated argument, relations pre-joined by the fetcher):
    | id | claim span | evidence spans | counter_pairs |

    To (type="canonical" — every row has a claim and evidence):
    | igor_gld_dataset:<id> | claim_bounds | evidence[...] |
    | ... relations (REBUTS, symmetric) |
    """
    rows: list[SchemaArgument] = []

    for record in frame.to_dict(orient="records"):
        original_id = str(record["id"])
        argument_id = f"{DataSource.IGOR_GLD_DATASET}:{original_id}"
        claim_span = record["claim"]  # dict: {kind, start, end, text}
        evidence_spans = record.get(
            "evidence", []
        )  # list of dicts: {kind, start, end, text}
        relations = [
            ArgumentRelation(
                target_argument_id=f"{DataSource.IGOR_GLD_DATASET}:{other_id}",
                stance=StanceType.REBUTS,
            )
            for other_id in record["counter_pairs"]
        ]
        document_id_value = record.get("documentId")
        default_document_id = f"{DataSource.IGOR_GLD_DATASET}:doc:{document_id_value}"

        rows.append(
            SchemaArgument(
                id=argument_id,
                type="canonical",
                argument=str(claim_span["text"]).strip(),
                claim=gld_dataset_claims(claim_span),
                evidence=gld_dataset_evidence(evidence_spans),
                document_id=default_document_id
                if document_id_value is not None
                else None,
                relations=relations,
                provenance=DataSource.IGOR_GLD_DATASET,
            )
        )
    return rows
