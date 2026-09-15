"""
Per-source adapters (parsers) normalising each dataset's own dataframe shape.

Original-form dataframe (as downloaded) -> List[:class:`RawArgumentRow`]
(canonical schema)

One adapter per dataset. Importing this module registers every adapter.
``fetch_datasets.py`` calls ``raw_schema.normalize(source_name, frame)`` right
after downloading, so the parquet it writes is already schema-compliant.
"""

import pandas as pd

from src.argument_dataset.datasets_handlers.raw_schema import (
    RawArgumentRow,
    register_adapter,
)

# --- Adapters ---
HITZ_COUNTER_ARGUMENT = "hitz_counter_argument"


@register_adapter(HITZ_COUNTER_ARGUMENT)
def hitz_counter_argument(frame: pd.DataFrame) -> list[RawArgumentRow]:
    """
    HITZ Dataset (https://huggingface.co/datasets/HiTZ/counter-argument)

    From:
    | argument | counter-argument |

    To:
    | 0:arg     | argument | counter_argument_ids [0:counter] | ...
    | 0:counter | argument | counter_argument_ids [0:arg]     | ...
    """
    rows: list[RawArgumentRow] = []
    for index, record in enumerate(frame.to_dict(orient="records")):
        argument = str(record.get("argument") or "").strip()
        counter = str(record.get("counter-argument") or "").strip()
        if not argument or not counter:
            continue

        argument_id = f"{index}:arg"
        counter_id = f"{index}:counter"
        rows.append(
            RawArgumentRow(
                id=argument_id,
                argument=argument,
                counter_argument_ids=[counter_id],
            )
        )
        rows.append(
            RawArgumentRow(
                id=counter_id,
                argument=counter,
                counter_argument_ids=[argument_id],
            )
        )
    return rows
