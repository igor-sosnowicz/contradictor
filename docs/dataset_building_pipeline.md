# Dataset Building Pipeline

This document defines the argument types and the two-stage LLM pipeline
(Etap 1: per-document extraction, Etap 2: global linking) used by
`src/argument_dataset/pipeline.py` to build the
contr-argument dataset.

### Running the pipeline

Setup: start the LM Studio server locally and set both model names in
`config.toml` (`extractor_model` / `judge_model`, currently GPT OSS 20B for
both). All requests are standalone pydantic-ai `Agent` runs (see
`src/argument_dataset/backend.py`).

```bash
uv run python -m src.argument_dataset.pipeline --phase all
```

Sources default to `data/counterargument-dataset/sources/` (override with
`--sources DIR`).

Useful flags: `--phase 1|2|all`, `--limit N`, `--top-k N`,
`--chunk-size N`, `--chunk-overlap N`, `--no-repair` (judge accepts/rejects
without repairing), `--dry-run` (no saving), `--extractor-model` /
`--judge-model` (override config), `--sources DIR`.

### Argument

            Inference
Premises ---------------> Claims  (Argument)


Evidence ----------------> Claims

### Argument Types

- CounterArgument: Attacks the underlying reasoning, premises, or inference of the target.
    - Logic: "Target's premises do not support their conclusion (If P, it doesn't mean Q)."
    - Context: "Padł deszcz, więc asfalt na drodze na pewno jest mokry."
    - Example: "Ten odcinek drogi biegnie pod zadaszonym tunelem, więc deszcz nie sprawił, że jest mokry."

- CounterExample: Provides a specific instance that invalidates a universal or general claim.
    - Logic: "Target claims 'All X are Y', but here is an X that is not Y."
    - Context: "Wszystkie ptaki potrafią latać."
    - Example: "Pingwiny są ptakami, a nie potrafią latać."

- Contradiction: Asserts a statement that directly conflicts with the target claim, without providing new evidence.
    - Logic: "Target claims A, but B is asserted, where A and B cannot both be true."
    - Context: "Pijana kofeina po godzinie 18:00 zawsze psuje jakość snu."
    - Example: "Kofeina wypita po 18:00 wcale nie psuje jakości snu."

- Rebuttal: Refutes the target's conclusion directly by presenting opposing evidence or a counter-claim.
    - Logic: "Target claims X, but evidence E proves non-X."
    - Context: "Praca z domu drastycznie obniża wydajność pracowników."
    - Example: "Ostatnie badania efektywności wskazują, że wydajność zespołu wzrosła o 15% po przejściu na model hybrydowy."


[Surowy Text]
      │
      ▼
┌───────────┐
│    LLM    │ ──> Zwraca ExtractedArgument (Structured Output)
└───────────┘
      │
      ▼
┌───────────┐
│ Exact     │ ──> Szuka `text` w dokumencie surowym (np. regex/string find)
│ Matcher   │     i wylicza `start_idx` oraz `end_idx` dla argumentu i przesłanek
└───────────┘
      │
      ▼
┌───────────┐
│ Embedding │ ──> Generuje wektor z połączenia: f"{domain} | argument: {argument}"
│ Model     │
└───────────┘
      │
      ▼
┌───────────┐
│ Zapis DB  │ ──> Tworzy `ArgumentRecord` i zapisuje w JSON + Pickle (embeddings)
└───────────┘



Pipeline:

===================================================================
ETAP 1: Ekstrakcja i Wektoryzacja (Per-Document)
===================================================================
[Plik 1] ──> Extractor LLM ──> [ExtractedArgs] ──> Judge (accept/repair/reject)
  ──> Fuzzy Match ──> Embeddings ──> ArgumentDatabase (args_db.json + pkl)
[Plik 2] ──> (jw.)
[Plik N] ──> (jw.)

===================================================================
ETAP 2: Globalne Linkowanie (Graf Relacji)
===================================================================
Dla każdego argumentu w Bazie:
  1. Vector Search ──> `fetch_similar_arguments` szuka Top-K najbardziej
     podobnych tez (ten sam document dozwolony)
  2. Connection Judge ──> "Czy Argument X obala Tezę Y?" (refutes TAK/NIE,
     typ linku, ewentualna korekta hipotezy)
  3. Zapis Relacji ──> `add_argument_link` zapisuje link w polu `argument_links`
