# Unified path system

Every filesystem path in Contradictor is declared in one place - `src/paths/`.
No other module builds paths by hand.

This document is the reference for that package. For the short version, see [Test path isolation](../README.md#test-path-isolation).

## Design

Roots come from `Configuration.data_directory` & `.cache_directory` (PlatformDirs-backed).
The registry owns everything below the roots. `resolve()` lazily self-initialises from config. `initialise()` exists only to override roots and pre-create directories.

```
Configuration (PlatformDirs)
  └─► registry roots: data, cache
        ├─ CorePaths                    data
        │   ├─ MODELS_DIR               models/
        │   ├─ RAW_DATASETS_DIR         raw_datasets/
        │   └─ PROCESSED_DATASETS_DIR   processed_dataset/
        ├─ ArgumentDetectionPaths       data
        │   ├─ CLAIM_PROCESSED_DIR      processed_dataset/claim_extraction/
        │   └─ EVIDENCE_PROCESSED_DIR   processed_dataset/evidence_extraction/
        ├─ ArgumentFramingPaths         data
        │   ├─ FRAME_CLASSIFIER_FILE    models/xgboost_frame_classifier.pkl
        │   └─ TFIDF_VECTORIZER_FILE    models/tfidf_vectorizer.pkl
        └─ SearchPaths                  cache
            └─ SEARCH_CACHE_DIR         search/

`CorePaths.CACHES_DIR` is omitted above for width. It declares the cache root itself, written `"."`, so dynamic cache directories like `claim_extractor/` sit directly under the cache root.
```

## Roots

Two roots, both from [`platformdirs`](https://pypi.org/project/platformdirs/), so a user gets platform-appropriate locations:

| Root | Linux default | Source |
|---|---|---|
| data | `~/.local/share/Contradictor` | `Configuration.data_directory` |
| cache | `~/.cache/Contradictor` | `Configuration.cache_directory` |

They live in one module-global, `registry._roots`, populated from configuration on first use.

## API

| Function | Purpose |
|---|---|
| `initialise(data_root, cache_root)` | Override the roots, then create every declared directory. Supply both roots to avoid reading `config.toml`. New roots activate only after every directory is created, so a failed `mkdir` leaves the previous state intact. `run()` in `src/main.py` calls it with no arguments. |
| `resolve(member)` | Resolve one wishlist member against its root. Rejects members of an unregistered wishlist. |
| `ensure_all()` | Create the directory, or file parent, of every declared path. |
| `raw_dataset_path(name)` | Dynamic dataset directory under the data root. |
| `processed_dataset_path(name)` | Dynamic processed-dataset directory. |
| `model_path(name)` | Dynamic model file under the shared models directory. |
| `cache_path(name)` | Dynamic cache directory under the cache root. |

The four dynamic helpers sanitise `name` into a single path segment: spaces and hyphens become underscores. A `name` that is empty, absolute, or that would add a segment or `..` raises `ValueError`. A name from config or a CLI cannot escape its parent. Every call site passes a hardcoded literal.

## Cross-platform determinism

A declared path must name the same location everywhere. Three spellings break that, and are refused at declaration time:

| Refused | Why |
|---|---|
| `a\b` | Backslash is a filename character on POSIX but a separator on Windows. Creates one oddly named entry on Linux, a nested directory on Windows. Use `/`. |
| `model.` / `model ` | Windows strips trailing dots and spaces, so `model.` collides with `model` there but stays distinct on Linux. |
| `C:/Windows`, `\\server\share` | Absolute on Windows, relative-looking on POSIX. |

The same rules apply to chained segments and dynamic names, so `PathSpec` declarations and `model_path`-style helpers cannot drift apart. Every current declaration and name resolves to an identical segment list under both `PurePosixPath` and `PureWindowsPath`.

Reserved Windows device names (`CON`, `aux`, `nul`, `COM1`, …) are **not** rejected. None is used today. Add a check to `_reject` and `_sanitise` if one becomes plausible.

## Declaring paths

Each module declares what it needs in its own *wishlist*: an enum whose values are `PathSpec` objects.

```python
# src/search_module/paths.py
class SearchPaths(Enum):
    SEARCH_CACHE_DIR = PathSpec(PathRoot.CACHE, "search", is_dir=True)
```

A `PathSpec` is always relative to a root. Construction rejects anything absolute, containing `..`, or naming no location, plus any spelling that differs between Windows and POSIX (see [Cross-platform determinism](#cross-platform-determinism)).

`PathSpec` chains with `/`, and children inherit `root` and `is_dir`. Mark file children with `.as_file()`. An empty or `.` child is rejected: `PurePosixPath` drops both, so the chain would otherwise produce a silent second name for the same path:

```python
FRAME_CLASSIFIER_FILE = (_MODELS / "xgboost_frame_classifier.pkl").as_file()
```

Consumers read paths through the registry:

```python
resolve(ArgumentFramingPaths.TFIDF_VECTORIZER_FILE)  # declared path
model_path("xgboost_claim_extractor.json")           # dynamic name
```

## Testing

Pointing the registry at real user directories would modify the developer's machine and make results depend on whatever is already there.

`tests/conftest.py` redirects both roots into a pytest temporary directory. The fixture is `autouse`, so **tests need no changes to benefit**: a component resolving a path during a test writes to the temporary directory instead of the home directory.

To keep a component out of the registry entirely, pass it a path explicitly. `DiskCacheBackend` accepts `directory`, and the search tests use it.

### Rules

**1. Never resolve a path at import time.** Class-body and module-level code runs during collection, *before* any fixture, so the override cannot help:

```python
class XGBoostFrameClassifier:
    MODEL_PATH = config.data_directory / "models" / "model.pkl"  # too early
```

Resolve in `__init__` or a property instead:

```python
@property
def path_to_model(self) -> Path:
    return resolve(ArgumentFramingPaths.FRAME_CLASSIFIER_FILE)
```

**2. Keep the isolation fixture at least as broad as the fixtures that resolve paths.** Pytest builds wider scopes first, so a narrower override applies too late: the wider fixture has already constructed its component and written to the real directories. Two integration fixtures use `scope="module"`, which is why `isolated_path_roots` is module-scoped. Widen the override when adding a `module`- or `session`-scoped fixture that resolves a path.

**3. Declare new required paths in a wishlist.** A path `initialise()` does not know about is never created, so it will be missing from the temporary roots during tests. Declare the *parent* for anything with a dynamic name: `MODELS_DIR` covers every model file, `CACHES_DIR` covers every cache directory. Only the parent is declarable, since the names are not known in advance.

**4. Do not repeat a `(root, relative, is_dir)` triple within one wishlist.** `PathSpec` is hashable and used as an enum value, so a duplicate silently becomes an enum alias: the second name disappears from iteration and is skipped by directory creation.

**5. Do not mutate `registry._roots` outside `tests/conftest.py`.** It is module-global state that the fixture saves and restores around each module.

## Known caveats

- `Configuration` loads `config.toml` from the current working directory, so importing `src.configuration` outside the repository root raises `ConfigurationError`. The registry imports it lazily, so this surfaces only when a root is needed: `initialise(data_root, cache_root)` reads no configuration and works from any directory, while `initialise()` with no arguments, or a call that supplies only one root, still requires `config.toml`.
- Rule 4 is invisible at the point of declaration, which is why `PathSpec` carries the warning in its own docstring.