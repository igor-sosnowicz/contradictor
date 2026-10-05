# Contradictor

**Contradictor** finds counterarguments to arguments detected in the input text, written in a similar style as the input text.

## Setup

This section describes the setup of Contradictor.

### User

To be written... (We should use Docker.)

### Developer

Prerequisites:

- `git` – code version control
- `uv` – dependency and Python management

**Setup process**:

1. Clone the repository.

    ```bash
    git clone git@github.com:igor-sosnowicz/contradictor.git
    ```

2. Install the developer dependencies.

    ```bash
    uv sync --dev
    ```

3. Install the pre-commit hooks.

    ```bash
    uv run pre-commit install
    ```

## Conventions

The following development and maintenance conventions were decided and should be followed in the project:

- commit message format: `<type>: Short description of changes` (see a section below to discover commit types we use)
- branch name format: `<contributor_first_name>-short-description`
- test fixture scope: the path isolation fixture must be at least as broad as the widest-scoped fixture that resolves a path (see [Test path isolation](#test-path-isolation))

### Commit Types

The [conventional commit specification](https://www.conventionalcommits.org/en/v1.0.0/) require a type to be present in a commit message.
We use the following types in the project:

- `feat:` – new features
- `fix:` – bug fixes
- `test:` – automated test definitions
- `ci:` – CI/CD pipeline
- `chore:` – general maintenance that does not fit the others
- `build:` – Docker-related changes, container configuration
- `docs:` – documentation
- `style:` – reformatting, code style rules
- `refactor:` – improvement of existing code without adding new features
- `perf:` – performance improvements

### Test Path Isolation

Contradictor resolves every filesystem path through a single registry, rooted at platform-appropriate data and cache directories.

Tests must not write to those real directories, so `tests/conftest.py` redirects the roots into a pytest temporary directory. The fixture is `autouse`, which means **new tests need no changes to benefit** — anything that writes a file during a test writes it to the temporary directory instead of your home directory.

Two rules matter when adding code or fixtures:

- **Do not resolve a path at import time.** Class-body and module-level code runs before any fixture, so it would capture the real directories. Resolve paths in `__init__` or a property.
- **Keep the isolation fixture at least as broad as any fixture that resolves paths.** Pytest builds wider scopes first, so a narrower override is applied too late.

For the registry API, the full rule list, and how to opt a component out, see [`docs/unified_path_system.md`](docs/unified_path_system.md).
