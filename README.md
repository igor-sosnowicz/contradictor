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
