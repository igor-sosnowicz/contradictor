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
