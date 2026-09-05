# AGENTS.md

This file contains operational guidance for OpenCode agents working with this repository.

## Setup and Environment

- Uses `uv` for Python dependency management
- Requires Python 3.13 (specified in pyproject.toml)
- Install developer dependencies with: `uv sync --dev`
- Install pre-commit hooks with: `uv run pre-commit install`

## Project Structure

- Main application entrypoint is `src/main.py` with script defined as `app = "src.main:main"` in pyproject.toml
- Source code is organized under `src/` directory with modules:
  - `argument_detection/`
  - `argument_framing/`
  - `configuration.py`
  - `contr_argument_dataset_tools/`
  - `data_models/`
  - `search_module/`
  - `style_extraction/`
  - `utils/`

## Development Workflow

- Run tests with: `uv run pytest` (or specific test files)
- Type checking with: `uv run mypy src`
- Linting and formatting with: `uv run ruff check --fix src` and `uv run ruff format src`
- Pre-commit hooks handle auto-formatting and validation
- Commit message format: `<type>: Short description` with types including `feat:`, `fix:`, `test:`, `ci:`, `chore:`, `build:`, `docs:`, `style:`, `refactor:`, `perf:`

## Testing

- Tests organized in `tests/` directory by module
- Run specific tests with: `uv run pytest tests/<module>`
- Test coverage via pytest-cov
