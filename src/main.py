"""A module being an entrypoint to the entire application."""

import asyncio

import typer

cli_app = typer.Typer(name="Contradictor", no_args_is_help=True)


@cli_app.command(name="api")
def start_api() -> None:
    """Start the Web API."""
    raise NotImplementedError("Start the FastAPI Web API.")


@cli_app.command(name="benchmark")
def execute_benchmark() -> None:
    """Perform a benchmark of Contradictor on the Golden Dataset."""
    # pylint: disable-next=import-outside-toplevel
    from src.benchmark.command import perform_benchmark

    asyncio.run(perform_benchmark())
