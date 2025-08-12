from __future__ import annotations

import sys
from pathlib import Path

import typer

from .utils import setup_rich_logging, get_logger

app = typer.Typer(add_completion=False, help="HSIFoodIngr-64 HDF5 builder CLI")


@app.callback()
def main(quiet: bool = typer.Option(False, "--quiet", "-q", help="Reduce log verbosity")) -> None:
    """CLI entrypoint: configure logging and shared state."""
    setup_rich_logging()
    logger = get_logger("hsifoodingr")
    if quiet:
        import logging

        logger.setLevel(logging.WARNING)


@app.command("version")
def version() -> None:
    from . import __version__

    typer.echo(__version__)


def run() -> None:  # For `python -m hsifoodingr.cli`
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
