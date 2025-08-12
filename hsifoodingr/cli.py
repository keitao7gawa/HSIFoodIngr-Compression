from __future__ import annotations

import sys
from pathlib import Path

import typer

from .utils import setup_rich_logging, get_logger
from .processing import build_manifest as build_manifest_fn
from .processing import build_ingredient_map as build_ingredient_map_fn

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


@app.command("build-manifest")
def build_manifest(
    raw_dir: Path = typer.Option(Path("data/raw"), exists=True, file_okay=False, resolve_path=True,
                                 help="Root directory containing raw dataset files"),
    artifacts_dir: Path = typer.Option(Path("data/artifacts"), file_okay=False, resolve_path=True,
                                       help="Directory to write manifest outputs"),
) -> None:
    """Scan raw data and write file manifest and processable lists."""
    build_manifest_fn(raw_dir=raw_dir, artifacts_dir=artifacts_dir)


@app.command("build-ingredient-map")
def build_ingredient_map(
    raw_dir: Path = typer.Option(Path("data/raw"), exists=True, file_okay=False, resolve_path=True,
                                 help="Root directory containing raw dataset files"),
    artifacts_dir: Path = typer.Option(Path("data/artifacts"), file_okay=False, resolve_path=True,
                                       help="Directory to write artifact outputs"),
) -> None:
    """Build global ingredient name→ID map from all JSON files and write JSON."""
    build_ingredient_map_fn(raw_dir=raw_dir, artifacts_dir=artifacts_dir)


def run() -> None:  # For `python -m hsifoodingr.cli`
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
