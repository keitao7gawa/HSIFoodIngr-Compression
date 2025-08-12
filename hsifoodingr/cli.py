from __future__ import annotations

import sys
from pathlib import Path

import typer

from .utils import setup_rich_logging, get_logger
from .processing import build_manifest as build_manifest_fn
from .processing import build_ingredient_map as build_ingredient_map_fn
from .io.hdf5_writer import initialize_h5

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


@app.command("init-h5")
def init_h5(
    h5_path: Path = typer.Option(Path("data/h5/HSIFoodIngr-64.h5"), resolve_path=True,
                                 help="Output HDF5 file path"),
    ingredient_map_path: Path = typer.Option(..., exists=True, file_okay=True, resolve_path=True,
                                             help="Path to ingredient_map.json"),
    wavelengths_path: Path = typer.Option(..., exists=True, file_okay=True, resolve_path=True,
                                          help="Path to wavelengths list (one value per line)"),
    overwrite: bool = typer.Option(False, help="Overwrite existing HDF5 file if present"),
) -> None:
    """Initialize an empty HDF5 file with metadata and resizable datasets."""
    import json
    import numpy as np

    ingredient_map = json.loads(ingredient_map_path.read_text())
    # wavelengths file: accept CSV or plain lines
    txt = wavelengths_path.read_text().strip()
    if "," in txt:
        values = [float(x) for x in txt.split(",") if x.strip()]
    else:
        values = [float(x) for x in txt.splitlines() if x.strip()]
    wavelengths = np.asarray(values, dtype=np.float32)

    initialize_h5(h5_path=h5_path, ingredient_map=ingredient_map, wavelengths=wavelengths, overwrite=overwrite)
    typer.echo(str(h5_path))


def run() -> None:  # For `python -m hsifoodingr.cli`
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
