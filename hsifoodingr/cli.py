from __future__ import annotations

import sys
from pathlib import Path

import typer

from .utils import setup_rich_logging, get_logger
from .processing import build_manifest as build_manifest_fn
from .processing import build_ingredient_map as build_ingredient_map_fn
from .processing import verify_h5 as verify_h5_fn
from .processing import summarize_h5 as summarize_h5_fn
from .processing import VerifyConfig as VerifyConfigCls
from .io.hdf5_writer import initialize_h5
from .processing.pipeline import process_all as process_all_fn

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


@app.command("process")
def process(
    raw_dir: Path = typer.Option(Path("data/raw"), exists=True, file_okay=False, resolve_path=True,
                                 help="Root directory of raw data"),
    artifacts_dir: Path = typer.Option(Path("data/artifacts"), file_okay=False, resolve_path=True,
                                       help="Artifacts directory (manifest, ingredient_map.json, logs)"),
    h5_path: Path = typer.Option(Path("data/h5/HSIFoodIngr-64.h5"), exists=True, file_okay=True, resolve_path=True,
                                 help="Target HDF5 file"),
    limit: int = typer.Option(None, help="Process at most N new samples"),
    no_skip_existing: bool = typer.Option(False, help="Do not skip already processed basenames"),
    allow_missing_json: bool = typer.Option(False, help="Allow processing samples without JSON (creates empty masks and dish labels)"),
) -> None:
    processed, skipped = process_all_fn(
        raw_dir=raw_dir,
        artifacts_dir=artifacts_dir,
        h5_path=h5_path,
        skip_existing=not no_skip_existing,
        limit=limit,
        allow_missing_json=allow_missing_json,
    )
    typer.echo(f"processed={processed} skipped={skipped}")


@app.command("verify")
def verify(
    h5_path: Path = typer.Option(Path("data/h5/HSIFoodIngr-64.h5"), exists=True, file_okay=True, resolve_path=True,
                                 help="HDF5 file to verify"),
    scan_limit: int | None = typer.Option(None, help="Scan at most N samples for NaN/Inf checks"),
    chunk_size: int = typer.Option(16, help="Chunk size when scanning samples"),
) -> None:
    cfg = VerifyConfigCls(scan_limit=scan_limit, chunk_size=chunk_size)
    result = verify_h5_fn(h5_path, config=cfg)
    if result.ok:
        typer.echo(f"OK: {result.num_samples} samples")
    else:
        typer.echo("FAILED:")
        for e in result.errors:
            typer.echo(f" - {e}")
        raise typer.Exit(code=1)


@app.command("summary")
def summary(
    h5_path: Path = typer.Option(Path("data/h5/HSIFoodIngr-64.h5"), exists=True, file_okay=True, resolve_path=True,
                                 help="HDF5 file to summarize"),
    top_k: int = typer.Option(20, help="Show top-K dish labels"),
) -> None:
    s = summarize_h5_fn(h5_path, top_k=top_k)
    # Pretty print concise summary
    typer.echo(f"samples: {s.num_samples}")
    typer.echo(f"hsi: {s.image_shape_hsi}; rgb: {s.image_shape_rgb}; mask: {s.image_shape_mask}")
    typer.echo(f"classes: {s.num_classes}; wavelengths[min/mean/max]: {s.wavelengths_min:.2f}/{s.wavelengths_mean:.2f}/{s.wavelengths_max:.2f}")
    if s.top_dishes:
        typer.echo("top dishes:")
        for name, cnt in s.top_dishes:
            typer.echo(f" - {name}: {cnt}")


def run() -> None:  # For `python -m hsifoodingr.cli`
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
