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
from .processing.progress import compute_progress as compute_progress_fn
from .io.hdf5_writer import initialize_h5
from .processing.pipeline import process_all as process_all_fn
from .download.downloader import download_dataset as download_dataset_fn
from .download.downloader import extract_zip as extract_zip_fn
from .download.downloader import DownloadOptions as DownloadOptionsCls

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


@app.command("progress")
def progress(
    raw_dir: Path = typer.Option(Path("data/raw"), exists=True, file_okay=False, resolve_path=True,
                                 help="Root directory of raw data"),
    h5_path: Path = typer.Option(Path("data/h5/HSIFoodIngr-64.h5"), exists=False, file_okay=True, resolve_path=True,
                                 help="Target HDF5 file (if exists, used to detect processed basenames)"),
    show_lists: bool = typer.Option(False, help="Print basenames lists for each category"),
) -> None:
    p = compute_progress_fn(raw_dir=raw_dir, h5_path=h5_path)
    typer.echo(f"basenames total: {p.total_basenames}")
    typer.echo(f"complete (all 4 files exist): {p.complete_total}")
    typer.echo(f"processed (in H5): {p.processed}")
    typer.echo(f"ready (complete but not processed): {p.ready_unprocessed}")
    typer.echo(f"incomplete (missing files): {p.incomplete}")
    if show_lists:
        if p.processed_basenames:
            typer.echo("processed:")
            for b in p.processed_basenames:
                typer.echo(f" - {b}")
        if p.ready_unprocessed_basenames:
            typer.echo("ready:")
            for b in p.ready_unprocessed_basenames:
                typer.echo(f" - {b}")
        if p.incomplete_basenames:
            typer.echo("incomplete:")
            for b in p.incomplete_basenames:
                missing = ",".join(p.missing_by_basename.get(b, []))
                typer.echo(f" - {b} (missing: {missing})")


@app.command("download")
def download(
    output_dir: Path = typer.Option(Path("data/raw"), file_okay=False, resolve_path=True, help="Output directory"),
    api_key: str | None = typer.Option(None, envvar="DATAVERSE_API_KEY", help="Dataverse API key"),
    base_url: str = typer.Option("https://dataverse.harvard.edu", help="Dataverse base URL"),
    persistent_id: str = typer.Option("doi:10.7910/DVN/E7WDNQ", help="Dataset persistent ID"),
    resume: bool = typer.Option(True, help="Resume partial downloads"),
    force: bool = typer.Option(False, help="Force re-download (overwrite ZIP)"),
    extract: bool = typer.Option(True, help="Extract ZIP after download"),
    extract_dir: Path | None = typer.Option(None, file_okay=False, resolve_path=True, help="Extraction directory (defaults to output_dir)"),
) -> None:
    opts = DownloadOptionsCls(
        output_dir=output_dir,
        api_key=api_key,
        base_url=base_url,
        persistent_id=persistent_id,
        resume=resume,
        force=force,
    )
    zip_path = download_dataset_fn(opts)
    typer.echo(str(zip_path))
    if extract:
        dest = extract_dir if extract_dir else output_dir
        extract_zip_fn(zip_path, dest)
        typer.echo(str(dest))


def run() -> None:  # For `python -m hsifoodingr.cli`
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
