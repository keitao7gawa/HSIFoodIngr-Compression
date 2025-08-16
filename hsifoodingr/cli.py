from __future__ import annotations

import sys
from pathlib import Path

import typer

from .utils import setup_rich_logging, get_logger
from .utils.archive import safe_extract, cleanup_path
from .processing import build_manifest as build_manifest_fn
from .processing import build_ingredient_map as build_ingredient_map_fn
from .processing import verify_h5 as verify_h5_fn
from .processing import summarize_h5 as summarize_h5_fn
from .processing import VerifyConfig as VerifyConfigCls
from .processing.progress import compute_progress as compute_progress_fn
from .io.hdf5_writer import initialize_h5
from .processing.pipeline import process_all as process_all_fn
from .processing.pipeline import ingest_labels as ingest_labels_fn
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


@app.command("process-archives")
def process_archives(
    input_dir: Path = typer.Option(Path("data/raw"), file_okay=False, resolve_path=True, help="Directory containing downloaded archives or extracted folders"),
    work_dir: Path = typer.Option(Path("data/tmp/extract"), file_okay=False, resolve_path=True, help="Temporary extraction directory"),
    output_h5: Path = typer.Option(Path("data/h5/HSIFoodIngr-64.h5"), file_okay=True, resolve_path=True, help="Target HDF5 file path (created if missing)"),
    archive_glob: str = typer.Option("HSIFoodIngr-64_*", help="Glob to select archives or folders to process"),
    remove_archive: bool = typer.Option(False, help="Remove archive file after successful processing"),
    remove_extracted: bool = typer.Option(True, help="Remove extracted temporary folder after processing (only applies to extracted archives)"),
    auto_bootstrap: bool = typer.Option(False, help="Auto-create manifest, ingredient_map, and initialize H5 if missing"),
    workers: int = typer.Option(1, min=1, max=4, help="Parallelism level (archive-level). Use 1 to avoid HDF5 contention"),
    dry_run: bool = typer.Option(False, help="Show planned actions without performing them"),
    allow_missing_json: bool = typer.Option(False, help="Allow appending samples without JSON (placeholder mask/label); use 'ingest-labels' later to fill labels"),
) -> None:
    """Process downloaded archives end-to-end: extract → append to HDF5 → cleanup."""
    import json
    import time
    import os
    import logging

    logger = get_logger("hsifoodingr.process_archives")
    artifacts_dir = Path("data/artifacts").resolve()
    processed_marker_dir = artifacts_dir / "processed_archives"
    processed_marker_dir.mkdir(parents=True, exist_ok=True)

    # Attach a file logger to capture full run details (idempotent)
    logs_dir = artifacts_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / "process-archives.log"
    root_logger = logging.getLogger()
    need_attach = True
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                if getattr(h, "baseFilename", "").endswith(str(logfile)):
                    need_attach = False
                    break
            except Exception:
                continue
    if need_attach:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
        logger.info("File logging enabled: %s", logfile)

    # Bootstrap artifacts and H5 as needed
    if auto_bootstrap:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Manifest for the full input_dir (may be directories today)
        try:
            build_manifest_fn(raw_dir=input_dir, artifacts_dir=artifacts_dir)
            logger.info("bootstrap: build-manifest done (raw_dir=%s)", input_dir)
        except Exception as e:
            # non-fatal
            logger.debug("bootstrap: build-manifest skipped/failed: %s", e)
        try:
            build_ingredient_map_fn(raw_dir=input_dir, artifacts_dir=artifacts_dir)
            logger.info("bootstrap: build-ingredient-map done (raw_dir=%s)", input_dir)
        except Exception as e:
            logger.debug("bootstrap: build-ingredient-map skipped/failed: %s", e)

        ingredient_map_path = artifacts_dir / "ingredient_map.json"
        wavelengths_path = artifacts_dir / "wavelengths.txt"
        if ingredient_map_path.exists() and not output_h5.exists():
            try:
                import numpy as np

                ingredient_map = json.loads(ingredient_map_path.read_text())
                txt = wavelengths_path.read_text().strip()
                if "," in txt:
                    values = [float(x) for x in txt.split(",") if x.strip()]
                else:
                    values = [float(x) for x in txt.splitlines() if x.strip()]
                wavelengths = np.asarray(values, dtype=np.float32)
                initialize_h5(h5_path=output_h5, ingredient_map=ingredient_map, wavelengths=wavelengths, overwrite=False)
                logger.info("bootstrap: init-h5 done (h5=%s)", output_h5)
            except Exception as e:
                logger.debug("bootstrap: init-h5 skipped/failed: %s", e)

    # Enumerate candidates: archives and folders
    items = []
    for p in input_dir.glob(archive_glob):
        if p.is_file() or p.is_dir():
            items.append(p)

    if not items:
        msg = f"No archives or folders matched: glob={archive_glob} in {input_dir}"
        logger.info(msg)
        typer.echo("No archives or folders matched. Nothing to do.")
        return

    # Helper to compute a base name without multi-extensions
    def _base_no_ext(path: Path) -> str:
        name = path.name
        while True:
            base, ext = os.path.splitext(name)
            if not ext:
                return base
            name = base

    total_processed = 0
    total_skipped = 0
    started = time.time()

    # Log plan overview
    logger.info(
        "begin process-archives: input_dir=%s work_dir=%s out_h5=%s glob=%s remove_archive=%s remove_extracted=%s auto_bootstrap=%s workers=%d dry_run=%s items=%d",
        input_dir,
        work_dir,
        output_h5,
        archive_glob,
        remove_archive,
        remove_extracted,
        auto_bootstrap,
        workers,
        dry_run,
        len(items),
    )
    sample_items = [str(p) for p in sorted(items)[:5]]
    logger.info("matched examples: %s%s", sample_items, " ..." if len(items) > 5 else "")

    for item in sorted(items):
        base = _base_no_ext(item)
        done_marker = processed_marker_dir / f"{base}.done"
        if done_marker.exists():
            logger.info("skip (already done): item=%s marker=%s", item, done_marker)
            continue

        plan = {
            "item": str(item),
            "base": base,
            "action": "extract-and-process" if item.is_file() else "process-folder",
            "work_dir": str(work_dir / base),
            "remove_archive": bool(remove_archive and item.is_file()),
            "remove_extracted": bool(remove_extracted and item.is_file()),
        }
        if dry_run:
            logger.info("plan: %s", plan)
            typer.echo(f"PLAN: {plan}")
            continue

        try:
            # extraction
            if item.is_file():
                t0 = time.time()
                extracted_root = safe_extract(item, work_dir)
                logger.info("extracted: item=%s -> root=%s (%.2fs)", item, extracted_root, time.time() - t0)
            else:
                extracted_root = item

            # process
            t1 = time.time()
            processed, skipped = process_all_fn(
                raw_dir=extracted_root,
                artifacts_dir=artifacts_dir,
                h5_path=output_h5,
                skip_existing=True,
                limit=None,
                allow_missing_json=allow_missing_json,
            )
            total_processed += processed
            total_skipped += skipped
            logger.info(
                "processed archive: base=%s processed=%d skipped=%d (%.2fs)",
                base,
                processed,
                skipped,
                time.time() - t1,
            )

            # cleanup extracted only if we created it from an archive
            if item.is_file() and remove_extracted:
                cleanup_path(extracted_root)
                logger.info("removed extracted: %s", extracted_root)

            # remove archive if requested
            if item.is_file() and remove_archive:
                cleanup_path(item)
                logger.info("removed archive: %s", item)

            done_marker.write_text("ok")
            logger.info("done marker created: %s", done_marker)
        except Exception as e:
            logger.exception("Failed processing %s: %s", item, e)
            (artifacts_dir / "failures.log").open("a").write(f"[archive] {item}\t{e}\n")

    elapsed = time.time() - started
    logger.info(
        "end process-archives: total processed=%d skipped=%d archives=%d elapsed=%.1fs",
        total_processed,
        total_skipped,
        len(items),
        elapsed,
    )
    typer.echo(f"done: processed={total_processed} skipped={total_skipped} in {elapsed:.1f}s")


@app.command("ingest-labels")
def ingest_labels(
    labels_root: Path = typer.Option(..., exists=True, file_okay=False, resolve_path=True, help="Root directory containing REFLECTANCE_*.json files (recursively scanned)"),
    artifacts_dir: Path = typer.Option(Path("data/artifacts"), file_okay=False, resolve_path=True, help="Artifacts directory (for ingredient_map.json and logs)"),
    h5_path: Path = typer.Option(Path("data/h5/HSIFoodIngr-64.h5"), exists=True, file_okay=True, resolve_path=True, help="Target HDF5 file"),
) -> None:
    """Update masks and dish labels in-place from external labels (JSON) by basename.

    Use after appending HDR/DAT/PNG without JSON.
    """
    updated, not_found = ingest_labels_fn(labels_root=labels_root, artifacts_dir=artifacts_dir, h5_path=h5_path)
    typer.echo(f"labels: updated={updated} not_found={not_found}")

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
    force: bool = typer.Option(False, help="Force re-download for existing files"),
) -> None:
    opts = DownloadOptionsCls(
        output_dir=output_dir,
        api_key=api_key,
        base_url=base_url,
        persistent_id=persistent_id,
        resume=resume,
        force=force,
    )
    marker = download_dataset_fn(opts)
    typer.echo(str(marker))


def run() -> None:  # For `python -m hsifoodingr.cli`
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
