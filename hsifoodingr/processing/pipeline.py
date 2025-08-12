from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..utils import get_logger
from ..io.hdf5_writer import open_h5, append_sample, contains_basename
from .manifest import build_manifest
from .mask_builder import build_mask, load_ingredient_map

logger = get_logger(__name__)


def _read_processable_list(artifacts_dir: Path) -> List[Dict[str, str]]:
    details_path = artifacts_dir / "processable.json"
    if not details_path.exists():
        return []
    try:
        data = json.loads(details_path.read_text())
        # basic validation
        if isinstance(data, list):
            cleaned: List[Dict[str, str]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                required = {"basename", "hdr", "dat", "png", "json"}
                if not required.issubset(item.keys()):
                    continue
                cleaned.append({k: str(item[k]) for k in required})
            return cleaned
    except Exception as e:
        logger.warning("Failed to read %s: %s", details_path, e)
    return []


def _ensure_manifest(raw_dir: Path, artifacts_dir: Path) -> List[Dict[str, str]]:
    items = _read_processable_list(artifacts_dir)
    if items:
        return items
    logger.info("No processable.json found. Building manifest...")
    build_manifest(raw_dir=raw_dir, artifacts_dir=artifacts_dir)
    return _read_processable_list(artifacts_dir)


def _append_failure(artifacts_dir: Path, basename: str, error: str) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "failures.log").open("a").write(f"{basename}\t{error}\n")


def process_all(
    raw_dir: Path,
    artifacts_dir: Path,
    h5_path: Path,
    skip_existing: bool = True,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """Process all samples into HDF5. Returns (num_processed, num_skipped)."""
    raw_dir = Path(raw_dir)
    artifacts_dir = Path(artifacts_dir)
    h5_path = Path(h5_path)

    items = _ensure_manifest(raw_dir, artifacts_dir)
    if not items:
        logger.warning("No processable items found under %s", raw_dir)
        return 0, 0

    ingredient_map_path = artifacts_dir / "ingredient_map.json"
    if not ingredient_map_path.exists():
        raise FileNotFoundError(f"Missing ingredient_map.json at {ingredient_map_path}")
    ingredient_map = load_ingredient_map(ingredient_map_path)

    processed = 0
    skipped = 0

    with open_h5(h5_path, mode="a") as f:
        existing_set: set[str] = set()
        if skip_existing:
            # Build set of existing basenames
            ds = f["metadata/image_basenames"]
            if ds.shape[0] > 0:
                try:
                    existing_list = ds.asstr()[...].tolist()
                except Exception:
                    existing_list = [
                        v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                        for v in ds[...].tolist()
                    ]
                existing_set = set(existing_list)

        # Lazy imports to avoid import-time side effects that complicate testing/mocking
        from ..io import envi_reader as envi_reader_module  # type: ignore
        from ..io import rgb_reader as rgb_reader_module  # type: ignore
        from ..io import json_reader as json_reader_module  # type: ignore

        for item in items:
            basename = item["basename"]
            if skip_existing and basename in existing_set:
                skipped += 1
                continue
            try:
                # Read inputs
                hsi = envi_reader_module.read_envi_hsi(Path(item["hdr"]), Path(item["dat"]))
                rgb = rgb_reader_module.read_rgb(Path(item["png"]))
                ann = json_reader_module.read_annotation(Path(item["json"]))

                # Build mask
                height, width = int(hsi.shape[0]), int(hsi.shape[1])
                mask = build_mask(ann, ingredient_map, image_size_hw=(height, width))

                # Append
                append_sample(
                    f,
                    hsi=hsi,
                    rgb=rgb,
                    mask=mask,
                    basename=basename,
                    dish_label=ann.dish if ann.dish else "",
                )
                processed += 1
            except Exception as e:  # pragma: no cover - failure path
                logger.exception("Failed processing %s: %s", basename, e)
                _append_failure(artifacts_dir, basename, str(e))

            if limit is not None and processed >= limit:
                break

    logger.info("Process completed: processed=%d, skipped=%d", processed, skipped)
    return processed, skipped
