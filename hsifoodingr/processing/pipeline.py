from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..utils import get_logger
from ..io.hdf5_writer import open_h5, append_sample, contains_basename, update_labels_by_basename
from .manifest import build_manifest
from .mask_builder import build_mask, load_ingredient_map

logger = get_logger(__name__)


def _read_processable_list(artifacts_dir: Path) -> List[Dict[str, str]]:
    details_path = artifacts_dir / "processable.json"
    if not details_path.exists():
        return []
    try:
        data = json.loads(details_path.read_text(encoding="utf-8"))
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
    allow_missing_json: bool = False,
) -> Tuple[int, int]:
    """Process all samples into HDF5. Returns (num_processed, num_skipped)."""
    raw_dir = Path(raw_dir)
    artifacts_dir = Path(artifacts_dir)
    h5_path = Path(h5_path)

    # Decide how to enumerate items
    if allow_missing_json:
        # Build items from available triplets (.hdr, .dat, .png); JSON optional
        files = [p for p in raw_dir.rglob("*") if p.is_file()]
        from .manifest import _group_by_basename  # type: ignore

        groups = _group_by_basename(files)
        items: List[Dict[str, str]] = []
        for basename, paths in groups.items():
            mapping = {p.suffix.lower(): p for p in paths}
            if all(k in mapping for k in [".hdr", ".dat", ".png"]):
                items.append(
                    {
                        "basename": basename,
                        "hdr": str(mapping[".hdr"]),
                        "dat": str(mapping[".dat"]),
                        "png": str(mapping[".png"]),
                        "json": str(mapping[".json"]) if ".json" in mapping else "",
                    }
                )
        items.sort(key=lambda d: d["basename"])  # stable order
    else:
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
                ann_path_str = item.get("json", "")
                has_json = bool(ann_path_str and Path(ann_path_str).exists())
                if has_json:
                    ann = json_reader_module.read_annotation(Path(ann_path_str))
                    height, width = int(hsi.shape[0]), int(hsi.shape[1])
                    mask = build_mask(ann, ingredient_map, image_size_hw=(height, width))
                    append_sample(
                        f,
                        hsi=hsi,
                        rgb=rgb,
                        mask=mask,
                        basename=basename,
                        dish_label=ann.dish if ann.dish else "",
                    )
                else:
                    # If json is missing and allowed, append with empty labels
                    if allow_missing_json:
                        height, width = int(hsi.shape[0]), int(hsi.shape[1])
                        empty_mask = np.zeros((height, width), dtype=np.uint8)
                        append_sample(
                            f,
                            hsi=hsi,
                            rgb=rgb,
                            mask=empty_mask,
                            basename=basename,
                            dish_label="",
                        )
                    else:
                        raise FileNotFoundError(f"Missing JSON for {basename}")
                processed += 1
            except Exception as e:  # pragma: no cover - failure path
                logger.exception("Failed processing %s: %s", basename, e)
                _append_failure(artifacts_dir, basename, str(e))

            if limit is not None and processed >= limit:
                break

    logger.info("Process completed: processed=%d, skipped=%d", processed, skipped)
    return processed, skipped


def ingest_labels(
    labels_root: Path,
    artifacts_dir: Path,
    h5_path: Path,
) -> Tuple[int, int]:
    """Update masks and dish labels for existing samples from JSON labels directory.

    Returns (updated, not_found).
    """
    from ..io import json_reader as json_reader_module  # type: ignore

    artifacts_dir = Path(artifacts_dir)
    h5_path = Path(h5_path)
    labels_root = Path(labels_root)

    ingredient_map_path = artifacts_dir / "ingredient_map.json"
    if not ingredient_map_path.exists():
        raise FileNotFoundError(f"Missing ingredient_map.json at {ingredient_map_path}")
    ingredient_map = load_ingredient_map(ingredient_map_path)

    # collect json files
    json_files = [p for p in labels_root.rglob("*.json") if p.is_file()]
    updated = 0
    not_found = 0
    with open_h5(h5_path, mode="a") as f:
        # Ensure H5 stores the latest ingredient_map so downstream summary has correct class count
        try:
            f["metadata/ingredient_map"][0] = json.dumps(ingredient_map, ensure_ascii=False)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to update ingredient_map in H5: %s", e)
        for jp in json_files:
            basename = jp.stem
            try:
                ann = json_reader_module.read_annotation(jp)
                # Build mask size from H5 dataset shape
                height, width = int(f["masks"].shape[1]), int(f["masks"].shape[2])
                mask = build_mask(ann, ingredient_map, image_size_hw=(height, width))
                # Determine dish label: prefer explicit ann.dish; otherwise derive from ingredient names
                dish_label = ann.dish.strip() if getattr(ann, "dish", "") else ""
                if not dish_label:
                    # Heuristic: take the most common prefix before '/' across ingredient names
                    prefixes: Dict[str, int] = {}
                    for ing in ann.ingredients:
                        name = (ing.name or "").strip()
                        if not name:
                            continue
                        prefix = name.split("/", 1)[0].strip()
                        if not prefix:
                            continue
                        prefixes[prefix] = prefixes.get(prefix, 0) + 1
                    if prefixes:
                        dish_label = max(prefixes.items(), key=lambda kv: (kv[1], kv[0]))[0]
                ok = update_labels_by_basename(f, basename, mask=mask, dish_label=dish_label)
                if ok:
                    updated += 1
                else:
                    not_found += 1
            except Exception as e:  # pragma: no cover - failure path
                logger.exception("Failed ingesting label for %s: %s", basename, e)
                _append_failure(artifacts_dir, basename, str(e))

    logger.info("Label ingest completed: updated=%d, not_found=%d", updated, not_found)
    return updated, not_found
