from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class VerifyConfig:
    expected_height: int = 512
    expected_width: int = 512
    expected_bands: int = 204
    # Number of samples to scan for NaN/Inf checks. None scans all.
    scan_limit: Optional[int] = None
    # Number of samples per chunk to read to control memory usage
    chunk_size: int = 16


@dataclass
class VerifyResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    num_samples: int


def _read_ingredient_map(f: h5py.File) -> Dict[str, int]:
    ds = f["metadata/ingredient_map"]
    raw = ds[0]
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def verify_h5(h5_path: Path, config: VerifyConfig | None = None) -> VerifyResult:
    config = config or VerifyConfig()

    h5_path = Path(h5_path)
    errors: List[str] = []
    warnings: List[str] = []

    if not h5_path.exists():
        return VerifyResult(False, [f"H5 file not found: {h5_path}"], warnings, 0)

    with h5py.File(h5_path, "r") as f:
        # Presence checks
        required = [
            "hsi",
            "rgb",
            "masks",
            "metadata/image_basenames",
            "metadata/dish_labels",
            "metadata/ingredient_map",
            "metadata/wavelengths",
        ]
        for key in required:
            if key not in f:
                errors.append(f"Missing dataset/group: {key}")

        if errors:
            return VerifyResult(False, errors, warnings, 0)

        hsi = f["hsi"]
        rgb = f["rgb"]
        masks = f["masks"]
        names = f["metadata/image_basenames"]
        dish = f["metadata/dish_labels"]
        wavelengths = f["metadata/wavelengths"]

        # Dtype checks
        if hsi.dtype != np.float32:
            errors.append(f"hsi dtype {hsi.dtype} != float32")
        if rgb.dtype != np.uint8:
            errors.append(f"rgb dtype {rgb.dtype} != uint8")
        if masks.dtype != np.uint8:
            errors.append(f"masks dtype {masks.dtype} != uint8")

        # Shape checks
        expected_hsi = (config.expected_height, config.expected_width, config.expected_bands)
        expected_rgb = (config.expected_height, config.expected_width, 3)
        expected_mask = (config.expected_height, config.expected_width)
        if tuple(hsi.shape[1:]) != expected_hsi:
            errors.append(f"hsi shape {tuple(hsi.shape[1:])} != expected {expected_hsi}")
        if tuple(rgb.shape[1:]) != expected_rgb:
            errors.append(f"rgb shape {tuple(rgb.shape[1:])} != expected {expected_rgb}")
        if tuple(masks.shape[1:]) != expected_mask:
            errors.append(f"masks shape {tuple(masks.shape[1:])} != expected {expected_mask}")

        # Count alignment
        n = int(hsi.shape[0])
        for ds_name, ds in ("rgb", rgb), ("masks", masks), ("image_basenames", names), ("dish_labels", dish):
            if int(ds.shape[0]) != n:
                errors.append(f"{ds_name} samples {int(ds.shape[0])} != hsi samples {n}")

        # Wavelength shape
        if wavelengths.shape != (config.expected_bands,):
            errors.append(
                f"wavelengths shape {wavelengths.shape} != expected ({config.expected_bands},)"
            )

        # Duplicate basenames
        if names.shape[0] > 0:
            try:
                name_list = names.asstr()[...].tolist()
            except Exception:
                name_list = [
                    v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                    for v in names[...].tolist()
                ]
            unique_count = len(set(name_list))
            if unique_count != len(name_list):
                errors.append(
                    f"Duplicate basenames detected: total={len(name_list)} unique={unique_count}"
                )

        # NaN/Inf checks on HSI (optionally limited)
        if n > 0:
            to_scan = n if config.scan_limit is None else min(n, int(config.scan_limit))
            if to_scan > 0:
                for start in range(0, to_scan, config.chunk_size):
                    end = min(to_scan, start + config.chunk_size)
                    batch = hsi[start:end]
                    if not np.isfinite(batch).all():
                        # Identify what occurred
                        if np.isnan(batch).any():
                            errors.append("NaN values found in HSI data")
                        if np.isinf(batch).any():
                            errors.append("Inf values found in HSI data")
                        break

    return VerifyResult(len(errors) == 0, errors, warnings, n if 'n' in locals() else 0)


@dataclass
class Summary:
    num_samples: int
    image_shape_hsi: Tuple[int, int, int]
    image_shape_rgb: Tuple[int, int, int]
    image_shape_mask: Tuple[int, int]
    num_classes: int
    class_counts: Dict[str, int]
    wavelengths_min: float
    wavelengths_max: float
    wavelengths_mean: float
    top_dishes: List[Tuple[str, int]]


def summarize_h5(h5_path: Path, top_k: int = 20) -> Summary:
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        hsi = f["hsi"]
        rgb = f["rgb"]
        masks = f["masks"]
        dish = f["metadata/dish_labels"]
        wavelengths = f["metadata/wavelengths"][...]

        ingredient_map = _read_ingredient_map(f)
        id_to_name = {v: k for k, v in ingredient_map.items()} if ingredient_map else {}
        max_id = max(id_to_name.keys(), default=int(masks[:1].max()) if masks.shape[0] > 0 else 0)

        # Class histogram across masks
        counts = np.zeros(max_id + 1, dtype=np.int64)
        n = int(masks.shape[0])
        if n > 0:
            chunk = 16
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                batch = masks[start:end]
                # Flatten then bincount
                bc = np.bincount(batch.reshape(-1), minlength=max_id + 1)
                # Ensure int64 for accumulation safety
                counts[: len(bc)] += bc.astype(np.int64)

        class_counts: Dict[str, int] = {}
        for idx, cnt in enumerate(counts.tolist()):
            name = id_to_name.get(idx, str(idx))
            class_counts[name] = int(cnt)

        # Dish distribution (top-k)
        top_dishes: List[Tuple[str, int]] = []
        if int(dish.shape[0]) > 0:
            try:
                dishes_list = dish.asstr()[...].tolist()
            except Exception:
                dishes_list = [
                    v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                    for v in dish[...].tolist()
                ]
            from collections import Counter

            c = Counter(dishes_list)
            top_dishes = c.most_common(top_k)

        return Summary(
            num_samples=int(hsi.shape[0]),
            image_shape_hsi=tuple(hsi.shape[1:]),
            image_shape_rgb=tuple(rgb.shape[1:]),
            image_shape_mask=tuple(masks.shape[1:]),
            num_classes=len(class_counts),
            class_counts=class_counts,
            wavelengths_min=float(np.min(wavelengths) if wavelengths.size else float("nan")),
            wavelengths_max=float(np.max(wavelengths) if wavelengths.size else float("nan")),
            wavelengths_mean=float(np.mean(wavelengths) if wavelengths.size else float("nan")),
            top_dishes=top_dishes,
        )
