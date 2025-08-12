from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class H5Shapes:
    height: int = 512
    width: int = 512
    bands: int = 204

    @property
    def hsi(self) -> Tuple[int, int, int]:
        return (self.height, self.width, self.bands)

    @property
    def rgb(self) -> Tuple[int, int, int]:
        return (self.height, self.width, 3)

    @property
    def mask(self) -> Tuple[int, int]:
        return (self.height, self.width)


DEFAULT_CHUNKS = {
    "hsi": (1, 64, 64, 32),
    "rgb": (1, 128, 128, 3),
    "masks": (1, 128, 128),
}


def _str_dtype():
    return h5py.string_dtype(encoding="utf-8")


def initialize_h5(
    h5_path: Path,
    ingredient_map: Dict[str, int],
    wavelengths: np.ndarray,
    shapes: H5Shapes | None = None,
    compression: str = "gzip",
    compression_opts: int = 4,
    shuffle: bool = True,
    overwrite: bool = False,
) -> Path:
    """Create a new HDF5 file with the required structure.

    If the file exists and overwrite=False, validates structure and returns.
    """
    shapes = shapes or H5Shapes()
    h5_path = Path(h5_path)
    if h5_path.exists() and not overwrite:
        # Validate minimal structure
        with h5py.File(h5_path, "a") as f:
            for key in ["hsi", "rgb", "masks", "metadata"]:
                if key not in f:
                    raise ValueError(f"Existing H5 missing dataset/group: {key}")
        return h5_path

    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as f:
        # Datasets with 0 along sample dimension
        f.create_dataset(
            "hsi",
            shape=(0, *shapes.hsi),
            maxshape=(None, *shapes.hsi),
            dtype=np.float32,
            chunks=DEFAULT_CHUNKS["hsi"],
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
        )
        f["hsi"].attrs["description"] = "Hyperspectral data cubes (height, width, bands)."

        f.create_dataset(
            "rgb",
            shape=(0, *shapes.rgb),
            maxshape=(None, *shapes.rgb),
            dtype=np.uint8,
            chunks=DEFAULT_CHUNKS["rgb"],
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
        )
        f["rgb"].attrs["description"] = "RGB images corresponding to HSI data."

        f.create_dataset(
            "masks",
            shape=(0, *shapes.mask),
            maxshape=(None, *shapes.mask),
            dtype=np.uint8,
            chunks=DEFAULT_CHUNKS["masks"],
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
        )
        f["masks"].attrs["description"] = (
            "Integer-based segmentation masks. See /metadata/ingredient_map for class labels."
        )

        meta = f.create_group("metadata")
        meta.create_dataset("image_basenames", shape=(0,), maxshape=(None,), dtype=_str_dtype())
        meta.create_dataset("dish_labels", shape=(0,), maxshape=(None,), dtype=_str_dtype())

        # Store ingredient map as JSON string in a scalar vlen string dataset sized (1,)
        meta.create_dataset(
            "ingredient_map",
            shape=(1,),
            maxshape=(1,),
            dtype=_str_dtype(),
        )
        meta["ingredient_map"][0] = json.dumps(ingredient_map, ensure_ascii=False)

        wavelengths = np.asarray(wavelengths, dtype=np.float32)
        if wavelengths.ndim != 1 or wavelengths.shape[0] != shapes.bands:
            raise ValueError(
                f"Wavelengths shape invalid: {wavelengths.shape}, expected ({shapes.bands},)"
            )
        meta.create_dataset("wavelengths", data=wavelengths, dtype=np.float32)

    return h5_path


def append_sample(
    f: h5py.File,
    hsi: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    basename: str,
    dish_label: str,
) -> int:
    """Append a single sample. Returns index written."""
    # Validate shapes/dtypes
    if hsi.dtype != np.float32:
        hsi = hsi.astype(np.float32, copy=False)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8, copy=False)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8, copy=False)

    hsi_ds = f["hsi"]
    rgb_ds = f["rgb"]
    mask_ds = f["masks"]

    expected_hsi = hsi_ds.shape[1:]
    expected_rgb = rgb_ds.shape[1:]
    expected_mask = mask_ds.shape[1:]

    if tuple(hsi.shape) != tuple(expected_hsi):
        raise ValueError(f"HSI shape {hsi.shape} != expected {expected_hsi}")
    if tuple(rgb.shape) != tuple(expected_rgb):
        raise ValueError(f"RGB shape {rgb.shape} != expected {expected_rgb}")
    if tuple(mask.shape) != tuple(expected_mask):
        raise ValueError(f"Mask shape {mask.shape} != expected {expected_mask}")

    n = hsi_ds.shape[0]
    new_n = n + 1

    hsi_ds.resize((new_n, *expected_hsi))
    rgb_ds.resize((new_n, *expected_rgb))
    mask_ds.resize((new_n, *expected_mask))

    hsi_ds[n] = hsi
    rgb_ds[n] = rgb
    mask_ds[n] = mask

    # Strings
    meta = f["metadata"]
    for name, value in ("image_basenames", str(basename)), ("dish_labels", str(dish_label)):
        ds = meta[name]
        ds.resize((new_n,))
        ds[n] = value

    return n


def contains_basename(f: h5py.File, basename: str) -> bool:
    ds = f["metadata/image_basenames"]
    if ds.shape[0] == 0:
        return False
    try:
        values = ds.asstr()[...].tolist()
    except Exception:
        values = [v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v) for v in ds[...].tolist()]
    return str(basename) in set(values)


def open_h5(h5_path: Path, mode: str = "a") -> h5py.File:
    return h5py.File(h5_path, mode)
