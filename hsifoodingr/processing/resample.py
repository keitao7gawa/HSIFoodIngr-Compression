from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import h5py

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ResampleConfig:
    input_h5: Path
    output_h5: Path
    new_wavelengths: np.ndarray
    chunk_size: int = 8
    overwrite: bool = False


def _create_output_file(out_path: Path, in_f: h5py.File, new_wavelengths: np.ndarray, overwrite: bool) -> h5py.File:
    out_path = Path(out_path)
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output H5 already exists: {out_path}")
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Input shapes
    hsi_in = in_f["hsi"]
    n, h, w, _ = hsi_in.shape

    f = h5py.File(out_path, "w")

    # Create HSI dataset with new band count
    new_b = int(new_wavelengths.shape[0])
    # Choose chunks similar to the existing scheme
    chunk_h = min(64, h)
    chunk_w = min(64, w)
    chunk_b = min(32, new_b)
    hsi_out = f.create_dataset(
        "hsi",
        shape=(n, h, w, new_b),
        dtype=np.float32,
        chunks=(1, chunk_h, chunk_w, chunk_b),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    hsi_out.attrs["description"] = "Hyperspectral data cubes (height, width, bands)."

    # RGB copy target
    rgb_in = in_f["rgb"]
    rgb_out = f.create_dataset(
        "rgb",
        shape=rgb_in.shape,
        dtype=rgb_in.dtype,
        chunks=rgb_in.chunks or (1, min(128, rgb_in.shape[1]), min(128, rgb_in.shape[2]), 3),
        compression=rgb_in.compression or "gzip",
        compression_opts=rgb_in.compression_opts if rgb_in.compression_opts is not None else 4,
        shuffle=True,
    )
    rgb_out.attrs["description"] = "RGB images corresponding to HSI data."

    # Masks copy target
    masks_in = in_f["masks"]
    masks_out = f.create_dataset(
        "masks",
        shape=masks_in.shape,
        dtype=masks_in.dtype,
        chunks=masks_in.chunks or (1, min(128, masks_in.shape[1]), min(128, masks_in.shape[2])),
        compression=masks_in.compression or "gzip",
        compression_opts=masks_in.compression_opts if masks_in.compression_opts is not None else 4,
        shuffle=True,
    )
    masks_out.attrs["description"] = masks_in.attrs.get(
        "description",
        "Integer-based segmentation masks. See /metadata/ingredient_map for class labels.",
    )

    # Metadata
    meta_out = f.create_group("metadata")
    # Copy string datasets in full after writing HSI/RGB/Mask
    for name in ("image_basenames", "dish_labels"):
        ds_in = in_f["metadata/"] [name]
        ds = meta_out.create_dataset(name, shape=ds_in.shape, dtype=ds_in.dtype)
        # fill later
    # ingredient_map copied
    meta_out.create_dataset("ingredient_map", data=in_f["metadata/ingredient_map"][...], dtype=in_f["metadata/ingredient_map"].dtype)
    # wavelengths replaced
    meta_out.create_dataset("wavelengths", data=np.asarray(new_wavelengths, dtype=np.float32), dtype=np.float32)

    return f


def _copy_dataset(src: h5py.Dataset, dst: h5py.Dataset, batch: int = 16) -> None:
    n = int(src.shape[0])
    for i in range(0, n, batch):
        j = min(i + batch, n)
        dst[i:j, ...] = src[i:j, ...]


def _interpolate_batch(batch_hsi: np.ndarray, wl_old: np.ndarray, wl_new: np.ndarray) -> np.ndarray:
    """Interpolate a batch of HSI (B,H,W,Cold) to (B,H,W,Cnew) using linear interpolation with edge clamp.

    NaN/Inf are set to 0 before interpolation.
    """
    b, h, w, c_old = batch_hsi.shape
    c_new = int(wl_new.shape[0])

    # Flatten pixels for vector ops
    x = batch_hsi.reshape(b * h * w, c_old).astype(np.float32, copy=False)
    # sanitize
    np.nan_to_num(x, copy=False)

    # Precompute indices for new wavelengths
    idx = np.searchsorted(wl_old, wl_new, side="left")  # shape (c_new,)
    # Prepare output
    y = np.empty((x.shape[0], c_new), dtype=np.float32)

    # Edges: left clamp and right clamp
    left_mask = idx <= 0
    right_mask = idx >= c_old
    middle_mask = ~(left_mask | right_mask)

    if np.any(left_mask):
        y[:, left_mask] = x[:, 0][:, None]
    if np.any(right_mask):
        y[:, right_mask] = x[:, -1][:, None]

    # Middle columns: linear interpolation
    cols = np.nonzero(middle_mask)[0]
    if cols.size:
        i0 = idx[cols] - 1  # shape (K,)
        i1 = idx[cols]      # shape (K,)
        x0 = wl_old[i0]
        x1 = wl_old[i1]
        alpha = (wl_new[cols] - x0) / (x1 - x0 + 1e-12)
        # For each target column, interpolate across all pixels using two source columns
        # Loop over K new-band columns (<=121), vectorized over pixels (Npix)
        for k_col, c0, c1, a in zip(cols, i0, i1, alpha):
            y[:, k_col] = x[:, c0] + (x[:, c1] - x[:, c0]) * a

    return y.reshape(b, h, w, c_new)


def resample_h5(config: ResampleConfig) -> Tuple[int, Tuple[int, int, int, int]]:
    """Resample input H5 HSI to new wavelengths and write a new H5.

    Returns (num_samples, output_hsi_shape)
    """
    input_h5 = Path(config.input_h5)
    output_h5 = Path(config.output_h5)

    with h5py.File(input_h5, "r") as fin:
        wl_old = np.asarray(fin["metadata/wavelengths"], dtype=np.float32)
        wl_new = np.asarray(config.new_wavelengths, dtype=np.float32)
        if wl_old.ndim != 1 or wl_old.size < 2:
            raise ValueError("Invalid input wavelengths")
        if wl_new.ndim != 1 or wl_new.size < 2:
            raise ValueError("Invalid target wavelengths")

        fout = _create_output_file(output_h5, fin, wl_new, config.overwrite)
        try:
            # Copy RGB/Masks in batches (after HSI), but we need them created upfront
            # Process HSI in batches
            hsi_in = fin["hsi"]
            hsi_out = fout["hsi"]
            n = int(hsi_in.shape[0])
            bsz = int(config.chunk_size)
            for i in range(0, n, bsz):
                j = min(i + bsz, n)
                batch = hsi_in[i:j, ...]
                out = _interpolate_batch(batch, wl_old=wl_old, wl_new=wl_new)
                hsi_out[i:j, ...] = out

            # Copy RGB/Masks
            _copy_dataset(fin["rgb"], fout["rgb"], batch=max(1, config.chunk_size * 2))
            _copy_dataset(fin["masks"], fout["masks"], batch=max(1, config.chunk_size * 2))

            # Copy metadata string datasets
            for name in ("image_basenames", "dish_labels"):
                src = fin["metadata/"][name]
                dst = fout["metadata/"][name]
                # Use asstr to ensure dtype conversion if needed
                try:
                    data = src.asstr()[...]
                except Exception:
                    data = src[...]
                dst[...] = data
        finally:
            fout.close()

    with h5py.File(output_h5, "r") as fchk:
        shape = fchk["hsi"].shape
        n_out = int(shape[0])
        return n_out, shape
