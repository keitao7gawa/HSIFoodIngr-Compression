from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import spectral as sp
except Exception as e:  # pragma: no cover
    sp = None  # type: ignore


def read_envi_hsi(hdr_path: Path, dat_path: Path, expected_shape: Tuple[int, int, int] = (512, 512, 204)) -> np.ndarray:
    """Read ENVI HDR/DAT (BIL, float) to np.float32 array shaped (H, W, Bands).

    - Does not rescale data; preserves original float range.
    - Validates shape if `expected_shape` is provided.
    """
    if sp is None:
        raise RuntimeError("spectral is not installed. Please install 'spectral' package.")

    hdr_path = Path(hdr_path)
    dat_path = Path(dat_path)

    img = sp.envi.open(str(hdr_path), str(dat_path))
    cube = np.asarray(img.load(), dtype=np.float32)  # lazy→array

    # Some ENVI readers return (Bands, Rows, Cols). Ensure (H,W,B)
    if cube.ndim == 3 and cube.shape[0] == expected_shape[2]:
        # assume (B, H, W) → (H, W, B)
        cube = np.moveaxis(cube, 0, -1)

    if expected_shape is not None and tuple(cube.shape) != expected_shape:
        raise ValueError(f"ENVI cube shape {cube.shape} != expected {expected_shape}")

    return cube
