from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb(png_path: Path, expected_shape: tuple[int, int, int] = (512, 512, 3)) -> np.ndarray:
    """Read PNG RGB to np.uint8 array (H,W,3)."""
    with Image.open(png_path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
    if expected_shape is not None and tuple(arr.shape) != expected_shape:
        raise ValueError(f"RGB image shape {arr.shape} != expected {expected_shape}")
    return arr
