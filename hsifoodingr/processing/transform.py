from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..utils import get_logger
from ..io.hdf5_writer import open_h5

logger = get_logger(__name__)


def rotate_hsi_inplace(
    h5_path: Path,
    k: int = -1,
    batch_size: int = 16,
    index_range: Optional[Tuple[int, int]] = None,
) -> Tuple[int, Tuple[int, int, int]]:
    """Rotate HSI by 90° steps in-place for all or a range of samples.

    Parameters
    - h5_path: Target HDF5
    - k: Number of 90° rotations. -1 means 90° clockwise, 1 means 90° counter-clockwise
    - batch_size: How many samples to process per batch
    - index_range: (start, end) inclusive-exclusive range; None for all

    Returns
    - processed_count, sample_shape_after
    """
    h5_path = Path(h5_path)
    processed = 0
    with open_h5(h5_path, mode="a") as f:
        hsi = f["hsi"]
        n = int(hsi.shape[0])
        h, w, c = int(hsi.shape[1]), int(hsi.shape[2]), int(hsi.shape[3])
        start = 0
        end = n
        if index_range is not None:
            start = max(0, int(index_range[0]))
            end = min(n, int(index_range[1]))
        for i in range(start, end, batch_size):
            j = min(i + batch_size, end)
            batch = hsi[i:j, ...]  # (B, H, W, C)
            rotated = np.rot90(batch, k=k, axes=(1, 2))
            hsi[i:j, ...] = rotated
            processed += (j - i)
    logger.info("HSI rotation completed: processed=%d shape=%s k=%d", processed, (h, w, c), k)
    return processed, (h, w, c)
