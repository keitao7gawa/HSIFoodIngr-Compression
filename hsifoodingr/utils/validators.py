from __future__ import annotations

import numpy as np


def assert_shape(array: np.ndarray, expected_shape: tuple[int, ...], name: str) -> None:
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} != expected {expected_shape}")


def assert_dtype(array: np.ndarray, expected_dtype, name: str) -> None:
    if array.dtype != expected_dtype:
        raise ValueError(f"{name} dtype {array.dtype} != expected {expected_dtype}")
