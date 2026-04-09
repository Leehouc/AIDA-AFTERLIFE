from __future__ import annotations

import numpy as np

from .constants import BOARD_SIZE, flatten_xy, unflatten_index


def transform_coord(y: int, x: int, sym: int, size: int = BOARD_SIZE) -> tuple[int, int]:
    rotations = sym % 4
    for _ in range(rotations):
        y, x = size - 1 - x, y
    if sym >= 4:
        x = size - 1 - x
    return y, x


def apply_symmetry_planes(planes: np.ndarray, sym: int) -> np.ndarray:
    out = np.asarray(planes)
    rotations = sym % 4
    if rotations:
        out = np.rot90(out, k=rotations, axes=(-2, -1))
    if sym >= 4:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def apply_symmetry_index(index: int, sym: int, size: int = BOARD_SIZE) -> int:
    x, y = unflatten_index(index)
    y, x = transform_coord(y, x, sym=sym, size=size)
    return flatten_xy(x, y)
