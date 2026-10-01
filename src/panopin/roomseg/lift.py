"""Room label image -> per-point labels (spec §4 step 8)."""
import numpy as np


def lift(filled, grid, xy):
    """Per-point room index 0..K-1, or -1 where the point's cell holds no room."""
    row, col = grid.cells(xy)
    return filled[row, col].astype(np.int32) - 1
