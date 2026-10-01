"""Metric 2D grid over the cloud's XY extent, and the images built on it (spec §3, §4)."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .errors import SegmentationError

PAD = 2   # empty cells around the extent


@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    cell: float
    ny: int
    nx: int

    @classmethod
    def fit(cls, xy, cell, max_cells):
        lo = xy.min(axis=0) - PAD * cell
        hi = xy.max(axis=0) + PAD * cell
        nx = int(np.floor((hi[0] - lo[0]) / cell)) + 1
        ny = int(np.floor((hi[1] - lo[1]) / cell)) + 1
        if nx * ny > max_cells:
            raise SegmentationError(
                f"XY extent {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} m needs {nx * ny} cells at "
                f"{cell} m (> {max_cells}); remove far outlier points first")
        return cls(float(lo[0]), float(lo[1]), float(cell), ny, nx)

    def cells(self, xy):
        """(row, col) of the cells holding each xy, clipped to the grid."""
        col = np.clip(np.floor((xy[:, 0] - self.x0) / self.cell).astype(np.int64), 0, self.nx - 1)
        row = np.clip(np.floor((xy[:, 1] - self.y0) / self.cell).astype(np.int64), 0, self.ny - 1)
        return row, col

    def world(self, row, col):
        """XY of the cell centres."""
        return np.stack([self.x0 + (np.asarray(col) + 0.5) * self.cell,
                         self.y0 + (np.asarray(row) + 0.5) * self.cell], axis=1)


def count_image(grid, xy):
    row, col = grid.cells(xy)
    flat = np.bincount(row * grid.nx + col, minlength=grid.ny * grid.nx)
    return flat.reshape(grid.ny, grid.nx).astype(np.int32)


def band_image(grid, xyz, z0, z1):
    """Points per cell with z0 <= z <= z1."""
    keep = (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1)
    return count_image(grid, xyz[keep, :2])


def disk(r_cells):
    r = int(r_cells)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def close(mask, r_cells):
    """Binary closing that is exact at the grid border (pad, close, crop)."""
    r = int(r_cells)
    if r < 1:
        return mask.copy()
    k = disk(r)
    big = np.pad(mask, r + 1)
    big = ndimage.binary_erosion(ndimage.binary_dilation(big, k), k, border_value=1)
    return big[r + 1:-(r + 1), r + 1:-(r + 1)]


def footprint(grid, xyz, close_m):
    """Cells inside the building: anything scanned at any height, holes filled, closed."""
    occ = ndimage.binary_fill_holes(count_image(grid, xyz[:, :2]) > 0)
    occ = close(occ, round(close_m / grid.cell))
    return ndimage.binary_fill_holes(occ)
