"""Band image + footprint -> room label image (spec §4 steps 2, 4-7, as amended 2026-10-01)."""
import numpy as np
from scipy import ndimage

from .raster import disk

FOUR = ndimage.generate_binary_structure(2, 1)   # 4-connectivity: no leaks through wall corners


def wall_mask(band_counts, cell, p):
    """Cells with >= wall_min_pts band points, GROWN by gap_close_r: seals scan gaps up to ~2r.

    A morphological closing cannot bridge a gap in a 1-cell-thin wall (a disk beside the gap
    misses the wall), so the walls are dilated instead. fill() hands the grown cells back to
    the nearest room, and min_room_halfwidth is measured on the free space that remains.
    """
    walls = band_counts >= p.wall_min_pts
    r = round(p.gap_close_r / cell)
    return ndimage.binary_dilation(walls, disk(r)) if r >= 1 else walls


def rooms(walls, footprint, cell, p):
    """Free-space components that are big and wide enough: 0 = none, 1..K."""
    free = footprint & ~walls
    lab, n = ndimage.label(free, structure=FOUR)
    if n == 0:
        return lab.astype(np.int32)
    idx = np.arange(1, n + 1)
    area = np.asarray(ndimage.sum(free, lab, idx)) * cell * cell
    halfw = np.asarray(ndimage.maximum(ndimage.distance_transform_edt(free), lab, idx)) * cell
    keep = idx[(area >= p.min_room_area) & (halfw >= p.min_room_halfwidth)]
    lut = np.zeros(n + 1, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[lab]


def fill(labels, footprint, cell, p):
    """Every footprint cell joins its nearest room if that room is within max_fill_dist."""
    if labels.max() == 0:
        return labels.astype(np.int32)
    dist, (r, c) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    out = labels[r, c].astype(np.int32)
    out[(dist * cell > p.max_fill_dist) | ~footprint] = 0
    return out
