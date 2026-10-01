"""Storey heights from the z histogram (spec §4 step 1, as amended 2026-10-01).

floor   = strongest peak of the LOWEST cluster of strong peaks.
ceiling = LOWEST peak of the HIGHEST cluster. Area_2's ceilings sit at 2.58-2.81 m, and the band
          must stay below every one of them, or a lower ceiling fills the band.
A strong peak between floor + 1.8 m and ceiling - 0.5 m is another storey's slab, so refuse.
Desk tops (~0.75 m, up to 0.38 x the tallest bin on Area_2) and lintels (~2.1 m, ~0.15 x) stay
outside that window or below the strength threshold.
"""
import numpy as np

from .errors import SegmentationError
from .params import DEFAULTS


def _strong_peaks(z, p):
    edges = np.arange(z.min(), z.max() + 2 * p.z_bin, p.z_bin)
    h, edges = np.histogram(z, bins=edges)
    centres = edges[:-1] + p.z_bin / 2
    padded = np.concatenate([[-1], h, [-1]])
    is_max = (padded[1:-1] >= padded[:-2]) & (padded[1:-1] >= padded[2:])
    strong = is_max & (h >= p.peak_rel * h.max())
    return centres[strong], h[strong]


def _clusters(zs, gap):
    groups = []
    for i, z in enumerate(zs):
        if groups and z - zs[groups[-1][-1]] <= gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def storey_heights(z, p=DEFAULTS):
    """(floor_z, ceiling_z, strong_peaks) of a single-storey, z-up cloud in metres."""
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        raise SegmentationError("empty cloud")
    zs, hs = _strong_peaks(z, p)
    peaks = [(round(float(a), 3), int(b)) for a, b in zip(zs, hs)]
    groups = _clusters(zs, p.peak_cluster_gap)
    if len(groups) < 2:
        raise SegmentationError(
            f"no clear floor and ceiling in the z histogram (strong peaks {peaks}); "
            "is the cloud z-up and does it include the ceiling?")
    lo, hi = groups[0], groups[-1]
    floor_z = float(zs[lo][int(np.argmax(hs[lo]))])
    ceiling_z = float(zs[hi].min())
    storey = ceiling_z - floor_z
    if not p.min_storey <= storey <= p.max_storey:
        raise SegmentationError(
            f"storey height {storey:.2f} outside [{p.min_storey}, {p.max_storey}]: "
            "is the cloud in metres, z-up and a single storey?")
    mid = [round(float(v), 2) for v in zs
           if floor_z + p.mid_slab_above_floor < v < ceiling_z - p.mid_slab_below_ceiling]
    if mid:
        raise SegmentationError(
            f"strong horizontal slab at z={mid} between floor {floor_z:.2f} and ceiling "
            f"{ceiling_z:.2f}: more than one storey? Split the cloud by storey first")
    return floor_z, ceiling_z, peaks
