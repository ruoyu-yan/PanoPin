"""segment(): one merged cloud -> per-point room labels + label image + report (spec §3-§5)."""
from dataclasses import asdict

import numpy as np

from .errors import SegmentationError
from .levels import storey_heights
from .lift import lift
from .params import DEFAULTS
from .partition import fill, rooms, wall_mask
from .raster import Grid, band_image, footprint


def segment(xyz, p=DEFAULTS, n_panos=None):
    """xyz (N,3) in metres, z-up, one storey -> (labels int32[N] in 0..K-1 or -1,
    image int32[ny,nx] in 0..K, report dict)."""
    xyz = np.asarray(xyz, dtype=float)
    floor_z, ceiling_z, peaks = storey_heights(xyz[:, 2], p)
    if p.band_from_floor is not None:
        z0 = floor_z + p.band_from_floor
    else:
        z0 = ceiling_z - p.band_bottom
    z1 = ceiling_z - p.band_top
    grid = Grid.fit(xyz[:, :2], p.cell, p.max_grid_cells)
    fp = footprint(grid, xyz, p.footprint_close)
    walls = wall_mask(band_image(grid, xyz, z0, z1), grid.cell, p)
    kept = rooms(walls, fp, grid.cell, p)
    if kept.max() == 0:
        raise SegmentationError(
            f"no room survived cleanup (band {z0:.2f}-{z1:.2f} m); check the band heights")
    filled = fill(kept, fp, grid.cell, p)
    labels, image, segments = _canonical_order(lift(filled, grid, xyz[:, :2]), filled, xyz, grid)
    n_un = int((labels < 0).sum())
    report = {
        "floor_z": floor_z, "ceiling_z": ceiling_z, "z_peaks": peaks, "band": [z0, z1],
        "params": asdict(p),
        "grid": {"x0": grid.x0, "y0": grid.y0, "cell": grid.cell, "ny": grid.ny, "nx": grid.nx},
        "segments": segments, "K": len(segments), "N": n_panos,
        "n_points": int(len(labels)), "n_unassigned": n_un,
        "unassigned_frac": n_un / max(len(labels), 1),
    }
    report["warnings"] = _warnings(report, p)
    return labels, image, report


def _canonical_order(raw, filled, xyz, grid):
    """Rename rooms by point count (desc), ties by centroid x then y; drop rooms with no points."""
    info = []
    for k in range(int(filled.max())):
        m = raw == k
        n = int(m.sum())
        if n == 0:
            continue
        c = xyz[m, :2].mean(axis=0)
        area = float((filled == k + 1).sum()) * grid.cell ** 2
        info.append((-n, float(c[0]), float(c[1]), k, area))
    info.sort()
    new_of_old = np.full(int(filled.max()) + 1, -1, np.int32)   # index = old label + 1
    segments = []
    for new, (neg_n, cx, cy, old, area) in enumerate(info):
        new_of_old[old + 1] = new
        segments.append({"name": f"seg_{new:02d}", "n_points": -neg_n, "area_m2": round(area, 3),
                         "centroid_xy": [round(cx, 3), round(cy, 3)]})
    labels = new_of_old[raw + 1]
    image = (new_of_old[filled] + 1).astype(np.int32)
    return labels, image, segments


def _warnings(r, p):
    out = []
    if r["N"] is not None and r["K"] > r["N"]:
        out.append(f"K={r['K']} > N={r['N']}: more segments than panoramas "
                   "(over-segmentation, or a space without a panorama)")
    if r["unassigned_frac"] > p.warn_unassigned_frac:
        out.append(f"{r['unassigned_frac']:.1%} of points unassigned "
                   f"(> {p.warn_unassigned_frac:.0%})")
    for s in r["segments"]:
        if s["area_m2"] < p.warn_small_room_area:
            out.append(f"{s['name']} is only {s['area_m2']} m2 (< {p.warn_small_room_area} m2)")
    return out
