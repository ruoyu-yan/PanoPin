"""Plan shape of one room segment, for the corridor seed rule (spec 2026-10-07 §4).

A corridor looks the same to the colour optimizer along its whole length, so PanoPin's seed
slides along it (4.65 m and 3.08 m from the station on Area_2_manhattan7's corridors,
2026-10-01), and FGPL's search cell follows the seed. The segment's centroid is within 0.9 m of
where those panoramas stood. Reads only the segment's points, never GT."""
from collections import namedtuple

import numpy as np

SegmentShape = namedtuple("SegmentShape", "centroid_xy extent_ratio n_points")

# Measured on Area_2_manhattan7 (2026-10-06): corridors 3.6 and 5.4, every office <= 1.5.
CORRIDOR_RATIO = 2.5


def segment_shape(xyz):
    """xyz (N, 3) -> SegmentShape: plan centroid, and the ratio (>= 1) of the segment's two
    principal plan extents (singular values of the centred xy; inf when the points are
    collinear). Fewer than 3 points is refused: there is no plane to measure."""
    xy = np.asarray(xyz, dtype=np.float64)[:, :2]
    if len(xy) < 3:
        raise ValueError(f"segment_shape needs at least 3 points, got {len(xy)}")
    c = xy.mean(axis=0)
    s = np.linalg.svd(xy - c, compute_uv=False)
    ratio = float(s[0] / s[1]) if s[1] > 1e-12 else float("inf")
    return SegmentShape((float(c[0]), float(c[1])), ratio, int(len(xy)))


def is_corridor(shape, ratio=CORRIDOR_RATIO):
    return shape.extent_ratio >= ratio
