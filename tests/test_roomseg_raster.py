import numpy as np
import pytest

from panopin.roomseg import DEFAULTS, SegmentationError
from panopin.roomseg.raster import Grid, band_image, close, count_image, footprint
from tests.roomseg_synth import two_rooms


def test_cells_and_world_round_trip():
    xyz = two_rooms()
    g = Grid.fit(xyz[:, :2], 0.05, DEFAULTS.max_grid_cells)
    row, col = g.cells(xyz[:, :2])
    back = g.world(row, col)
    assert np.abs(back - xyz[:, :2]).max() <= 0.05 * np.sqrt(2) / 2 + 1e-9
    assert row.min() >= 1 and col.min() >= 1          # PAD cells on every side (float-safe)


def test_count_image_counts_every_point():
    xyz = two_rooms()
    g = Grid.fit(xyz[:, :2], 0.05, DEFAULTS.max_grid_cells)
    assert count_image(g, xyz[:, :2]).sum() == len(xyz)


def test_band_image_keeps_only_the_band():
    xyz = two_rooms()
    g = Grid.fit(xyz[:, :2], 0.05, DEFAULTS.max_grid_cells)
    inside = (xyz[:, 2] >= 2.2) & (xyz[:, 2] <= 2.7)
    assert band_image(g, xyz, 2.2, 2.7).sum() == inside.sum()


def test_far_outlier_is_refused_with_the_extent_not_a_memory_error():
    xyz = np.concatenate([two_rooms(), [[500.0, 500.0, 1.0]]])
    with pytest.raises(SegmentationError, match="extent"):
        Grid.fit(xyz[:, :2], 0.05, DEFAULTS.max_grid_cells)


def test_close_seals_narrow_gaps_between_blobs_only():
    """close() is for the footprint (thick blobs). It cannot bridge a gap in a 1-cell line,
    which is why walls are grown instead (Task 3)."""
    m = np.zeros((40, 60), bool)
    m[10:30, 5:20] = True; m[10:30, 24:40] = True    # 4-cell gap
    m[10:30, 49:60] = True                            # 9-cell gap; touches the border
    c = close(m, 3)
    assert c[20, 20:24].all()                         # 4-cell gap sealed (<= 2r)
    assert not c[20, 41:48].any()                     # 9-cell gap kept
    assert c[10:30, 49:60].all()                      # border blob intact


def test_footprint_covers_the_building_and_not_outside():
    xyz = two_rooms()
    g = Grid.fit(xyz[:, :2], 0.05, DEFAULTS.max_grid_cells)
    fp = footprint(g, xyz, 0.30)
    r, c = g.cells(np.array([[2.0, 2.0], [6.0, 2.0]]))
    assert fp[r, c].all()
    assert not fp[0, 0]
