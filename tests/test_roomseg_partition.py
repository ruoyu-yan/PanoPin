import numpy as np

from panopin.roomseg import DEFAULTS
from panopin.roomseg.partition import fill, rooms, wall_mask
from panopin.roomseg.raster import Grid, band_image, footprint
from tests.roomseg_synth import box, shell, thick_partition_x, two_rooms

BAND = (2.2, 2.7)   # ceiling 2.8 - 0.6 .. - 0.1


def _rooms_of(xyz, p=DEFAULTS):
    g = Grid.fit(xyz[:, :2], p.cell, p.max_grid_cells)
    fp = footprint(g, xyz, p.footprint_close)
    walls = wall_mask(band_image(g, xyz, *BAND), g.cell, p)
    return g, fp, rooms(walls, fp, g.cell, p)


def test_door_under_a_lintel_separates_two_rooms():
    _, _, lab = _rooms_of(two_rooms(door_top=2.1))
    assert lab.max() == 2


def test_opening_up_to_the_ceiling_joins_them():
    """Documents the open-plan limit: no lintel, no wall in the band."""
    _, _, lab = _rooms_of(two_rooms(door_top=None, gap=(1.0, 3.0)))
    assert lab.max() == 1


def test_a_narrow_scan_hole_is_sealed():
    xyz = np.concatenate([shell(8.0, 4.0), thick_partition_x(4.0, 4.0, openings=[(2.0, 2.2, None)])])
    _, _, lab = _rooms_of(xyz)
    assert lab.max() == 2


def test_a_door_across_a_corridor_splits_it():
    xyz = np.concatenate([shell(10.0, 2.0), thick_partition_x(5.0, 2.0, openings=[(0.2, 1.8, 2.1)])])
    _, _, lab = _rooms_of(xyz)
    assert lab.max() == 2


def test_furniture_below_the_band_changes_nothing():
    plain = _rooms_of(two_rooms())[2]
    furnished = _rooms_of(np.concatenate([two_rooms(), box(0.5, 1.5, 0.5, 1.5, 0.8),
                                          box(5.0, 7.0, 3.0, 3.6, 2.0)]))[2]
    assert np.array_equal(plain, furnished)


def test_thin_slivers_are_dropped():
    walls = np.zeros((60, 60), bool)
    walls[:, 30] = True; walls[:, 39] = True           # 8-cell (0.40 m) channel between two walls
    grown = wall_mask(walls.astype(np.int32) * 10, 0.05, DEFAULTS)   # 2 cells (0.10 m) remain free
    lab = rooms(grown, np.ones_like(walls), 0.05, DEFAULTS)
    assert lab.max() == 2                               # left and right rooms only
    assert lab[:, 31:39].max() == 0                     # the sliver is dropped


def test_fill_gives_wall_cells_their_nearest_room_and_caps_distance():
    g, fp, lab = _rooms_of(two_rooms())
    filled = fill(lab, fp, g.cell, DEFAULTS)
    r, c = g.cells(np.array([[3.9, 3.0], [4.1, 3.0], [2.0, 2.0], [6.0, 2.0]]))
    left_face, right_face, left, right = filled[r, c]
    assert left_face == left and right_face == right and left != right
    far = np.zeros_like(lab); far[5, 5] = 1
    capped = fill(far, np.ones_like(fp), g.cell, DEFAULTS)
    assert capped[5, 5 + 19] == 1 and capped[5, 5 + 21] == 0   # 0.95 m in, 1.05 m out
