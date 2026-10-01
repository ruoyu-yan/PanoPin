import numpy as np
import pytest

from panopin.roomseg import DEFAULTS, SegmentationError, segment
from panopin.roomseg.lift import lift
from panopin.roomseg.raster import Grid
from tests.roomseg_synth import H, box, two_rooms


def _floor(xyz):
    return np.abs(xyz[:, 2]) < 1e-9


def test_two_rooms_with_their_own_wall_faces():
    xyz = two_rooms()
    labels, image, rep = segment(xyz)
    assert rep["K"] == 2 and image.max() == 2
    left = _floor(xyz) & (xyz[:, 0] < 3.8)
    right = _floor(xyz) & (xyz[:, 0] > 4.2)
    assert len(set(labels[left])) == 1 and len(set(labels[right])) == 1
    assert labels[left][0] != labels[right][0]
    lface = np.abs(xyz[:, 0] - 3.9) < 1e-9
    rface = np.abs(xyz[:, 0] - 4.1) < 1e-9
    assert set(labels[lface]) == {labels[left][0]}
    assert set(labels[rface]) == {labels[right][0]}


def test_rooms_are_named_by_size():
    xyz = two_rooms(split=3.0)                      # right room is the bigger one
    labels, _, rep = segment(xyz)
    big = _floor(xyz) & (xyz[:, 0] > 3.2)
    assert set(labels[big]) == {0}
    assert [s["name"] for s in rep["segments"]] == ["seg_00", "seg_01"]
    assert rep["segments"][0]["n_points"] > rep["segments"][1]["n_points"]


def test_furniture_points_join_the_room_they_stand_in():
    xyz = np.concatenate([two_rooms(), box(5.0, 7.0, 3.0, 3.6, 2.0)])
    labels, _, _ = segment(xyz)
    cab = (xyz[:, 0] >= 5.0) & (xyz[:, 0] <= 7.0) & (xyz[:, 1] >= 3.0) & (xyz[:, 2] > 0.1) & (xyz[:, 2] < 2.05)
    right = _floor(xyz) & (xyz[:, 0] > 4.2)
    assert set(labels[cab]) == {labels[right][0]}


def test_a_point_far_outside_is_unassigned():
    xyz = np.concatenate([two_rooms(), [[20.0, 2.0, 1.0]]])
    labels, _, rep = segment(xyz)
    assert labels[-1] == -1 and rep["n_unassigned"] == 1
    assert rep["warnings"] == []


def test_more_segments_than_panos_warns():
    _, _, rep = segment(two_rooms(), n_panos=1)
    assert any("K=2 > N=1" in w for w in rep["warnings"])


def test_report_records_heights_band_and_params():
    _, _, rep = segment(two_rooms())
    assert abs(rep["ceiling_z"] - H) <= DEFAULTS.z_bin
    assert rep["band"][1] - rep["band"][0] == pytest.approx(0.5)
    assert rep["params"]["cell"] == 0.05


def test_no_room_raises():
    xyz = two_rooms(door_top=None, gap=(0.0, 4.0))     # wall removed entirely -> still 1 room
    assert segment(xyz)[2]["K"] == 1
    tiny = two_rooms() * np.array([0.1, 0.1, 1.0])     # 0.8 x 0.4 m: below min_room_area
    with pytest.raises(SegmentationError, match="no room"):
        segment(tiny)


def test_segment_is_deterministic():
    a = segment(two_rooms())
    b = segment(two_rooms())
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_lift_maps_cells_to_zero_based_labels():
    g = Grid(0.0, 0.0, 1.0, 2, 2)
    filled = np.array([[1, 0], [2, 2]], np.int32)
    out = lift(filled, g, np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5]]))
    assert out.tolist() == [0, -1, 1]
