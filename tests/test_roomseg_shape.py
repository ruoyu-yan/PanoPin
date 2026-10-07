import numpy as np
import pytest
from panopin.roomseg.shape import segment_shape, is_corridor, CORRIDOR_RATIO


def _box(w, d, n=20000, cx=0.0, cy=0.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform([-w / 2, -d / 2], [w / 2, d / 2], size=(n, 2)) + [cx, cy]
    return np.column_stack([xy, rng.uniform(0.0, 2.8, size=n)])


def test_corridor_box_is_elongated_and_centred():
    s = segment_shape(_box(10.0, 2.0, cx=3.0, cy=20.0))
    assert abs(s.extent_ratio - 5.0) < 0.3
    assert np.allclose(s.centroid_xy, (3.0, 20.0), atol=0.05)
    assert s.n_points == 20000
    assert is_corridor(s)


def test_office_box_is_not_a_corridor():
    s = segment_shape(_box(4.0, 4.0))
    assert abs(s.extent_ratio - 1.0) < 0.1
    assert not is_corridor(s)


def test_ratio_threshold_is_the_measured_one():
    assert CORRIDOR_RATIO == 2.5          # corridors 3.6 and 5.4, offices <= 1.5 (Area_2_manhattan7)


def test_collinear_points_are_an_infinite_corridor():
    xyz = np.column_stack([np.linspace(0, 10, 100), np.zeros(100), np.ones(100)])
    s = segment_shape(xyz)
    assert s.extent_ratio == float("inf") and is_corridor(s)


def test_fewer_than_three_points_is_refused():
    with pytest.raises(ValueError):
        segment_shape(np.zeros((2, 3)))
