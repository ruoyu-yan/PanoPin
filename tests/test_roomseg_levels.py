import numpy as np
import pytest

from panopin.roomseg import DEFAULTS, SegmentationError
from panopin.roomseg.levels import storey_heights
from tests.roomseg_synth import H, two_rooms


def test_floor_and_ceiling_of_a_single_storey():
    floor, ceiling, peaks = storey_heights(two_rooms()[:, 2])
    assert abs(floor - 0.0) <= DEFAULTS.z_bin
    assert abs(ceiling - H) <= DEFAULTS.z_bin
    assert len(peaks) >= 2


def test_ceiling_is_the_lowest_peak_of_the_top_cluster():
    """Area_2 ceilings sit at 2.58-2.81 m: the band must stay below the LOWEST of them."""
    xyz = two_rooms()
    lower = xyz[xyz[:, 0] < 4.0].copy()
    lower = lower[np.abs(lower[:, 2] - H) < 1e-9]
    lower[:, 2] = H - 0.2                      # a second, lower ceiling level 0.2 m down
    _, ceiling, _ = storey_heights(np.concatenate([xyz[:, 2], lower[:, 2]]))
    assert abs(ceiling - (H - 0.2)) <= DEFAULTS.z_bin


def test_z_outliers_do_not_move_the_heights():
    z = two_rooms()[:, 2]
    noisy = np.concatenate([z, np.full(20, -8.0), np.full(20, 12.0)])
    # bin edges start at z.min(), so the outliers may shift a peak by one bin -- no more
    assert storey_heights(noisy)[:2] == pytest.approx(storey_heights(z)[:2], abs=DEFAULTS.z_bin)


def test_millimetre_cloud_is_refused_and_says_metres():
    with pytest.raises(SegmentationError, match="metres"):
        storey_heights(two_rooms()[:, 2] * 1000.0)


def test_two_storeys_are_refused():
    xyz = two_rooms()
    upper = xyz + np.array([0.0, 0.0, H + 0.2])
    with pytest.raises(SegmentationError, match="storey"):
        storey_heights(np.concatenate([xyz, upper])[:, 2])


def test_missing_ceiling_is_refused():
    with pytest.raises(SegmentationError, match="floor and ceiling"):
        storey_heights(two_rooms(ceiling=False)[:, 2])


def test_empty_cloud_is_refused():
    with pytest.raises(SegmentationError, match="empty"):
        storey_heights(np.zeros(0))
