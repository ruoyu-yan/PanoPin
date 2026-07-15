import math
import numpy as np
import pytest
from panopin.fgpl_export import raw_t_to_camera_position


def _yaw(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def test_camera_position_roundtrips_for_yaw():
    R = _yaw(37.0)
    t = [2.5, -1.3, 1.4]
    cam = raw_t_to_camera_position(t, R)
    back = (np.array(R).T @ np.array([cam[0], cam[1], 0.0]))[:2]  # FGPL's inverse
    assert np.allclose(back, [2.5, -1.3], atol=1e-6)


def test_identity_metadata_is_passthrough():
    cam = raw_t_to_camera_position([1.0, 2.0, 3.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(cam, [1.0, 2.0])


def test_non_yaw_rotation_fails_loud():
    R = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # 90 deg about x: mixes z into y
    with pytest.raises(ValueError):
        raw_t_to_camera_position([1.0, 2.0, 3.0], R)
