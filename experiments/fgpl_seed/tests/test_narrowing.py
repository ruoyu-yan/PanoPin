import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, "/home/ruoyu/scan2measure-webframework/src/pose_estimation")
import pose_search  # noqa

# NOTE: inputs use torch tensors (not plain np.array) because
# generate_translation_grid/build_rotation_candidates access `.device` and use
# torch ops (torch.quantile, torch.svd, ...) on their args -- that's how every
# real caller in multiroom_pose_estimation.py invokes them. Plain numpy arrays
# fail immediately on `starts.device` under numpy 1.23.5 (no Array-API
# `.device`, that's numpy>=2.0 only), which is a pre-existing input-type
# requirement unrelated to the seed_trans_radius/seed_yaw_deg params under
# test here.


def test_translation_grid_radius_filters_points():
    starts = torch.tensor([[0, 0, 0], [5, 5, 0], [10, 0, 0]], dtype=torch.float32)
    ends = torch.tensor([[1, 0, 0], [6, 5, 0], [11, 0, 0]], dtype=torch.float32)
    full = pose_search.generate_translation_grid(starts, ends, num_trans=200)
    near = pose_search.generate_translation_grid(starts, ends, num_trans=200,
                                                 center=[0, 0, 0], radius=2.0)
    assert len(near) <= len(full)
    assert np.all(np.linalg.norm(np.asarray(near)[:, :2], axis=1) <= 2.0 + 1e-6)


def test_rotation_yaw_filter_reduces_count():
    p2 = torch.eye(3); p3 = torch.eye(3)
    full, _ = pose_search.build_rotation_candidates(p2, p3)
    sub, _ = pose_search.build_rotation_candidates(p2, p3, seed_yaw_deg=0.0, seed_yaw_tol=20)
    assert len(sub) <= len(full) and len(sub) >= 1
