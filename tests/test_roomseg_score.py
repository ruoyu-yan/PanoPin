import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import roomseg_gt      # noqa: E402
import roomseg_score   # noqa: E402

ROOMS = ["a", "b"]


def _two_blocks():
    """Room a: x in [0, 2); room b: x in [2, 4); 5 cm voxels, one point per voxel."""
    xs, ys = np.meshgrid(np.arange(0, 4, 0.05) + 0.025, np.arange(0, 1, 0.05) + 0.025)
    xyz = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.025)])
    gt = (xyz[:, 0] >= 2.0).astype(np.int64)
    return xyz, gt


def test_perfect_partition_scores_one():
    xyz, gt = _two_blocks()
    vk, vg, vp = roomseg_score.voxels(xyz, gt, gt.copy(), 0.05)
    m = roomseg_score.metrics(vg, vp, ROOMS, ["seg_00", "seg_01"])
    assert m["f1_07"] == 1.0 and m["pq"] == pytest.approx(1.0)
    assert m["splits"] == [] and m["merges"] == {}
    assert m["matching"]["a"]["segment"] == "seg_00"


def test_merge_and_split_are_named():
    xyz, gt = _two_blocks()
    merged = np.zeros_like(gt)
    m = roomseg_score.metrics(*roomseg_score.voxels(xyz, gt, merged, 0.05)[1:], ROOMS, ["seg_00"])
    assert m["merges"] == {"seg_00": ["a", "b"]}
    split = np.where(xyz[:, 0] < 1.0, 0, np.where(xyz[:, 0] < 2.0, 1, 2))
    m = roomseg_score.metrics(*roomseg_score.voxels(xyz, gt, split, 0.05)[1:], ROOMS,
                              ["seg_00", "seg_01", "seg_02"])
    assert m["splits"] == ["a"]


def test_majority_label_per_voxel():
    xyz = np.array([[0.01, 0.01, 0.01]] * 3 + [[0.07, 0.01, 0.01]])
    vk, vg, vp = roomseg_score.voxels(xyz, np.array([0, 0, 1, 1]), np.array([1, 1, 0, -1]), 0.05)
    assert vg.tolist() == [0, 1] and vp.tolist() == [1, -1]


def test_boundary_band_flags_only_voxels_near_another_room():
    xyz, gt = _two_blocks()
    vk, vg, _ = roomseg_score.voxels(xyz, gt, gt, 0.05)
    near = roomseg_score.near_other_room(vk, vg, 4)
    x = (vk[:, 0] + 0.5) * 0.05
    assert near[np.abs(x - 2.0) < 0.15].all()
    assert not near[np.abs(x - 2.0) > 0.25].any()


def test_containment_uses_points_around_the_camera():
    xyz, gt = _two_blocks()
    pred = gt.copy()
    panos = {"camera_u1_a": [0.5, 0.5, 1.4], "camera_u2_b": [3.5, 0.5, 1.4]}
    matching = {"a": {"segment": "seg_00", "iou": 1.0}, "b": {"segment": "seg_01", "iou": 1.0}}
    rows = roomseg_score.containment(xyz, gt, pred, panos, ROOMS, ["seg_00", "seg_01"], matching)
    assert rows["camera_u1_a"]["gt_inside"] and rows["camera_u1_a"]["pred_inside"]
    rows = roomseg_score.containment(xyz, gt, np.zeros_like(gt), panos, ROOMS, ["seg_00"],
                                     {"a": {"segment": "seg_00", "iou": 0.5}, "b": None})
    assert rows["camera_u2_b"]["pred_inside"] is False


def test_gt_labels_match_room_files_in_any_order(tmp_path):
    xyz, gt = _two_blocks()
    for i, r in enumerate(ROOMS):
        (tmp_path / r).mkdir()
        sel = xyz[gt == i]
        np.savetxt(tmp_path / r / f"{r}.txt", np.column_stack([sel, np.zeros((len(sel), 3))]),
                   fmt="%.6f")
    perm = np.random.RandomState(0).permutation(len(xyz))
    labels = roomseg_gt.gt_labels_for(xyz[perm], tmp_path, ROOMS)
    assert np.array_equal(labels, gt[perm])
    with pytest.raises(ValueError, match="points"):
        roomseg_gt.gt_labels_for(xyz[:-1], tmp_path, ROOMS)
