"""Unit test for the round-trip's per-room coverage metric (pure logic, no GPU/cache)."""
from experiments.fgpl_seed.roundtrip import per_room_coverage


def test_per_room_coverage_counts_true_room_landings():
    # centroids: A at x=0, B at x=10. A pano's pose "lands" in the room of its nearest centroid.
    cents = {"A": [0.0, 0.0, 0.0], "B": [10.0, 0.0, 0.0]}
    admitted_rows = [
        {"pano_name": "a1", "room": "A"},   # lands near A -> covers A
        {"pano_name": "b1", "room": "B"},   # lands near A (wrong) -> does NOT cover B
    ]
    poses_cw = {
        "a1": {"translation": [0.5, 0.0, 0.0], "rotation": None},
        "b1": {"translation": [1.0, 0.0, 0.0], "rotation": None},
    }
    n_cov, n_rooms, covered = per_room_coverage(poses_cw, admitted_rows, ["A", "B"], cents)
    assert n_rooms == 2
    assert covered == {"A": True, "B": False}
    assert n_cov == 1


def test_per_room_coverage_ignores_none_poses():
    cents = {"A": [0.0, 0.0, 0.0]}
    admitted_rows = [{"pano_name": "a1", "room": "A"}]
    n_cov, n_rooms, covered = per_room_coverage({"a1": None}, admitted_rows, ["A"], cents)
    assert n_cov == 0 and covered == {"A": False}


def test_per_room_coverage_wrong_landing_does_not_credit_landed_room():
    # A single pano whose TRUE room is B but which lands near A's centroid. Correct code:
    # B is uncovered (its pano didn't land in B) AND A is uncovered (no pano's TRUE room is A).
    # A buggy variant that credits the LANDED room would wrongly mark A covered -> n_cov==1.
    cents = {"A": [0.0, 0.0, 0.0], "B": [10.0, 0.0, 0.0]}
    admitted_rows = [{"pano_name": "b1", "room": "B"}]
    poses_cw = {"b1": {"translation": [0.3, 0.0, 0.0], "rotation": None}}  # lands near A
    n_cov, n_rooms, covered = per_room_coverage(poses_cw, admitted_rows, ["A", "B"], cents)
    assert covered == {"A": False, "B": False}
    assert n_cov == 0
