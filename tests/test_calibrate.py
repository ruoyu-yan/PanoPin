"""Unit tests for the fair (no-GT) calibration + confidence gate (D24).
Deterministic hand-computable matrices — no CPO, no data."""
from panopin.calibrate import assign, minmax_scores


def test_calibration_demotes_loss_sink():
    # 'sink' is low for EVERY pano (a loss-sink); each pano genuinely matches its own room.
    # Raw argmin would pick 'sink' for pa/pb; calibration must recover the true room.
    lm = {
        "pa": {"r1": 0.10, "r2": 0.50, "sink": 0.05},
        "pb": {"r1": 0.40, "r2": 0.10, "sink": 0.06},
        "pc": {"r1": 0.60, "r2": 0.55, "sink": 0.04},
    }
    assert min(lm["pa"], key=lm["pa"].get) == "sink"   # raw would be wrong
    res = assign(lm, conf_threshold=-0.3)
    assert res["pa"].room == "r1"
    assert res["pb"].room == "r2"
    assert res["pa"].is_confident        # strong match -> low (negative) score -> confident


def test_scores_leave_one_out_and_range():
    lm = {"pa": {"r1": 0.1}, "pb": {"r1": 0.4}, "pc": {"r1": 0.6}}
    s = minmax_scores(lm)
    # pa vs r1: others {0.4, 0.6} -> (0.1-0.4)/(0.6-0.4) = -1.5
    assert abs(s["pa"]["r1"] - (-1.5)) < 1e-9


def test_empty_matrix():
    assert assign({}) == {}
