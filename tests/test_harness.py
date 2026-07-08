"""Harness sanity tests — no external deps, no dataset needed.
Run: python tests/test_harness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import metrics  # noqa: E402


def test_room_accuracy():
    gt = {"a": "office_1", "b": "office_2", "c": "hallway_1", "d": "office_2"}
    pr = {"a": "office_1", "b": "office_2", "c": "office_5", "d": "office_2"}
    r = metrics.room_accuracy(pr, gt)
    assert r["n_scored"] == 4 and r["correct"] == 3
    assert abs(r["accuracy"] - 0.75) < 1e-9
    r2 = metrics.room_accuracy({"a": "office_1"}, gt)
    assert r2["n_scored"] == 1 and r2["n_missing"] == 3


def test_perfect_and_translation():
    gt = {"a": "r1", "b": "r2"}
    assert metrics.room_accuracy(dict(gt), gt)["accuracy"] == 1.0
    te = metrics.translation_errors({"a": [0, 0, 0], "b": [1, 2, 2]},
                                    {"a": [0, 0, 0], "b": [0, 0, 0]})
    assert abs(te["max"] - 3.0) < 1e-9  # sqrt(1+4+4) = 3


def test_rotation():
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Rz90 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    ro = metrics.rotation_errors({"a": I, "b": Rz90}, {"a": I, "b": I})
    assert abs(ro["per_uuid"]["a"]) < 1e-9
    assert abs(ro["per_uuid"]["b"] - 90.0) < 1e-6


if __name__ == "__main__":
    test_room_accuracy()
    test_perfect_and_translation()
    test_rotation()
    print("harness tests: OK")
