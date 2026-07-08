"""Metrics for the PanoPin harness. Stdlib-only.

Primary: room-assignment accuracy (did each pano get put in the right room?).
Secondary: coarse-pose error for panos where the solver supplies a pose
  - translation error (metres): || t_pred - t_gt ||
  - rotation error (degrees): geodesic angle between camera->world rotations
    (convention: R is camera->world, matching the pipeline's Stage-2 projection).
"""
from __future__ import annotations
import math


def room_accuracy(pred_rooms: dict, gt_rooms: dict):
    scored = [u for u in gt_rooms if u in pred_rooms]
    missing = [u for u in gt_rooms if u not in pred_rooms]
    correct = sum(1 for u in scored if pred_rooms[u] == gt_rooms[u])
    n = len(scored)
    per_room = {}
    for u in scored:
        d = per_room.setdefault(gt_rooms[u], {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += int(pred_rooms[u] == gt_rooms[u])
    return {
        "n_scored": n,
        "n_missing": len(missing),
        "correct": correct,
        "accuracy": (correct / n) if n else 0.0,
        "per_room": per_room,
        "missing_uuids": missing,
    }


def _norm(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def translation_errors(pred_poses: dict, gt_locations: dict):
    errs = {u: _norm(p, gt_locations[u])
            for u, p in pred_poses.items()
            if u in gt_locations and p is not None}
    return _summarize(errs)


def _trace_AtB(A, B):
    """trace(A^T B) = sum_ij A[i][j] * B[i][j]."""
    return sum(A[i][j] * B[i][j] for i in range(3) for j in range(3))


def rotation_errors(pred_R: dict, gt_R: dict):
    errs = {}
    for u, R in pred_R.items():
        if u in gt_R and R is not None:
            c = max(-1.0, min(1.0, (_trace_AtB(R, gt_R[u]) - 1.0) / 2.0))
            errs[u] = math.degrees(math.acos(c))
    return _summarize(errs)


def _summarize(errs: dict):
    vals = sorted(errs.values())
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "max": None, "per_uuid": {}}
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    return {"n": n, "mean": sum(vals) / n, "median": median, "max": vals[-1], "per_uuid": errs}


if __name__ == "__main__":
    gtr = {"a": "office_1", "b": "office_2", "c": "hallway_1"}
    pr = {"a": "office_1", "b": "office_2", "c": "office_5"}
    assert abs(room_accuracy(pr, gtr)["accuracy"] - 2 / 3) < 1e-9
    te = translation_errors({"a": [0, 0, 0], "b": [3, 4, 0]}, {"a": [0, 0, 0], "b": [0, 0, 0]})
    assert abs(te["max"] - 5.0) < 1e-9
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert abs(rotation_errors({"a": I}, {"a": I})["max"]) < 1e-9
    print("metrics self-test: OK")
