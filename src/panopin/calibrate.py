"""Per-room loss calibration + confidence gate for PanoPin room assignment (D22/D24).

Raw CPO loss is degenerate across rooms: a fixed set of small/corridor "loss-sink" rooms
scores low for every pano, burying the true room (D21). Calibration fixes this WITHOUT
ground truth by normalizing each candidate room's loss against that room's own loss
distribution over the other panos (leave-one-out on the scored pano; no room labels — D5).

Min-max calibration (fair sweep winner, D24): score(p, r) = (loss(p,r) - min_q loss(q,r)) /
(max_q loss(q,r) - min_q loss(q,r)), q != p. A genuine match makes p the room's minimum
-> score near/below 0; a loss-sink (low for everyone) gives every pano a middling score.
(An earlier `percentile` variant looked stronger but only via a GT leak in the baseline —
fairly it is no better than raw; see DECISIONS D24.)

Assign = min-score room. confidence = the winner's score (lower = more confident). A pano
whose winner-score exceeds `conf_threshold` is FLAGGED (fails to match any room well —
window/occlusion/out-of-frame, D23) rather than silently mis-assigned. NOTE: the gate is a
precision/coverage tradeoff, not a clean separator (D24) — tune the threshold to the
required precision.
"""
from collections import namedtuple

Assignment = namedtuple("Assignment", "room confidence is_confident")


def minmax_scores(loss_matrix):
    """loss_matrix: {pano: {room: loss}}. Returns {pano: {room: score}}; lower = better.
    Per-room min/max are leave-one-out on the scored pano (fair; no GT)."""
    panos = list(loss_matrix)
    if not panos:
        return {}
    rooms = list(loss_matrix[panos[0]])
    out = {}
    for p in panos:
        out[p] = {}
        for r in rooms:
            others = [loss_matrix[q][r] for q in panos if q != p]
            if not others:
                out[p][r] = 0.5
                continue
            lo, hi = min(others), max(others)
            out[p][r] = (loss_matrix[p][r] - lo) / (hi - lo) if hi > lo else 0.0
    return out


def assign(loss_matrix, conf_threshold=-0.3):
    """Return {pano: Assignment(room, confidence, is_confident)}.
    room = min-score room; confidence = its score (lower = better);
    is_confident = confidence <= conf_threshold (default -0.3 favors precision)."""
    ss = minmax_scores(loss_matrix)
    res = {}
    for p, scores in ss.items():
        room = min(scores, key=lambda r: scores[r])
        conf = scores[room]
        res[p] = Assignment(room, conf, conf <= conf_threshold)
    return res
