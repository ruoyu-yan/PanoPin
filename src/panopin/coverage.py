"""Deployment room seeding for the PanoPin -> FGPL hand-off (D32).

In the real deployment (a 3-5 room cloud, EVERY room covered by >=1 pano) the success metric is
per-ROOM COVERAGE — one correct seed per room for FGPL — NOT per-pano recall. Per-pano recall is
capped at ~75-92% by weak-lock panos (window/occlusion; D23/D31 hard floor, ~1/3 of panos), which
no score-matrix trick recovers. But those weak panos are never a room's ONLY pano and they carry
the WEAKEST color lock, so two threshold-free rules give full room coverage at 100% precision
offline (COVERAGE_RESULTS.md, both caches, k in {3..6}):

  room_anchored_seeds  each room picks its best-matching pano -> 100% correct, loss-sink-immune.
  pano_confidence      per-pano (room, confidence=-winner_score); ranking admits weak-lock panos
                       LAST, so a confidence gate hands FGPL zero wrong seeds.

Input is a per-room score matrix {pano: {room: score}} (lower = better), e.g. from
`robust_score.low_percentile_scores`. Fair: reads only scores, no GT (D5). Caveat: validated at
n=12; a larger-pano GPU run should confirm.
"""
import numpy as np


def room_anchored_seeds(score_matrix):
    """{pano: {room: score}} -> {room: (pano, score)}: each room seeded by its best-matching pano.

    Threshold-free and loss-sink-immune: a loss-sink room attracts many panos under per-pano
    argmin, but here we ask which pano best matches the ROOM — and a room's genuine panos match it
    best — so each room self-seeds with one of its own panos. Note: a pano may seed >1 room (rare);
    that is acceptable for the FGPL hand-off (each room gets an independent seed to refine)."""
    if not score_matrix:
        return {}
    panos = list(score_matrix)
    rooms = list(score_matrix[panos[0]])
    return {r: (min(panos, key=lambda p: score_matrix[p][r]),
                min(score_matrix[p][r] for p in panos)) for r in rooms}


def pano_confidence(score_matrix):
    """{pano: {room: score}} -> {pano: (room, confidence)}: room = min-score room; confidence =
    -winner_score (higher = more confident). Genuine locks have a low absolute score (~0.06-0.08),
    weak-lock panos ~0.12+, so sorting by confidence descending puts weak-lock panos last. Admit
    high-confidence panos (gate) to hand FGPL only trustworthy seeds while still covering every
    room via its strong panos."""
    out = {}
    for p, sc in score_matrix.items():
        room = min(sc, key=lambda r: sc[r])
        out[p] = (room, -sc[room])
    return out


def assign_rooms(score_matrix, room_order):
    """{pano: {room: score}} x room_order -> {pano: room}: the one-to-one assignment of panos
    to rooms with the smallest total score (Hungarian method, scipy). With more panos than
    rooms every room gets exactly one pano and the rest are absent from the result; with more
    rooms than panos every pano gets a room and some rooms are absent. Threshold-free.

    Why not per-pano argmin: two corridors score within a few percent of each other for BOTH
    corridor panos, so argmin seeds both in the same corridor, and FGPL then searches the wrong
    corridor and never leaves it (Area_2_manhattan7, 2026-10-05: hallway_2 11 m off). On the
    three score matrices measured on 2026-10-06 argmin placed 6 of 7 panos, this 7 of 7."""
    if not score_matrix:
        return {}
    from scipy.optimize import linear_sum_assignment
    panos = list(score_matrix)
    rooms = list(room_order)
    cost = np.array([[float(score_matrix[p][r]) for r in rooms] for p in panos])
    rows, cols = linear_sum_assignment(cost)
    return {panos[i]: rooms[j] for i, j in zip(rows, cols)}
