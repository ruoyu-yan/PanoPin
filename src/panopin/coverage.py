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
