"""Pure per-pano candidate-selection rules for FGPL⊕PanoPin fusion (fair: reads only
scores, no GT — D5). Candidate = dict with 'geom' (line-inlier count, higher=better) and
'color' (mean color residual, lower=better). F1 = verify-select (InLoc pattern: color
alone selects). F2 = rank/score blends. See docs/specs/2026-07-14-candidate-fusion-design.md."""


def verify_select(candidates):
    """F1: pick the lowest-color candidate. Geometry only defined the pool."""
    return min(range(len(candidates)), key=lambda i: candidates[i]["color"])
