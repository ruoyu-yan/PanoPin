"""Colour arbitration over FGPL's refined candidates (spec 2026-10-07 §6).

FGPL ranks its candidates by line matching, which prefers a 90 or 180 degree turned pose in a
symmetric room (Area_2_manhattan4/office_8: flipped in 5 of 6 runs from a seed within 1 cm).
PanoPin's colour residual evaluated AT a candidate pose tells the two apart by about 3x
(2026-10-06: 0.127 vs 0.035; 0.106 vs 0.036). This module picks, per panorama, the candidate
with the lowest low-percentile residual against the panorama's own room cloud, and records
every score so a wrong pick can be audited. Reads panos, clouds, candidates; never GT."""
import json
from pathlib import Path

import numpy as np

from panopin import robust_score

MISSING = 999.0   # the score of an empty residual vector, as seed._grid reports it


def score_candidates(residuals, q=robust_score.DEPLOY_Q):
    """One low-percentile score per residual vector (lower = better); empty -> MISSING."""
    return [float(np.percentile(r, q)) if len(r) else MISSING for r in residuals]


def select(candidates_doc, scores):
    """candidates.json content + one score per candidate -> the arbitration record."""
    cands = candidates_doc["candidates"]
    if not cands:
        raise ValueError(f"{candidates_doc.get('pano')}: no candidates to choose from")
    if len(cands) != len(scores):
        raise ValueError(f"{candidates_doc.get('pano')}: {len(cands)} candidates but {len(scores)} scores")
    best = min(range(len(scores)), key=lambda i: (scores[i], i))
    j = int(candidates_doc["fgpl_choice"])
    return {"R": cands[best]["R"], "t": cands[best]["t"], "index": int(cands[best]["index"]),
            "origin": cands[best]["origin"], "score": float(scores[best]),
            "fgpl_choice": {"index": j, "score": float(scores[j])},
            "scores": [float(s) for s in scores]}


def arbitrate(panos, clouds, alignment, candidates_dir, scorer):
    """panos {pano: image path}; clouds {room: cloud path}; alignment = demo6_alignment.json
    content (pano -> assigned room); candidates_dir holds <pano>/candidates.json;
    scorer(pano_path, cloud_path, [(t, R), ...]) -> [score, ...]. Returns {pano: select(...)}.

    An alignment pano absent from `panos` is skipped (not being localized). A room absent from
    `clouds` is a KeyError and a pano without candidates a FileNotFoundError: both mean an
    earlier stage did not do its job, and a silent fallback would hide that."""
    out = {}
    for m in alignment["matches"]:
        pano = m["pano_name"]
        if pano not in panos:
            continue
        cloud = clouds[m["room_label"]]
        path = Path(candidates_dir) / pano / "candidates.json"
        if not path.is_file():
            raise FileNotFoundError(f"{pano}: no candidates at {path} -- the estimator did not finish it")
        doc = json.loads(path.read_text())
        poses = [(c["t"], c["R"]) for c in doc["candidates"]]
        out[pano] = select(doc, scorer(panos[pano], cloud, poses))
    return out


def gpu_scorer(cfg):
    """The deployed scorer: PanoPin's residuals at each pose, one cloud load per panorama."""
    from panopin.cpo_adapter import residuals_at_poses

    def scorer(pano_path, cloud_path, poses):
        return score_candidates(residuals_at_poses(cfg, pano_path, cloud_path, poses))
    return scorer
