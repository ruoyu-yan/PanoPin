"""GPU-free integration smoke: build a real demo6_alignment.json from the cached D33 grids +
poses, then verify FGPL recovers each admitted pano's raw position. Always checks the inline
FGPL transform; additionally checks FGPL's real load_panorama_positions when importable."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from panopin import robust_score
from panopin.fgpl_export import build_matches, write_alignment_json

_REPO = Path(__file__).resolve().parent.parent
_CACHE = _REPO / "experiments" / "fgpl_seed" / "work" / "seeds"
_FGPL = Path(os.environ.get("SCAN2MEASURE_REPO", "/home/ruoyu/scan2measure-webframework"))
_ID = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

pytestmark = pytest.mark.skipif(
    not (_CACHE / "largeval_cache.json").exists()
    or not (_CACHE / "largeval_residuals.json").exists(),
    reason="D33 largeval cache absent")


def _load():
    cache = json.loads((_CACHE / "largeval_cache.json").read_text())
    grids = json.loads((_CACHE / "largeval_residuals.json").read_text())
    scores = robust_score.low_percentile_scores(grids, q=robust_score.DEPLOY_Q)
    poses = {p: {r: (v["t"], v["R"]) for r, v in cache[p]["poses"].items()} for p in cache}
    room_order = sorted({r for p in scores for r in scores[p]})
    return cache, scores, poses, room_order


def test_smoke_positions_recover_inline(tmp_path):
    cache, scores, poses, room_order = _load()
    matches, admitted = build_matches(scores, poses, room_order, _ID, tau=0.10)
    write_alignment_json(matches, admitted, tmp_path / "demo6_alignment.json")
    # identity metadata: raw == aligned; FGPL inverse is raw = R.T @ [ax,ay,0] = [ax,ay]
    for m in matches:
        exp = np.array(cache[m["pano_name"]]["poses"][m["room_label"]]["t"])[:2]
        assert np.allclose(m["camera_position"], exp, atol=1e-6)
    # every candidate room covered by the backstop
    assert {m["room_label"] for m in matches} == set(room_order)
    # per-pano uniqueness
    assert len(admitted) == len(set(admitted))


def test_smoke_through_real_fgpl_loader(tmp_path, monkeypatch):
    cache, scores, poses, room_order = _load()
    matches, admitted = build_matches(scores, poses, room_order, _ID, tau=0.10)
    out = tmp_path / "demo6_alignment.json"
    write_alignment_json(matches, admitted, out)
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps({"rotation_matrix": _ID}))
    monkeypatch.syspath_prepend(str(_FGPL / "src" / "pose_estimation"))
    try:
        from multiroom_pose_estimation import load_panorama_positions
    except Exception as e:                       # heavy deps / repo layout
        pytest.skip(f"FGPL loader not importable here: {e}")
    positions = load_panorama_positions(str(out), str(meta), admitted)
    for m in matches:
        exp = np.array(cache[m["pano_name"]]["poses"][m["room_label"]]["t"])[:2]
        assert np.allclose(positions[m["pano_name"]], exp, atol=1e-6)
