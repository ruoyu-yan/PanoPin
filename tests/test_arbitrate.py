import json
import numpy as np
import pytest
from panopin import arbitrate as arb

_ID = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _cand(i, t, origin="xdf"):
    return {"index": i, "origin": origin, "rot_idx": i % 4, "R": _ID, "t": t, "cost": -1000.0 - i,
            "n_matched": 10, "n_tight": 5, "avg_dist": 0.1}


def _doc(n=3, choice=0):
    return {"pano": "p", "fgpl_choice": choice, "candidates": [_cand(i, [float(i), 0.0, 1.5]) for i in range(n)]}


def test_score_candidates_is_the_low_percentile_and_empty_is_999():
    r = [np.linspace(0, 1, 101), np.array([]), np.full(10, 0.2)]
    s = arb.score_candidates(r)
    assert abs(s[0] - 0.20) < 1e-9 and s[1] == 999.0 and s[2] == 0.2


def test_select_takes_the_minimum_and_records_fgpls_choice():
    out = arb.select(_doc(3, choice=0), [0.12, 0.04, 0.09])
    assert out["index"] == 1 and out["t"] == [1.0, 0.0, 1.5] and out["R"] == _ID
    assert out["score"] == 0.04 and out["origin"] == "xdf"
    assert out["fgpl_choice"] == {"index": 0, "score": 0.12}
    assert out["scores"] == [0.12, 0.04, 0.09]


def test_select_breaks_ties_by_lower_index():
    assert arb.select(_doc(3), [0.05, 0.05, 0.05])["index"] == 0


def test_select_refuses_mismatched_lengths_and_empty_lists():
    with pytest.raises(ValueError):
        arb.select(_doc(3), [0.1, 0.2])
    with pytest.raises(ValueError):
        arb.select({"pano": "p", "fgpl_choice": 0, "candidates": []}, [])


def _scene(tmp_path, panos=("pA", "pB"), rooms=("r1", "r2")):
    pano_paths = {p: str(tmp_path / f"{p}.png") for p in panos}
    cloud_paths = {r: str(tmp_path / f"{r}.txt") for r in rooms}
    alignment = {"metadata": {}, "matches": [{"pano_name": p, "room_label": r} for p, r in zip(panos, rooms)]}
    cdir = tmp_path / "poses"
    for p in panos:
        (cdir / p).mkdir(parents=True)
        (cdir / p / "candidates.json").write_text(json.dumps(_doc(3)))
    return pano_paths, cloud_paths, alignment, cdir


def test_arbitrate_scores_each_pano_against_its_own_room_cloud(tmp_path):
    pano_paths, cloud_paths, alignment, cdir = _scene(tmp_path)
    seen = []

    def scorer(pano_path, cloud_path, poses):
        seen.append((pano_path, cloud_path, len(poses)))
        return [0.3, 0.1, 0.2] if pano_path.endswith("pA.png") else [0.1, 0.2, 0.3]

    out = arb.arbitrate(pano_paths, cloud_paths, alignment, cdir, scorer)
    assert seen == [(pano_paths["pA"], cloud_paths["r1"], 3), (pano_paths["pB"], cloud_paths["r2"], 3)]
    assert out["pA"]["index"] == 1 and out["pB"]["index"] == 0


def test_arbitrate_skips_alignment_panos_not_being_localized(tmp_path):
    pano_paths, cloud_paths, alignment, cdir = _scene(tmp_path)
    del pano_paths["pB"]
    out = arb.arbitrate(pano_paths, cloud_paths, alignment, cdir, lambda *a: [0.1, 0.2, 0.3])
    assert set(out) == {"pA"}


def test_arbitrate_fails_loud_on_a_missing_room_cloud_or_candidates_file(tmp_path):
    pano_paths, cloud_paths, alignment, cdir = _scene(tmp_path)
    del cloud_paths["r2"]
    with pytest.raises(KeyError):
        arb.arbitrate(pano_paths, cloud_paths, alignment, cdir, lambda *a: [0.1, 0.2, 0.3])
    pano_paths, cloud_paths, alignment, cdir = _scene(tmp_path / "b")
    (cdir / "pB" / "candidates.json").unlink()
    with pytest.raises(FileNotFoundError):
        arb.arbitrate(pano_paths, cloud_paths, alignment, cdir, lambda *a: [0.1, 0.2, 0.3])


def test_select_refuses_an_fgpl_choice_outside_the_candidates():
    doc = _doc(3, choice=3)
    doc["pano"] = "pX"
    with pytest.raises(ValueError, match="pX"):
        arb.select(doc, [0.1, 0.2, 0.3])


def test_select_refuses_a_candidate_index_that_is_not_its_position():
    doc = _doc(3)
    doc["pano"] = "pY"
    doc["candidates"][1]["index"] = 2
    with pytest.raises(ValueError, match=r"pY.*position 1"):
        arb.select(doc, [0.1, 0.2, 0.3])


def test_gpu_scorer_forwards_poses_once_and_scores_the_low_percentile(monkeypatch):
    from panopin import cpo_adapter, robust_score
    res = [np.linspace(0, 1, 11), np.linspace(2, 3, 21)]
    calls = []

    def fake(cfg, pano_path, cloud_path, poses, **kw):
        calls.append((cfg, pano_path, cloud_path, poses))
        return [r.copy() for r in res]

    monkeypatch.setattr(cpo_adapter, "residuals_at_poses", fake)
    poses = [([0.0, 0.0, 1.5], _ID), ([5.0, 0.0, 1.5], _ID)]
    cfg = object()
    out = arb.gpu_scorer(cfg)("fake.png", "fake.txt", poses)
    assert len(calls) == 1
    assert calls[0] == (cfg, "fake.png", "fake.txt", poses)
    assert out == [float(np.percentile(r, robust_score.DEPLOY_Q)) for r in res]
