import json
import math
import numpy as np
import pytest
from panopin.fgpl_export import raw_t_to_camera_position, build_matches, write_alignment_json


_ID = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _yaw(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def test_camera_position_roundtrips_for_yaw():
    R = _yaw(37.0)
    t = [2.5, -1.3, 1.4]
    cam = raw_t_to_camera_position(t, R)
    back = (np.array(R).T @ np.array([cam[0], cam[1], 0.0]))[:2]  # FGPL's inverse
    assert np.allclose(back, [2.5, -1.3], atol=1e-6)


def test_identity_metadata_is_passthrough():
    cam = raw_t_to_camera_position([1.0, 2.0, 3.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(cam, [1.0, 2.0])


def test_non_yaw_rotation_fails_loud():
    R = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # 90 deg about x: mixes z into y
    with pytest.raises(ValueError):
        raw_t_to_camera_position([1.0, 2.0, 3.0], R)


def _poses(pairs):
    # pairs: {(pano, room): [x,y,z]} -> {pano:{room:(t, R)}}
    out = {}
    for (p, r), t in pairs.items():
        out.setdefault(p, {})[r] = (t, _ID)
    return out


def test_gate_admits_confident_omits_weak():
    scores = {
        "g1": {"A": 0.06, "B": 0.30},   # confident A
        "g2": {"A": 0.30, "B": 0.07},   # confident B
        "w":  {"A": 0.15, "B": 0.18},   # weak everywhere -> omitted
    }
    poses = _poses({("g1", "A"): [1, 0, 0], ("g2", "B"): [2, 0, 0], ("w", "A"): [3, 0, 0]})
    matches, admitted = build_matches(scores, poses, ["A", "B"], _ID, tau=0.10)
    assert sorted(admitted) == ["g1", "g2"]
    assert all(m["pano_name"] != "w" for m in matches)
    assert {m["room_label"] for m in matches} == {"A", "B"}


def test_backstop_covers_room_with_no_confident_pano():
    scores = {
        "g1": {"A": 0.06, "B": 0.30},   # confident A
        "w":  {"A": 0.40, "B": 0.15},   # B is uncovered; w is B's best (weak)
    }
    poses = _poses({("g1", "A"): [1, 0, 0], ("w", "B"): [5, 0, 0]})
    m_on, adm_on = build_matches(scores, poses, ["A", "B"], _ID, guarantee_coverage=True)
    assert {m["room_label"] for m in m_on} == {"A", "B"}      # backstop added B via w
    assert ("w" in adm_on)
    m_off, _ = build_matches(scores, poses, ["A", "B"], _ID, guarantee_coverage=False)
    assert {m["room_label"] for m in m_off} == {"A"}          # B dropped


def test_pano_appears_at_most_once():
    # g1 is confident in A AND is the global argmin for uncovered B; must NOT be double-emitted.
    scores = {
        "g1": {"A": 0.05, "B": 0.08},   # winner A (0.05); also lowest at B (0.08)
        "w":  {"A": 0.40, "B": 0.15},   # free pano; B's best FREE option
    }
    poses = _poses({("g1", "A"): [1, 0, 0], ("g1", "B"): [1, 1, 0],
                    ("w", "A"): [9, 0, 0], ("w", "B"): [5, 0, 0]})
    matches, admitted = build_matches(scores, poses, ["A", "B"], _ID, guarantee_coverage=True)
    assert admitted.count("g1") == 1
    assert len(admitted) == len(set(admitted))
    g1_rooms = [m["room_label"] for m in matches if m["pano_name"] == "g1"]
    assert g1_rooms == ["A"]                                   # g1 stays in its winner room
    assert {m["room_label"] for m in matches} == {"A", "B"}    # B covered by free pano w


def test_match_schema_and_room_idx():
    scores = {"g1": {"A": 0.06, "B": 0.30}, "g2": {"A": 0.30, "B": 0.07}}
    poses = _poses({("g1", "A"): [1, 2, 0], ("g2", "B"): [3, 4, 0]})
    matches, _ = build_matches(scores, poses, ["A", "B"], _ID, tau=0.10)
    for m in matches:
        assert set(m) == {"pano_name", "room_idx", "room_label", "score",
                          "rotation_deg", "camera_position"}
        assert m["room_idx"] == ["A", "B"].index(m["room_label"])
        assert m["rotation_deg"] == 0.0
        assert len(m["camera_position"]) == 2
        assert isinstance(m["score"], float)


def test_empty_score_matrix():
    assert build_matches({}, {}, ["A"], _ID) == ([], [])


def test_write_alignment_json_shape(tmp_path):
    matches = [{"pano_name": "g1", "room_idx": 0, "room_label": "A", "score": 0.06,
                "rotation_deg": 0.0, "camera_position": [1.0, 2.0]}]
    out = tmp_path / "demo6_alignment.json"
    write_alignment_json(matches, ["g1"], out, extra_meta={"tau": 0.10})
    d = json.loads(out.read_text())
    assert set(d) == {"metadata", "matches"}
    assert d["metadata"]["pano_names"] == ["g1"]
    assert d["metadata"]["tau"] == 0.10
    assert d["matches"] == matches
