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


def test_backstop_no_free_pano_leaves_room_uncovered():
    # Only one pano exists (g1), gated to A; B has no confident pano AND no free pano left to
    # borrow from (g1 is already assigned) -> B stays uncovered, exercising `if not free: continue`.
    scores = {"g1": {"A": 0.06, "B": 0.30}}
    poses = _poses({("g1", "A"): [1, 0, 0], ("g1", "B"): [1, 1, 0]})
    matches, admitted = build_matches(scores, poses, ["A", "B"], _ID, guarantee_coverage=True)
    assert {m["room_label"] for m in matches} == {"A"}
    assert admitted.count("g1") == 1


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
                          "rotation_deg", "camera_position", "assignment"}
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


def test_write_alignment_json_creates_parent_dir(tmp_path):
    matches = [{"pano_name": "g1", "room_idx": 0, "room_label": "A", "score": 0.06,
                "rotation_deg": 0.0, "camera_position": [1.0, 2.0]}]
    out = tmp_path / "nested" / "sub" / "demo6_alignment.json"
    write_alignment_json(matches, ["g1"], out, extra_meta={"tau": 0.10})
    assert out.exists()


def test_export_alignment_end_to_end(tmp_path):
    from panopin.fgpl_export import export_alignment
    scores = {"g1": {"A": 0.06, "B": 0.30}, "g2": {"A": 0.30, "B": 0.07}}
    poses = _poses({("g1", "A"): [1, 2, 0], ("g2", "B"): [3, 4, 0]})
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps({"rotation_matrix": _ID}))
    out = tmp_path / "demo6_alignment.json"
    admitted = export_alignment(scores, poses, ["A", "B"], meta_path, out, tau=0.10)
    assert sorted(admitted) == ["g1", "g2"]
    d = json.loads(out.read_text())
    assert d["metadata"]["pano_names"] == admitted
    assert d["metadata"]["tau"] == 0.10
    assert len(d["matches"]) == 2


def test_joint_assignment_separates_tied_corridor_panos_and_tags_them():
    scores = {"h2": {"c0": 0.082, "c3": 0.084}, "h3": {"c0": 0.083, "c3": 0.095}}
    poses = _poses({("h2", "c0"): [1, 0, 0], ("h2", "c3"): [2, 0, 0],
                    ("h3", "c0"): [3, 0, 0], ("h3", "c3"): [4, 0, 0]})
    matches, admitted = build_matches(scores, poses, ["c0", "c3"], _ID, tau=0.10)
    by = {m["pano_name"]: m for m in matches}
    assert sorted(admitted) == ["h2", "h3"]
    assert by["h2"]["room_label"] == "c3" and by["h3"]["room_label"] == "c0"
    assert by["h2"]["assignment"] == "joint" and by["h3"]["assignment"] == "joint"
    assert by["h2"]["camera_position"] == [2.0, 0.0]     # the pose of the ASSIGNED room
    assert by["h2"]["score"] == 0.084


def test_more_panos_than_rooms_uses_the_legacy_rule_for_all():
    scores = {"g1": {"A": 0.06, "B": 0.30}, "g2": {"A": 0.30, "B": 0.07},
              "x": {"A": 0.08, "B": 0.20}}                 # third pano, two rooms
    poses = _poses({("g1", "A"): [1, 0, 0], ("g2", "B"): [2, 0, 0], ("x", "A"): [3, 0, 0]})
    matches, admitted = build_matches(scores, poses, ["A", "B"], _ID, tau=0.10)
    by = {m["pano_name"]: m for m in matches}
    assert sorted(admitted) == ["g1", "g2", "x"]
    assert by["g1"]["room_label"] == "A" and by["g2"]["room_label"] == "B"
    assert by["x"]["room_label"] == "A"
    assert all(m["assignment"] == "argmin" for m in matches)
    assert len(admitted) == len(set(admitted))


def test_a_confident_pano_is_never_forced_into_an_empty_room():
    # Z is a segment with no genuine pano. A one-to-one assignment would move a1 into Z.
    scores = {"a1": {"A": 0.06, "B": 0.40, "Z": 0.20}, "a2": {"A": 0.07, "B": 0.40, "Z": 0.30},
              "b1": {"A": 0.40, "B": 0.06, "Z": 0.25}, "w": {"A": 0.15, "B": 0.16, "Z": 0.30}}
    poses = _poses({("a1", "A"): [1, 0, 0], ("a2", "A"): [2, 0, 0],
                    ("b1", "B"): [3, 0, 0], ("w", "Z"): [4, 0, 0]})
    matches, admitted = build_matches(scores, poses, ["A", "B", "Z"], _ID, tau=0.10)
    by = {m["pano_name"]: m for m in matches}
    assert sorted(admitted) == ["a1", "a2", "b1", "w"]
    assert by["a1"]["room_label"] == "A" and by["a2"]["room_label"] == "A"
    assert by["b1"]["room_label"] == "B" and by["w"]["room_label"] == "Z"   # backstop
    assert not any(m["assignment"] == "joint" for m in matches)


def test_joint_pair_is_admitted_even_above_tau():
    scores = {"p": {"A": 0.05, "B": 0.50}, "q": {"A": 0.60, "B": 0.40}}   # q only covers B
    poses = _poses({("p", "A"): [1, 0, 0], ("q", "B"): [2, 0, 0]})
    matches, admitted = build_matches(scores, poses, ["A", "B"], _ID, tau=0.10)
    by = {m["pano_name"]: m for m in matches}
    assert sorted(admitted) == ["p", "q"]
    assert by["q"]["room_label"] == "B" and by["q"]["score"] == 0.40
    assert by["q"]["assignment"] == "joint" and by["p"]["assignment"] == "joint"


def test_legacy_gate_without_coverage_is_per_pano():
    scores = {"g1": {"A": 0.06, "B": 0.30}, "w": {"A": 0.40, "B": 0.15}}
    poses = _poses({("g1", "A"): [1, 0, 0], ("w", "B"): [5, 0, 0]})
    matches, admitted = build_matches(scores, poses, ["A", "B"], _ID, guarantee_coverage=False)
    assert admitted == ["g1"] and matches[0]["assignment"] == "argmin"
