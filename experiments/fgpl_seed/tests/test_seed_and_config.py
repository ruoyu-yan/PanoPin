import json
from experiments.fgpl_seed import seed_and_config as sc

ROWS = [
    {"pano_name": "u1", "uuid": "u1", "room": "office_4", "cloud_txt": "/x", "pano_jpg": "/x.jpg"},
    {"pano_name": "u2", "uuid": "u2", "room": "office_6", "cloud_txt": "/y", "pano_jpg": "/y.jpg"},
]
GT = {"u1": {"location": [1.0, 2.0, 1.4], "R_cw": [[1,0,0],[0,1,0],[0,0,1]], "room": "office_4"},
      "u2": {"location": [8.0, 9.0, 1.4], "R_cw": [[1,0,0],[0,1,0],[0,0,1]], "room": "office_6"}}
CPO = {"u1": {"t": [1.1, 2.1, 1.3], "R": [[1,0,0],[0,1,0],[0,0,1]], "loss": 0.1, "room": "office_4"},
       "u2": {"t": [7.9, 9.2, 1.5], "R": [[1,0,0],[0,1,0],[0,0,1]], "loss": 0.1, "room": "office_6"}}

def test_oracle_seed_uses_gt_xy(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.paths, "WORK", tmp_path)
    p = sc.write_seed("oracle", ROWS, GT, CPO)
    m = {x["pano_name"]: x["camera_position"] for x in json.load(open(p))["matches"]}
    assert m["u1"] == [1.0, 2.0]                      # GT xy, z dropped
    assert m["u2"] == [8.0, 9.0]

def test_p1_seed_uses_cpo_xy(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.paths, "WORK", tmp_path)
    p = sc.write_seed("p1", ROWS, GT, CPO)
    m = {x["pano_name"]: x["camera_position"] for x in json.load(open(p))["matches"]}
    assert m["u1"] == [1.1, 2.1]                      # CPO t xy

def test_wrong_room_seed_uses_sibling_centroid(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.paths, "WORK", tmp_path)
    # sibling of office_4 is office_6 -> u1 should be seeded at u2's room area, not its own
    cents = {"office_4": [1.0, 2.0, 1.4], "office_6": [8.0, 9.0, 1.4]}
    p = sc.write_seed("wrong_room", ROWS, GT, CPO, centroids=cents)
    m = {x["pano_name"]: x["camera_position"] for x in json.load(open(p))["matches"]}
    assert m["u1"] == [8.0, 9.0]                      # office_6 centroid (deliberately wrong)

def test_identity_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.paths, "WORK", tmp_path)
    p = sc.write_identity_metadata()
    md = json.load(open(p))
    assert md["rotation_matrix"] == [[1,0,0],[0,1,0],[0,0,1]]

BASE_CONFIG_KEYS = {
    "point_cloud_name", "pano_names", "use_local_filtering", "pkl_3d_path",
    "alignment_path", "metadata_path", "point_cloud_path", "features_2d_dir",
    "pano_dir", "output_dir",
}

def test_write_config_key_contract_no_narrowing(tmp_path, monkeypatch):
    # Load-bearing: the external FGPL estimator reads exactly these keys.
    # A wrong/missing/renamed key makes it silently fall back to TMB defaults.
    monkeypatch.setattr(sc.paths, "WORK", tmp_path)
    p = sc.write_config("p1", ROWS, "/seed.json", "/line_map.pkl",
                         "/metadata.json", "/feat_dir", "/pano_dir", narrowing=None)
    cfg = json.load(open(p))
    assert set(cfg.keys()) == BASE_CONFIG_KEYS
    assert cfg["pano_names"] == [r["pano_name"] for r in ROWS]
    assert cfg["use_local_filtering"] is True

def test_write_config_key_contract_with_narrowing(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.paths, "WORK", tmp_path)
    p = sc.write_config("p1", ROWS, "/seed.json", "/line_map.pkl",
                         "/metadata.json", "/feat_dir", "/pano_dir",
                         narrowing={"seed_trans_radius": 2.0})
    cfg = json.load(open(p))
    assert set(cfg.keys()) == BASE_CONFIG_KEYS | {"seed_trans_radius"}
    assert cfg["seed_trans_radius"] == 2.0
    assert cfg["pano_names"] == [r["pano_name"] for r in ROWS]
    assert cfg["use_local_filtering"] is True
