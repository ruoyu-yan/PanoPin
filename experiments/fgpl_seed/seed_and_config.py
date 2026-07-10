"""Identity metadata + per-arm demo6 seed + per-arm estimator config.
All positions are RAW S3DIS 3D; only XY is emitted (estimator maps [x,y]->[x,y,0])."""
import json, numpy as np
from experiments.fgpl_seed import paths

I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Deliberately-wrong same-shape target for the Wrong-room floor arm.
SIBLING = {"office_4": "office_6", "office_6": "office_4",
           "office_5": "office_7", "office_7": "office_5",
           "office_1": "office_5", "hallway_3": "office_1"}


def _seeds_dir():
    d = paths.WORK / "seeds"; d.mkdir(parents=True, exist_ok=True); return d

def _configs_dir():
    d = paths.WORK / "configs"; d.mkdir(parents=True, exist_ok=True); return d

def _density_png_path():
    return _configs_dir() / "density.png"

def _ensure_density_png():
    """Write the blank 256x256 viz-only density placeholder if absent. The estimator
    does density_img.shape unconditionally (multiroom_pose_estimation.py:259) on a
    cv2-loaded image; ours is viz-only (topdown render, after camera_pose.json is
    written) since we run in the raw S3DIS frame with no floorplan. Called by BOTH
    write_identity_metadata AND write_config so a config never points at a missing
    file regardless of call order (review fix)."""
    p = _density_png_path()
    if not p.exists():
        import cv2
        cv2.imwrite(str(p), np.zeros((256, 256), dtype=np.uint8))
    return p


def write_identity_metadata():
    md = {"min_coords": [0.0, 0.0, 0.0], "max_dim": 1.0, "offset": [0.0, 0.0],
          "image_width": 256, "image_height": 256, "translation": [0.0, 0.0, 0.0],
          "rotation_matrix": I3}
    p = _configs_dir() / "metadata.json"
    with open(p, "w") as f:
        json.dump(md, f, indent=2)
    _ensure_density_png()
    return p


def room_centroids(rows):
    cents = {}
    for c in sorted({r["cloud_txt"] for r in rows}):
        arr = np.loadtxt(c)
        room = [r["room"] for r in rows if r["cloud_txt"] == c][0]
        cents[room] = arr[:, :3].mean(axis=0).tolist()
    return cents


def _pos_for(arm, r, gt, cpo, centroids):
    name, room = r["pano_name"], r["room"]
    if arm == "oracle":
        return gt[name]["location"][:2]
    if arm == "wrong_room":
        sib = SIBLING[room]
        return centroids[sib][:2]
    return cpo[name]["t"][:2]                       # p1/p2/p3 all use CPO position


def write_seed(arm, rows, gt, cpo, centroids=None):
    if arm == "wrong_room" and centroids is None:
        centroids = room_centroids(rows)
    matches = []
    for r in rows:
        pos = _pos_for(arm, r, gt, cpo, centroids)
        matches.append({"pano_name": r["pano_name"], "room_label": r["room"],
                        "room_idx": 0, "score": 1.0, "rotation_deg": 0.0,
                        "camera_position": [float(pos[0]), float(pos[1])]})
    doc = {"metadata": {"pipeline": f"panopin-seed:{arm}"}, "matches": matches}
    p = _seeds_dir() / f"{arm}.json"
    with open(p, "w") as f:
        json.dump(doc, f, indent=2)
    return p


def write_config(arm, rows, seed_path, line_map, metadata_path, feat_dir, pano_dir, narrowing=None):
    _ensure_density_png()   # guarantee density_image_path resolves, independent of call order
    cfg = {
        "point_cloud_name": paths.SCENE,
        "pano_names": [r["pano_name"] for r in rows],
        "use_local_filtering": True,
        "pkl_3d_path": str(line_map),
        "alignment_path": str(seed_path),
        "metadata_path": str(metadata_path),
        "density_image_path": str(_density_png_path()),
        "point_cloud_path": str(paths.WORK / "clouds" / f"{paths.SCENE}.ply"),
        "features_2d_dir": str(feat_dir),
        "pano_dir": str(pano_dir),
        "output_dir": str(paths.WORK / "poses" / arm),
    }
    if narrowing:
        cfg.update(narrowing)                        # seed_trans_radius / seed_yaw_deg / seed_yaw_tol
    p = _configs_dir() / f"pose_{arm}.json"
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    return p
