"""S3DIS ground-truth loader for the PanoPin harness.

Loads, per S3DIS area:
  - pano -> room assignment GT (canonical room name)
  - pano -> camera pose GT (location + camera->world rotation)
  - the candidate room set (all rooms in the area, from Stanford3dDataset)

GT is derived from the 2D-3D-S per-pano pose JSONs (which also carry the room and
corroborate camera_to_room.json). Room names are normalized by stripping the trailing
"_<areaindex>" so they match the Stanford3dDataset room folders (office_3_3 -> office_3).

FAIRNESS: the room token is present in pano filenames and pose JSONs. It is GROUND TRUTH.
Solvers must never read it; they see only anonymized panos + candidate room clouds
(see make_manifest.py). This module is HARNESS-ONLY.

Stdlib-only by design (portable, fast).
"""
from __future__ import annotations
import glob
import json
import os
import re
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    p = Path(path) if path else repo_root() / "config" / "datasets.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def canonical_room(name: str) -> str:
    """'office_3_3' -> 'office_3', 'WC_1_3' -> 'WC_1'."""
    return re.sub(r"_\d+$", "", name)


def _transpose3(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def load_gt(area="Area_3", cfg=None):
    """Return {uuid: {'room','location','rt','R_cw'}} for one area."""
    cfg = cfg or load_config()
    a = cfg["s3dis"][area]
    gt = {}
    for pf in sorted(glob.glob(os.path.join(a["pano_pose_dir"], "*.json"))):
        with open(pf, "r", encoding="utf-8") as f:
            d = json.load(f)
        uuid = d["camera_uuid"]
        rt = d["camera_rt_matrix"]                 # 3x4 [R_wc | t_wc], world->camera
        R_wc = [row[:3] for row in rt]
        gt[uuid] = {
            "room": canonical_room(d["room"]),
            "location": list(d["camera_location"]),
            "rt": rt,
            "R_cw": _transpose3(R_wc),             # camera->world (pipeline Stage-2 convention)
        }
    return gt


def candidate_rooms(area="Area_3", cfg=None):
    """All room names available in the area's Stanford3dDataset dir (sorted)."""
    cfg = cfg or load_config()
    rooms_dir = Path(cfg["s3dis"][area]["rooms_dir"])
    return [p.name for p in sorted(rooms_dir.iterdir()) if p.is_dir()]


def gt_rooms(gt):
    return {u: v["room"] for u, v in gt.items()}


def gt_locations(gt):
    return {u: v["location"] for u, v in gt.items()}


if __name__ == "__main__":
    from collections import Counter
    gt = load_gt()
    rooms = candidate_rooms()
    dist = Counter(v["room"] for v in gt.values())
    print(f"panos with GT    : {len(gt)}")
    print(f"candidate rooms  : {len(rooms)} -> {rooms}")
    print(f"rooms with panos : {len(dist)}")
    print("panos per room   :")
    for r, c in sorted(dist.items()):
        print(f"    {r:20s} {c}")
    u0, v0 = next(iter(gt.items()))
    print("sample uuid      :", u0)
    print("sample room/loc  :", v0["room"], v0["location"])
