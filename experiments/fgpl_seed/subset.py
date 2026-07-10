"""Resolve the 6-room same-shape subset to (pano_name, uuid, room, jpg, cloud) rows.
Uses the PanoPin harness GT loader; pano_name is the uuid (clean, collision-free)."""
import os, sys
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from eval import s3dis_gt

ROOMS = ["office_1", "office_4", "office_5", "office_6", "office_7", "hallway_3"]
MAX_PANOS_PER_ROOM = 2


def _resolve_pano(uuid, pano_rgb_dir):
    for p in sorted(os.listdir(pano_rgb_dir)):
        parts = p.split("_")
        if len(parts) >= 2 and parts[1] == uuid:
            return os.path.join(pano_rgb_dir, p)
    return None


def build_subset():
    cfg = s3dis_gt.load_config(None)
    a = cfg["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", cfg)
    by_room = {}
    for u, v in gt.items():
        by_room.setdefault(v["room"], []).append(u)
    rows = []
    for room in ROOMS:
        cloud = os.path.join(a["rooms_dir"], room, room + ".txt")
        for u in sorted(by_room.get(room, []))[:MAX_PANOS_PER_ROOM]:
            jpg = _resolve_pano(u, a["pano_rgb_dir"])
            if jpg:
                rows.append({"pano_name": u, "uuid": u, "room": room,
                             "pano_jpg": jpg, "cloud_txt": cloud})
    return rows
