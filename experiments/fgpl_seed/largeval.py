"""Larger-n validation pool for the coverage/gate claim (D32 -> D33). The n=12 subset (6 rooms,
2 panos each) is under-powered. This pool = 8 DIVERSE rooms (office/hallway/lounge/conference/WC)
with ALL their in-frame panos = 32 panos (2.7x n), a generalization test beyond the original 6
offices. In-frame = camera XY inside the room cloud's XY bbox (D17 unlocalizable panos excluded).
Reuses the harness GT loader + subset's pano-resolver."""
import os, sys
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from eval import s3dis_gt
from experiments.fgpl_seed.subset import _resolve_pano

POOL_ROOMS = ["office_3", "office_5", "office_7", "office_8",
              "hallway_1", "lounge_1", "conferenceRoom_1", "WC_1"]


def _xy_bbox(cloud_txt):
    xmn = ymn = 1e18; xmx = ymx = -1e18
    with open(cloud_txt) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            xmn = min(xmn, x); xmx = max(xmx, x); ymn = min(ymn, y); ymx = max(ymx, y)
    return xmn, xmx, ymn, ymx


def build_pool():
    cfg = s3dis_gt.load_config(None)
    a = cfg["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", cfg)
    by_room = {}
    for u, v in gt.items():
        by_room.setdefault(v["room"], []).append((u, v["location"]))
    rows = []
    for room in POOL_ROOMS:
        cloud = os.path.join(a["rooms_dir"], room, room + ".txt")
        xmn, xmx, ymn, ymx = _xy_bbox(cloud)
        for u, loc in sorted(by_room.get(room, [])):
            if not (xmn <= loc[0] <= xmx and ymn <= loc[1] <= ymx):
                continue                                    # out-of-frame, D17
            jpg = _resolve_pano(u, a["pano_rgb_dir"])
            if jpg:
                rows.append({"pano_name": u, "uuid": u, "room": room,
                             "pano_jpg": jpg, "cloud_txt": cloud})
    return rows


def pool_clouds():
    cfg = s3dis_gt.load_config(None)
    rd = cfg["s3dis"]["Area_3"]["rooms_dir"]
    return {room: os.path.join(rd, room, room + ".txt") for room in POOL_ROOMS}


if __name__ == "__main__":
    rows = build_pool()
    from collections import Counter
    c = Counter(r["room"] for r in rows)
    print(f"pool: {len(rows)} in-frame panos, {len(POOL_ROOMS)} rooms")
    for room in POOL_ROOMS:
        print(f"  {room:18s} {c[room]}")
