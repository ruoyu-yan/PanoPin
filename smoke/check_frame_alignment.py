"""Frame-validity diagnostic: for every Area_3 pano, is its GT camera_location
actually INSIDE its room's point-cloud bounding box?

CPO seeds candidate translations from the cloud's spatial extent, so a camera that
sits outside the cloud can never be localized (office_9 was such a case: 5.8 m self-
error). The config claims "Original cloud <-> pano frame = identity (~5 mm)" — this
checks whether that holds room-by-room, which is a prerequisite for interpreting any
CPO result. GT is dev-only (never in src/panopin/*).

Usage: conda run -n panopin python smoke/check_frame_alignment.py [--area Area_3] [--stride 40]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import s3dis_gt  # noqa: E402


def cloud_bbox(path, stride):
    mn = np.array([np.inf, np.inf, np.inf])
    mx = np.array([-np.inf, -np.inf, -np.inf])
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i % stride:
                continue
            p = line.split()
            xyz = np.array([float(p[0]), float(p[1]), float(p[2])])
            mn = np.minimum(mn, xyz)
            mx = np.maximum(mx, xyz)
    return mn, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="Area_3")
    ap.add_argument("--stride", type=int, default=40)
    args = ap.parse_args()

    cfg = s3dis_gt.load_config(None)
    rooms_dir = cfg["s3dis"][args.area]["rooms_dir"]
    gt = s3dis_gt.load_gt(args.area, cfg)

    # group pano uuids by room
    by_room = {}
    for u, g in gt.items():
        by_room.setdefault(g["room"], []).append(u)

    bbox_cache = {}
    rows = []
    for room in sorted(by_room):
        cloud = os.path.join(rooms_dir, room, room + ".txt")
        if not os.path.exists(cloud):
            for u in by_room[room]:
                rows.append((room, u, None, None))
            continue
        if room not in bbox_cache:
            bbox_cache[room] = cloud_bbox(cloud, args.stride)
        mn, mx = bbox_cache[room]
        for u in by_room[room]:
            loc = np.asarray(gt[u]["location"], float)
            # signed outside distance per axis (0 if inside); max over axes
            outside = np.maximum(np.maximum(mn - loc, loc - mx), 0.0)
            rows.append((room, u, float(outside.max()), loc.tolist()))

    print(f"{'room':20s} {'n':>2s} {'max_out(m)':>10s}  {'worst_uuid':>12s}")
    n_inside = n_total = 0
    bad_rooms = []
    per_room = {}
    for room, u, out, loc in rows:
        if out is None:
            continue
        per_room.setdefault(room, []).append((u, out))
    TOL = 0.10  # a camera up to 10 cm outside the bbox is tolerated (wall grazing / sampling)
    for room in sorted(per_room):
        entries = per_room[room]
        outs = [o for _, o in entries]
        worst_u, worst = max(entries, key=lambda e: e[1])
        n = len(entries)
        inside = sum(1 for o in outs if o <= TOL)
        n_inside += inside
        n_total += n
        flag = "" if worst <= TOL else "  <-- OUT OF FRAME"
        print(f"{room:20s} {n:2d} {worst:10.3f}  {worst_u[:12]}{flag}")
        if worst > TOL:
            bad_rooms.append(room)

    print(f"\npanos with camera inside cloud (<= {TOL} m): {n_inside}/{n_total} "
          f"({100*n_inside/max(1,n_total):.0f}%)")
    print(f"rooms with an out-of-frame pano: {bad_rooms}")


if __name__ == "__main__":
    main()
