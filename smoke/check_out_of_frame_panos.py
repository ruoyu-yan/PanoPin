"""Follow-up to check_frame_alignment.py: for each pano whose GT camera is OUTSIDE
its own room's cloud bbox, is the camera nonetheless INSIDE some OTHER Area_3 room's
bbox (i.e. inside the building, at a boundary/adjacent segment) or outside ALL rooms
(genuinely off the point cloud)? This decides whether the 9 out-of-frame panos are
recoverable vs truly unmatchable. GT is dev-only.

Usage: conda run -n panopin python smoke/check_out_of_frame_panos.py [--stride 40]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import s3dis_gt  # noqa: E402

TOL = 0.10


def cloud_bbox(path, stride):
    mn = np.array([np.inf] * 3)
    mx = np.array([-np.inf] * 3)
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i % stride:
                continue
            p = line.split()
            v = np.array([float(p[0]), float(p[1]), float(p[2])])
            mn = np.minimum(mn, v)
            mx = np.maximum(mx, v)
    return mn, mx


def outside_dist(loc, mn, mx):
    """Max per-axis distance the point lies outside [mn,mx]; 0 if inside."""
    return float(np.maximum(np.maximum(mn - loc, loc - mx), 0.0).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="Area_3")
    ap.add_argument("--stride", type=int, default=40)
    args = ap.parse_args()

    cfg = s3dis_gt.load_config(None)
    rooms_dir = cfg["s3dis"][args.area]["rooms_dir"]
    gt = s3dis_gt.load_gt(args.area, cfg)
    all_rooms = s3dis_gt.candidate_rooms(args.area, cfg)

    # bbox for every room (once)
    bbox = {}
    for r in all_rooms:
        c = os.path.join(rooms_dir, r, r + ".txt")
        if os.path.exists(c):
            bbox[r] = cloud_bbox(c, args.stride)

    # find out-of-frame panos (camera outside its OWN room bbox)
    oof = []
    for u, g in gt.items():
        room = g["room"]
        if room not in bbox:
            continue
        loc = np.asarray(g["location"], float)
        d_own = outside_dist(loc, *bbox[room])
        if d_own > TOL:
            oof.append((u, room, loc, d_own))

    print(f"{len(oof)} out-of-frame panos (camera > {TOL} m outside its OWN room bbox):\n")
    for u, room, loc, d_own in sorted(oof, key=lambda x: -x[3]):
        # which rooms' bbox contains this camera? nearest room otherwise
        containing = [r for r, (mn, mx) in bbox.items() if outside_dist(loc, mn, mx) <= TOL]
        dists = sorted(((outside_dist(loc, mn, mx), r) for r, (mn, mx) in bbox.items()))
        nearest_d, nearest_r = dists[0]
        print(f"  {room:12s} uuid {u[:10]}  own_out={d_own:.2f} m  loc={np.round(loc,2).tolist()}")
        print(f"      inside these room bboxes: {containing if containing else '(none)'}")
        print(f"      nearest room bbox: {nearest_r} at {nearest_d:.2f} m")
    # summary
    inside_building = sum(1 for u, room, loc, _ in oof
                          if any(outside_dist(loc, mn, mx) <= TOL for mn, mx in bbox.values()))
    print(f"\nof {len(oof)} out-of-frame panos: {inside_building} fall inside SOME room's bbox "
          f"(in the building, boundary/adjacent), {len(oof) - inside_building} are outside ALL rooms.")


if __name__ == "__main__":
    main()
