"""Task-5 real-data validation: run the two-tier select_room on a real Area_3 pano
over a real candidate room set and check it returns the true room.

The funnel LOGIC is unit-tested with mocks (tests/test_select_room.py); this is the
honest end-to-end check that the Tier-1 -> Tier-2 handoff works on real CPO output
(DECISIONS D15). GT (true room) is dev-only here; never imported by src/panopin/*.

Usage:
  conda run -n panopin python smoke/select_room_real.py [--room office_3] \
      [--candidates office_1,office_2,office_6,office_9,hallway_1] \
      [--sample-rate 30] [--top-k 2] [--seed 0]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from panopin.select_room import select_room  # noqa: E402
import s3dis_gt                               # noqa: E402


def resolve_pano(uuid, rgb_dir):
    for p in glob.glob(os.path.join(rgb_dir, "*.png")):
        parts = os.path.basename(p).split("_")
        if len(parts) >= 2 and parts[1] == uuid:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="Area_3")
    ap.add_argument("--room", default="office_3")
    ap.add_argument("--candidates", default="office_1,office_2,office_6,office_9,hallway_1")
    ap.add_argument("--sample-rate", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    gtcfg = s3dis_gt.load_config(None)
    acfg = gtcfg["s3dis"][args.area]
    rooms_dir = acfg["rooms_dir"]
    gt = s3dis_gt.load_gt(args.area, gtcfg)

    uuid = next((u for u, g in gt.items() if g["room"] == args.room), None)
    if uuid is None:
        sys.exit(f"no GT pano for room={args.room}")
    pano = resolve_pano(uuid, acfg["pano_rgb_dir"])
    if pano is None:
        sys.exit(f"no rgb pano for uuid={uuid}")

    others = [r.strip() for r in args.candidates.split(",") if r.strip() and r.strip() != args.room]
    names = [args.room] + others
    candidate_rooms = {r: os.path.join(rooms_dir, r, r + ".txt") for r in names}

    print(f"query room  : {args.room}   candidates: {names}")
    print(f"settings    : sample_rate={args.sample_rate} top_k={args.top_k} seed={args.seed}")
    t0 = time.time()
    res = select_room(pano, candidate_rooms, top_k=args.top_k, sample_rate=args.sample_rate)
    dt = time.time() - t0

    print(f"selected    : {res.room}  loss={res.loss:.4f}  t={res.t}")
    print(f"elapsed     : {dt:.1f} s")
    ok = res.room == args.room
    print(f"SELECT_ROOM : {'PASS (true room selected)' if ok else 'FAIL (selected ' + res.room + ')'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
