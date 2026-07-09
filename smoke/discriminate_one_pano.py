"""M0+ discrimination probe (validates DECISIONS D13, the project's top open risk).

M0 (reproduce_cpo_one_room.py) proved CPO can *self*-localize a real pano to its
real room with near-GT pose. But the method's actual job is DISCRIMINATION: does the
CORRECT room score a lower CPO loss than OTHER candidate rooms — especially the
near-identical offices where the thesis' shape matching fails?

This runs ONE real query pano against a set of candidate room clouds and ranks them by
CPO loss. PASS = the query's true room has the minimum loss (i.e. content, not shape,
picks the right room). The synthetic box-room test could not show this (flat walls have
no texture); real rooms do.

GT (the query's true room + camera_location) is used for DEV VALIDATION ONLY; never
imported by src/panopin/* (fairness, D5).

Usage:
  conda run -n panopin python smoke/discriminate_one_pano.py [--room office_3] \
      [--candidates office_1,office_2,office_6,office_9,hallway_1] \
      [--sample-rate 30] [--top-k 1] [--num-iter 20] [--seed 0]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from panopin.cpo_config import load_cfg          # noqa: E402
from panopin.cpo_adapter import localize_pair    # noqa: E402
import s3dis_gt                                   # noqa: E402


def resolve_pano(uuid, rgb_dir):
    for p in glob.glob(os.path.join(rgb_dir, "*.png")):
        parts = os.path.basename(p).split("_")
        if len(parts) >= 2 and parts[1] == uuid:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="Area_3")
    ap.add_argument("--room", default="office_3", help="query pano's true room")
    ap.add_argument("--candidates", default="office_1,office_2,office_6,office_9,hallway_1",
                    help="comma-separated OTHER rooms to test against (the true room is added)")
    ap.add_argument("--sample-rate", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--num-iter", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gtcfg = s3dis_gt.load_config(None)
    acfg = gtcfg["s3dis"][args.area]
    rooms_dir = acfg["rooms_dir"]
    gt = s3dis_gt.load_gt(args.area, gtcfg)

    uuid = next((u for u, g in gt.items() if g["room"] == args.room), None)
    if uuid is None:
        sys.exit(f"no GT pano for room={args.room}")
    t_gt = np.asarray(gt[uuid]["location"], float)
    pano = resolve_pano(uuid, acfg["pano_rgb_dir"])
    if pano is None:
        sys.exit(f"no rgb pano for uuid={uuid}")

    # candidate set = true room + the requested others (dedup, keep order, true first)
    others = [r.strip() for r in args.candidates.split(",") if r.strip() and r.strip() != args.room]
    candidates = [args.room] + others

    print(f"query room  : {args.room}   (pano uuid {uuid})")
    print(f"candidates  : {candidates}")
    print(f"settings    : sample_rate={args.sample_rate} top_k={args.top_k} "
          f"num_iter={args.num_iter} seed={args.seed}")
    print(f"{'room':22s} {'loss':>9s} {'t_err(m)':>9s} {'sec':>6s}")

    cfg = load_cfg(sample_rate=args.sample_rate, top_k_candidate=args.top_k,
                   num_iter=args.num_iter)

    results = []
    for room in candidates:
        cloud = os.path.join(rooms_dir, room, room + ".txt")
        if not os.path.exists(cloud):
            print(f"{room:22s} {'MISSING':>9s}")
            continue
        np.random.seed(args.seed)   # reset before each room so subsampling is comparable
        t0 = time.time()
        t_est, R, loss = localize_pair(cfg, pano, cloud)
        dt = time.time() - t0
        terr = float(np.linalg.norm(t_est - t_gt))   # only meaningful for the true room
        results.append((room, loss, terr, dt))
        print(f"{room:22s} {loss:9.4f} {terr:9.3f} {dt:6.1f}")

    results.sort(key=lambda r: r[1])
    winner = results[0][0]
    true_loss = next(l for r, l, _, _ in results if r == args.room)
    margin = results[0][1] - (results[1][1] if len(results) > 1 else results[0][1])
    print(f"\nranked by loss (low=best): {[r for r, *_ in results]}")
    print(f"winner={winner}  true={args.room}  true_loss={true_loss:.4f}")
    ok = winner == args.room
    if ok and len(results) > 1:
        runner_up = results[1][1]
        print(f"margin to runner-up: {runner_up - true_loss:+.4f} "
              f"({100*(runner_up - true_loss)/true_loss:+.1f}%)")
    print(f"DISCRIMINATION: {'PASS (content picks the right room)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
