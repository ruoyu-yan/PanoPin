"""M0 evidence (plan Task 4): run localize_pair on ONE real Area_3 room + its GT
pano and report translation error vs GT.

This directly tests the project's top open risk (DECISIONS D13): can CPO localize a
REAL pano to its REAL room with low loss + near-GT pose? The synthetic box-room test
could not validate this (flat walls give no texture for CPO's colour-histogram pose
search). Real S3DIS rooms have texture, so this is the honest test.

GT (room + camera_location) is used here for DEV VALIDATION ONLY. Never imported by
src/panopin/* (fairness, D5). Room clouds are on ext4; panos on the slow /mnt/d bridge.

Usage:
  conda run -n panopin python smoke/reproduce_cpo_one_room.py [--room office_3] \
      [--sample-rate 15] [--top-k 2] [--num-iter 100]
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
    """uuid -> rgb path (filename = camera_<uuid>_<room>_..._rgb.png)."""
    for p in glob.glob(os.path.join(rgb_dir, "*.png")):
        parts = os.path.basename(p).split("_")
        if len(parts) >= 2 and parts[1] == uuid:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="Area_3")
    ap.add_argument("--room", default="office_3")
    ap.add_argument("--sample-rate", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--num-iter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # read_txt_pcd subsamples via unseeded np.random.permutation; seed it to pin
    # that one source. NOTE: this does NOT make CPO bit-reproducible — torch CPU
    # ops (score-map scatter / Adam) still vary run-to-run (~+/-0.02 loss, cm-level
    # pose). Full determinism (D1) is a deferred polish item, tracked in DECISIONS.
    np.random.seed(args.seed)

    gtcfg = s3dis_gt.load_config(None)
    acfg = gtcfg["s3dis"][args.area]
    gt = s3dis_gt.load_gt(args.area, gtcfg)

    # pick a pano whose GT room == args.room
    uuid = next((u for u, g in gt.items() if g["room"] == args.room), None)
    if uuid is None:
        sys.exit(f"no GT pano for room={args.room} in {args.area}")

    room = gt[uuid]["room"]
    t_gt = np.asarray(gt[uuid]["location"], float)
    cloud = os.path.join(acfg["rooms_dir"], room, room + ".txt")
    pano = resolve_pano(uuid, acfg["pano_rgb_dir"])
    if pano is None:
        sys.exit(f"no rgb pano file for uuid={uuid}")

    print(f"room        : {room}")
    print(f"uuid        : {uuid}")
    print(f"cloud       : {cloud}")
    print(f"pano        : {os.path.basename(pano)}")
    print(f"t_gt        : {t_gt}")
    print(f"sample_rate={args.sample_rate}  top_k_candidate={args.top_k}  num_iter={args.num_iter}")

    cfg = load_cfg(sample_rate=args.sample_rate, top_k_candidate=args.top_k,
                   num_iter=args.num_iter)

    t0 = time.time()
    t_est, R, loss = localize_pair(cfg, pano, cloud)
    dt = time.time() - t0

    err = float(np.linalg.norm(t_est - t_gt))
    print(f"t_est       : {t_est}")
    print(f"loss        : {loss:.4f}")
    print(f"trans error : {err:.3f} m")
    print(f"elapsed     : {dt:.1f} s")
    ok = err < 1.5
    print(f"M0 (<1.5 m) : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
