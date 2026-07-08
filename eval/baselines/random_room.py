"""Random-room baseline: assigns each pano a uniformly random candidate room.
Produces a predictions.json so the harness loop runs end-to-end (a floor to beat).

Usage:
  python eval/baselines/random_room.py [--area Area_3] [--out runs/random_room.json] [--seed 42]
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s3dis_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="Area_3")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = s3dis_gt.load_config(args.config)
    gt = s3dis_gt.load_gt(args.area, cfg)          # used ONLY for the pano UUID list
    rooms = s3dis_gt.candidate_rooms(args.area, cfg)
    rng = random.Random(args.seed)

    preds = {u: {"room": rng.choice(rooms)} for u in gt}
    out = args.out or str(s3dis_gt.repo_root() / "runs" / "random_room.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"area": args.area, "method": "random_room_baseline",
                   "predictions": preds}, f, indent=2)
    print(f"wrote {out}  ({len(preds)} panos, {len(rooms)} candidate rooms, "
          f"chance ~= {1 / len(rooms):.3f})")


if __name__ == "__main__":
    main()
