"""Score a PanoPin predictions file against S3DIS GT.

Usage:
  python eval/score.py --predictions runs/preds.json [--area Area_3] [--config PATH] [--out runs/report.json]

Predictions schema: see eval/PREDICTIONS_SCHEMA.md.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
import s3dis_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--area", default="Area_3")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = s3dis_gt.load_config(args.config)
    gt = s3dis_gt.load_gt(args.area, cfg)
    gtr = s3dis_gt.gt_rooms(gt)
    gl = s3dis_gt.gt_locations(gt)
    gR = {u: v["R_cw"] for u, v in gt.items()}

    with open(args.predictions, "r", encoding="utf-8") as f:
        pred = json.load(f)
    P = pred.get("predictions", {})
    pred_rooms = {u: v.get("room") for u, v in P.items() if v.get("room") is not None}
    pred_t = {u: v["coarse_pose"]["t"] for u, v in P.items()
              if v.get("coarse_pose") and v["coarse_pose"].get("t") is not None}
    pred_R = {u: v["coarse_pose"]["R"] for u, v in P.items()
              if v.get("coarse_pose") and v["coarse_pose"].get("R") is not None}

    acc = metrics.room_accuracy(pred_rooms, gtr)
    te = metrics.translation_errors(pred_t, gl)
    ro = metrics.rotation_errors(pred_R, gR)

    report = {
        "area": args.area,
        "method": pred.get("method", "unknown"),
        "predictions_file": args.predictions,
        "room_assignment": {k: acc[k] for k in ("n_scored", "n_missing", "correct", "accuracy")},
        "coarse_translation_m": {k: te[k] for k in ("n", "mean", "median", "max")},
        "coarse_rotation_deg": {k: ro[k] for k in ("n", "mean", "median", "max")},
    }
    print(json.dumps(report, indent=2))

    if args.out:
        detail = dict(report)
        detail["room_per_room"] = acc["per_room"]
        detail["room_missing_uuids"] = acc["missing_uuids"]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
