"""BLOCKING R1 pre-check: does residuals_at_pose + the frame conventions give a LOW
color residual at the GT pose and a HIGHER one at a wrong (sibling-room) pose? If this
fails, the color scorer or the rotation convention is wrong and NO fusion number is valid.
Run in panopin-gpu. -> work/seeds/framecheck.json.

Rotation convention (DERIVED empirically 2026-07-14 via fusion_convention_probe.py, not
guessed): residuals_at_pose expects the EQUIRECT-convention rotation C @ R_wc (the same
convention FGPL emits as Rp), NOT the raw S3DIS world->camera R_wc. For a GT pose, R_wc =
R_cw.T, so feed R = C @ R_cw.T. The probe showed C @ R_cw.T lands at the CPO genuine-lock
floor (~0.12-0.18) while every other composition sits at ~0.40-0.46."""
import os, sys, json, statistics
import numpy as np
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, seed_and_config as sc, paths
from eval import s3dis_gt
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import residuals_at_pose

C = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # equirect signed-perm (D25)


def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    clouds = {r["room"]: r["cloud_txt"] for r in rows}
    out = {}
    for r in rows:
        name, room = r["pano_name"], r["room"]
        loc = gt[name]["location"]
        R_cw = np.array(gt[name]["R_cw"], float)
        R_feed = C @ R_cw.T                             # equirect-convention rotation (see docstring)
        gt_res = float(residuals_at_pose(cfg, r["pano_jpg"], clouds[room], loc, R_feed.tolist()).mean())
        sib = sc.SIBLING[room]
        wrong_res = float(residuals_at_pose(cfg, r["pano_jpg"], clouds[sib], loc, R_feed.tolist()).mean())
        out[name] = {"gt_color": gt_res, "wrong_color": wrong_res, "room": room, "sibling": sib}
        print(f"{name} {room:11s} gt={gt_res:.4f}  wrong({sib})={wrong_res:.4f}  "
              f"{'OK' if gt_res < wrong_res else 'X'}", flush=True)
    p = paths.subdir("seeds") / "framecheck.json"
    json.dump(out, open(p, "w"), indent=2)
    n_ok = sum(1 for v in out.values() if v["gt_color"] < v["wrong_color"])
    med_gt = statistics.median(v["gt_color"] for v in out.values())
    print(f"\nGT<wrong on {n_ok}/{len(out)} panos; median GT residual={med_gt:.4f}. Wrote {p}")


if __name__ == "__main__":
    main()
