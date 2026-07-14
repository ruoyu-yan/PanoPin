"""Phase-1 diagnostic (systematic-debugging): DERIVE the rotation convention that makes
residuals_at_pose agree with the S3DIS GT pose. At the TRUE GT position, sweep candidate
rotations (natural compositions of GT R_cw with the equirect signed-perm C from D25); the
candidate whose residual lands near the CPO known-good floor is the correct convention.
Not a fix — evidence only. Run in panopin-gpu."""
import os, sys, json
import numpy as np
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from eval import s3dis_gt
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import residuals_at_pose

C = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


def candidates(R_cw):
    Rwc = R_cw.T
    return {
        "R_cw.T (framecheck)": Rwc,
        "C@R_cw.T":            C @ Rwc,
        "C.T@R_cw.T":          C.T @ Rwc,
        "R_cw":                R_cw,
        "C@R_cw":              C @ R_cw,
        "C.T@R_cw":            C.T @ R_cw,
        "(C@C)@R_cw.T":        C @ C @ Rwc,
        "R_cw.T@C":            Rwc @ C,
        "R_cw.T@C.T":          Rwc @ C.T,
        "C@R_cw.T@C.T":        C @ Rwc @ C.T,
    }


def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    cpo = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    clouds = {r["room"]: r["cloud_txt"] for r in rows}
    good = sorted([r for r in rows if cpo[r["pano_name"]]["room"] == r["room"]],
                  key=lambda r: cpo[r["pano_name"]]["loss"])[:3]
    for r in good:
        name, room = r["pano_name"], r["room"]
        loc = gt[name]["location"]
        R_cw = np.array(gt[name]["R_cw"], float)
        tc, Rc = cpo[name]["t"], np.array(cpo[name]["R"], float)
        floor = float(residuals_at_pose(cfg, r["pano_jpg"], clouds[room], tc, Rc).mean())
        print(f"\n{name} {room} cpo_loss={cpo[name]['loss']:.3f}  CPO-own-pose floor={floor:.4f}", flush=True)
        best = None
        for label, R in candidates(R_cw).items():
            res = residuals_at_pose(cfg, r["pano_jpg"], clouds[room], loc, R.tolist())
            m = float(res.mean()) if len(res) else float("nan")
            print(f"    {label:22s} n={len(res):6d} residual={m:.4f}", flush=True)
            if len(res) and (best is None or m < best[1]):
                best = (label, m)
        print(f"    -> lowest: {best[0]} = {best[1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
