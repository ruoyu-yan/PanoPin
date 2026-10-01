"""Bar (b) end-to-end (spec §8.3): Stage 0 = PanoPin seed + upright-prior FGPL on one scene, run FRESH
in two arms by the same command, differing ONLY in --clouds-json:
  gt   = the S3DIS room clouds (the scene's stage0_localization/clouds.json)
  pred = this branch's segments (runs/roomseg/<scene>/clouds.json)
Pass: pred room accuracy >= gt's, pred translation median <= gt's + 0.05 m, and every e2e pano
contained in its room's matched segment (runs/roomseg_eval/<scene>/score.json).

Run (system python3; run_localization.py shells into the conda envs):
  python3 experiments/roomseg/two_arm.py --scene Area_3_manhattan4
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
P360 = Path("/home/ruoyu/Point_360")
MARGIN_M = 0.05


def run_arm(scene, stage0, clouds_json, out):
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", str(P360 / "run_localization.py"), "--pose-source", "panopin", "--scene", scene,
           "--output", str(out / "pose_contract.json"), "--sidecar", str(out / "pose_contract.sidecar.json"),
           "--panos-json", str(stage0 / "panos.json"), "--clouds-json", str(clouds_json),
           "--metadata", str(stage0 / "metadata.json"),
           "--line-map", str(stage0 / "linemap" / "3d_line_map.pkl"),
           "--cloud-ply", str(stage0 / "cloud" / f"{scene}.ply"), "--density-png", str(stage0 / "density.png"),
           "--features-dir", str(stage0 / "features"), "--pano-dir", str(stage0 / "panos_fgpl"),
           "--work-dir", str(out / "_work"), "--tau", "0.10",
           "--fgpl-root", "/home/ruoyu/scan2measure-webframework", "--panopin-root", str(REPO)]
    with open(out / "run.log", "w") as log:
        subprocess.run(cmd, check=True, cwd=P360, stdout=log, stderr=subprocess.STDOUT)


def evaluate(scene, key_map, out, gt_pose, room_of_label):
    """Per pano: the room PanoPin chose (via room_of_label) and the FGPL translation error."""
    align = json.loads((out / "_work" / scene / "demo6_alignment.json").read_text())
    chosen = {m["pano_name"]: m["room_label"] for m in align["matches"]}
    poses = json.loads((out / "pose_contract.json").read_text())
    rows = {}
    for uuid, key in key_map.items():
        room = key.split("_", 2)[2]
        label = chosen.get(uuid)
        err = (float(np.linalg.norm(np.asarray(poses[uuid]["t"]) - np.asarray(gt_pose[key]["t"])))
               if uuid in poses else float("inf"))
        rows[uuid] = {"room": room, "label": label,
                      "room_ok": label is not None and room_of_label(label) == room,
                      "trans_err_m": round(err, 3)}
    errs = [r["trans_err_m"] for r in rows.values()]
    return {"per_pano": rows, "room_acc": sum(r["room_ok"] for r in rows.values()) / len(rows),
            "trans_median_m": float(np.median(errs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    a = ap.parse_args()
    cfg = json.loads((REPO / "config" / "roomseg_scenes.json").read_text())["scenes"][a.scene]
    stage0 = Path(cfg["e2e_stage0"])
    key_map = json.loads((stage0 / "key_map.json").read_text())
    gt_pose = json.loads(Path(cfg["pose_json"]).read_text())
    score = json.loads((REPO / "runs" / "roomseg_eval" / a.scene / "score.json").read_text())
    seg_room = {m["segment"]: room for room, m in score["matching"].items() if m}
    out = REPO / "runs" / "roomseg_e2e" / a.scene
    run_arm(a.scene, stage0, stage0 / "clouds.json", out / "gt")
    run_arm(a.scene, stage0, REPO / "runs" / "roomseg" / a.scene / "clouds.json", out / "pred")
    gt = evaluate(a.scene, key_map, out / "gt", gt_pose, lambda lab: lab)
    pred = evaluate(a.scene, key_map, out / "pred", gt_pose, seg_room.get)
    contained = {u: bool(score["containment"].get(k, {}).get("pred_inside")) for u, k in key_map.items()}
    result = {"scene": a.scene, "gt": gt, "pred": pred, "contained": contained, "margin_m": MARGIN_M,
              "pass": {"room_acc": pred["room_acc"] >= gt["room_acc"],
                       "trans_median": pred["trans_median_m"] <= gt["trans_median_m"] + MARGIN_M,
                       "containment": all(contained.values())}}
    (out / "two_arm.json").write_text(json.dumps(result, indent=1))
    print(f"{a.scene}: room acc gt {gt['room_acc']:.2f} pred {pred['room_acc']:.2f} | "
          f"trans median gt {gt['trans_median_m']:.3f} pred {pred['trans_median_m']:.3f} m | "
          f"contained {sum(contained.values())}/{len(contained)} | pass {result['pass']}")


if __name__ == "__main__":
    main()
