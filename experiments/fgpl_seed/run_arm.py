"""Run the FGPL estimator for one arm config; collect per-pano camera_pose.json.
camera_pose.json is written (line 524) BEFORE the composite viz (line 581), so a
viz-stage crash still leaves per-pano poses readable — we tolerate a non-zero exit
and read whatever poses were produced."""
import json, subprocess, os
from experiments.fgpl_seed import paths

def run_arm(config_path, rows):
    with open(config_path) as f:
        cfg = json.load(f)
    out_base = cfg["output_dir"]
    env = dict(os.environ)
    # Estimator runs in paths.ESTIMATOR_ENV (panopin-gpu, Ada-native cu118). The XDF
    # search is CPU-bound numpy regardless (~330-390s/pano), so GPU is only ~1.2x.
    cmd = ["conda", "run", "--no-capture-output", "-n", paths.ESTIMATOR_ENV,
           "python", str(paths.ESTIMATOR), "--config", str(config_path)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(paths.FGPL_ROOT), env=env, check=False)  # tolerate viz crash
    poses = {}
    for r in rows:
        cp = os.path.join(out_base, r["pano_name"], "camera_pose.json")
        if os.path.exists(cp):
            with open(cp) as f:
                d = json.load(f)
            poses[r["pano_name"]] = {"translation": d["translation"], "rotation": d["rotation"]}
        else:
            poses[r["pano_name"]] = None
    return poses


def run_arm_pool(config_path, rows):
    """Like run_arm but also collect each pano's dumped candidate pool (candidates.json).
    Returns {pano_name: {"best": {translation,rotation}|None, "candidates": [ ... ]}}."""
    import json, subprocess, os
    from experiments.fgpl_seed import paths
    with open(config_path) as f:
        cfg = json.load(f)
    out_base = cfg["output_dir"]
    cmd = ["conda", "run", "--no-capture-output", "-n", paths.ESTIMATOR_ENV,
           "python", str(paths.ESTIMATOR), "--config", str(config_path)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(paths.FGPL_ROOT), env=dict(os.environ), check=False)
    res = {}
    for r in rows:
        d = os.path.join(out_base, r["pano_name"])
        cp, cand = os.path.join(d, "camera_pose.json"), os.path.join(d, "candidates.json")
        best = None
        if os.path.exists(cp):
            j = json.load(open(cp)); best = {"translation": j["translation"], "rotation": j["rotation"]}
        cands = json.load(open(cand)) if os.path.exists(cand) else []
        res[r["pano_name"]] = {"best": best, "candidates": cands}
    return res
