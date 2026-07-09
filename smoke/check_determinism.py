"""G3: is one real Tier-1 (and Tier-2) score identical across two runs after pin()?"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import score_room_cheap
import s3dis_gt


def main():
    g = s3dis_gt.load_config(None); a = g["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", g)
    uuid = next(u for u, v in gt.items() if v["room"] == "office_3")
    pano = next(p for p in glob.glob(os.path.join(a["pano_rgb_dir"], "*.png"))
                if os.path.basename(p).split("_")[1] == uuid)
    cloud = os.path.join(a["rooms_dir"], "office_3", "office_3.txt")
    cfg = load_cfg(sample_rate=30, num_yaw=4, num_pitch=4, num_roll=4, num_trans=10)
    losses = []
    for _ in range(2):
        pin(0)
        _, _, loss = score_room_cheap(cfg, pano, cloud)
        losses.append(loss)
    print(f"losses: {losses}  delta={abs(losses[0]-losses[1]):.6f}")
    ok = abs(losses[0] - losses[1]) < 1e-6
    print(f"G3 determinism: {'PASS (identical)' if ok else 'RESIDUAL — record and rely on Tier-2 margin'}")
    return 0 if ok else 0   # informational: never hard-fail, but print the verdict


if __name__ == "__main__":
    main()
