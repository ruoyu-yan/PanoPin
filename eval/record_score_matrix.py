"""Record one draw of PanoPin's seed-stage score matrix for a scene, as a test fixture.

    cd /home/ruoyu/PanoPin && conda run -n panopin-gpu python eval/record_score_matrix.py \
        --scene Area_2_manhattan7 \
        --panos  /home/ruoyu/Point_360/data/full_runs/Area_2_manhattan7/stage0_localization/panos.json \
        --clouds /home/ruoyu/Point_360/data/full_runs/Area_2_manhattan7/stage0_localization/roomseg/clouds.json \
        --gt-poses /home/ruoyu/Point_360/data/full_runs/Area_2_manhattan7/stage0_localization/pose_gt_reference.json \
        --out tests/fixtures/score_matrix/Area_2_manhattan7_draw1.json

The GT poses label the fixture with each pano's TRUE segment (the segment with a point nearest
the GT station); the scores themselves read panos + clouds only. The seed stage is
nondeterministic on the GPU, so every run is a new draw -- that is the point of recording
several."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from panopin import seed                       # noqa: E402
from panopin.roomseg.files import read_cloud   # noqa: E402


def truth_segments(gt_poses, clouds, stride=20):
    pts = {room: read_cloud(path)[0][::stride, :2] for room, path in clouds.items()}
    out = {}
    for pano, pose in gt_poses.items():
        t = np.asarray(pose["t"][:2], dtype=float)
        out[pano] = min(pts, key=lambda r: float(np.min(np.linalg.norm(pts[r] - t, axis=1))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--panos", required=True, type=Path)
    ap.add_argument("--clouds", required=True, type=Path)
    ap.add_argument("--gt-poses", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    panos = json.loads(args.panos.read_text())
    clouds = json.loads(args.clouds.read_text())
    gt = json.loads(args.gt_poses.read_text())
    missing = sorted(set(panos) - set(gt))
    if missing:
        raise SystemExit(f"panos without a GT pose: {missing}")

    t0 = time.time()
    score_matrix, _poses = seed.localize_and_score(panos, clouds, seed.load_cfg(sample_rate=30))
    doc = {"scene": args.scene, "recorded": time.strftime("%Y-%m-%d %H:%M"),
           "seconds": round(time.time() - t0, 1), "room_order": list(clouds),
           "score_matrix": score_matrix,
           "truth_segment": {p: s for p, s in truth_segments(gt, clouds).items() if p in panos}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {args.out} ({doc['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
