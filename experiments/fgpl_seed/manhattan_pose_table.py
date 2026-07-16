"""Per-pano estimated vs GT pose table for the strictly-Manhattan round-trip (2026-07-16).

Lists, for every pano: FGPL's estimated pose, the S3DIS GT pose, and the difference.

Conventions: translation is raw S3DIS metres. FGPL emits `Rp = C @ R_wc`; we convert to
camera->world (`Rp.T @ C`, D25) to match GT's `R_cw`. Rotation is summarised as yaw about Z
(atan2(R[1][0], R[0][0])) — extracted IDENTICALLY for estimate and GT, so the two are
comparable; the reported difference is the true geodesic angle, not a yaw subtraction.

    conda run -n panopin python -m experiments.fgpl_seed.manhattan_pose_table [arm]
"""
import os
import sys
import json
import math

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from experiments.fgpl_seed import manhattan, paths, roundtrip
from eval import s3dis_gt, metrics


def _yaw(R):
    return math.degrees(math.atan2(R[1][0], R[0][0]))


def _v(t):
    return f"({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})"


def main(arm="manhattan_export"):
    rows = manhattan.build_pool()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    res = json.load(open(paths.WORK / "results" / f"{arm}.json"))
    seed_room, true_room = res["seed_room"], res["true_room"]

    recs = []
    for r in rows:
        u = r["pano_name"]
        cp = paths.WORK / "poses" / arm / u / "camera_pose.json"
        if not cp.exists():
            continue
        d = json.load(open(cp))
        est_t = d["translation"]
        est_R = roundtrip._fgpl_rot_to_cw(d["rotation"])       # -> camera->world, like GT
        gt_t, gt_R = gt[u]["location"], gt[u]["R_cw"]
        dt = math.dist(est_t, gt_t)
        drot = metrics.rotation_errors({u: est_R}, {u: gt_R})["per_uuid"][u]
        recs.append({"u": u, "room": true_room[u], "seed": seed_room.get(u, "-"),
                     "est_t": est_t, "gt_t": gt_t, "dt": dt,
                     "est_yaw": _yaw(est_R), "gt_yaw": _yaw(gt_R), "drot": drot})
    recs.sort(key=lambda x: x["dt"])

    L = []

    def out(s=""):
        L.append(s)
        print(s)

    out(f"# Per-pano estimated vs GT pose — `{arm}` (strictly-Manhattan, "
        f"{len(manhattan.POOL_ROOMS)} rooms, {len(recs)} panos)\n")
    out("Translation in raw S3DIS metres. Yaw about Z, extracted identically from the "
        "camera->world rotation for estimate and GT. **Δrot is the geodesic angle** between the "
        "full rotations (not a yaw subtraction). Sorted by translation error.\n")
    out("| # | pano | room | seed room | estimated (x,y,z) | GT (x,y,z) | Δt (m) | est yaw | "
        "GT yaw | Δrot (°) |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(recs, 1):
        out(f"| {i} | `{r['u'][:8]}` | {r['room']} | "
            f"{r['seed'] if r['seed'] == r['room'] else '**' + r['seed'] + '**'} | "
            f"{_v(r['est_t'])} | {_v(r['gt_t'])} | {r['dt']:.3f} | {r['est_yaw']:+.1f}° | "
            f"{r['gt_yaw']:+.1f}° | {r['drot']:.1f} |")
    out("\nBold seed room = PanoPin assigned the wrong room.")

    lock = [r for r in recs if r["drot"] <= 45]
    ali = [r for r in recs if r["drot"] > 45]
    out(f"\n**Rotation locked ({len(lock)}/{len(recs)}):** Δt "
        f"{min(r['dt'] for r in lock):.3f}–{max(r['dt'] for r in lock):.3f} m, Δrot "
        f"{min(r['drot'] for r in lock):.1f}–{max(r['drot'] for r in lock):.1f}°.")
    out(f"**Rotation aliased ({len(ali)}/{len(recs)}):** Δt "
        f"{min(r['dt'] for r in ali):.3f}–{max(r['dt'] for r in ali):.3f} m, Δrot "
        f"{min(r['drot'] for r in ali):.1f}–{max(r['drot'] for r in ali):.1f}°.")

    p = paths.HERE / f"MANHATTAN_POSE_TABLE_{arm}.md"
    p.write_text("\n".join(L) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "manhattan_export")
