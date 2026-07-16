"""Meeting summary across all strictly-Manhattan arms (2026-07-16).

Four configurations on IDENTICAL inputs (same 22 panos, same 5-room Manhattan map, same
estimator); the only variable is how FGPL is seeded:

  manhattan_global        FGPL alone -- global mode, no Voronoi, seed never read (baseline)
  manhattan_export        PanoPin, one seed per pano (22 seeds)
  manhattan_oracle        GT position per pano (22 seeds) -- reference
  manhattan_anchored      PanoPin room-anchored, one seed per room (5 seeds) = DEPLOYMENT
  manhattan_aliased_solo  4 previously-aliased panos, identical seeds, 4 competing panos
                          (controlled test of the pano-crowding mechanism)

    conda run -n panopin python -m experiments.fgpl_seed.manhattan_summary
"""
import os
import sys
import json
import math
import statistics as st

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from experiments.fgpl_seed import manhattan, paths, roundtrip
from eval import s3dis_gt, metrics

# Current estimator (upright rotation prior ON) — the fair like-for-like comparison.
ARMS = [
    ("manhattan_global_upright", "FGPL alone (no PanoPin)"),
    ("manhattan_upright", "PanoPin, 22 seeds (1/pano)"),
    ("manhattan_oracle_upright", "GT seed, 22 seeds (reference)"),
    ("manhattan_anchored_upright", "PanoPin, 5 seeds (1/room) = DEPLOYMENT"),
]

# Same arms on the previous estimator (prior OFF), so the prior's contribution is visible
# without contaminating the seeding comparison.
ARMS_NOPRIOR = [
    ("manhattan_global", "FGPL alone (no PanoPin)"),
    ("manhattan_export", "PanoPin, 22 seeds (1/pano)"),
    ("manhattan_oracle", "GT seed, 22 seeds (reference)"),
    ("manhattan_anchored", "PanoPin, 5 seeds (1/room) = DEPLOYMENT"),
]


def _errs(arm, panos, gt):
    dts, drs = [], []
    for u in panos:
        cp = paths.WORK / "poses" / arm / u / "camera_pose.json"
        if not cp.exists():
            continue
        d = json.load(open(cp))
        eR = roundtrip._fgpl_rot_to_cw(d["rotation"])
        dts.append(math.dist(d["translation"], gt[u]["location"]))
        drs.append(metrics.rotation_errors({u: eR}, {u: gt[u]["R_cw"]})["per_uuid"][u])
    return dts, drs


def main():
    rows = manhattan.build_pool()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    L = []

    def out(s=""):
        L.append(s)
        print(s)

    out("# Strictly-Manhattan PanoPin -> FGPL: configuration comparison\n")
    out(f"Scene `{manhattan.SCENE}`: {len(manhattan.POOL_ROOMS)} Manhattan rooms "
        f"({', '.join(manhattan.POOL_ROOMS)}), 22 in-frame panos, 5.16M-point map built from "
        "Manhattan rooms only. Non-Manhattan rooms (office_3/7/8) excluded per Point_360 "
        "`roadmap.md` §5. The line map recovered 3 exactly axis-aligned principal directions "
        "with 0% unclassified sparse lines.\n")
    out("All arms share the same panos, map, features and estimator. **The only variable is how "
        "FGPL is seeded.**\n")

    def table(arms):
        out("| configuration | panos posed | rooms covered | trans median | rot median | "
            "rotation locked | wrong-room |")
        out("|---|---|---|---|---|---|---|")
        for arm, label in arms:
            res = json.load(open(paths.WORK / "results" / f"{arm}.json"))
            panos = list(res["true_room"])
            dts, drs = _errs(arm, panos, gt)
            lock = sum(1 for x in drs if x <= 45)
            cov = res["coverage"]
            out(f"| {label} | {len(dts)} | {cov['n_covered']}/{cov['n_rooms']} | "
                f"{st.median(dts):.3f} m | {st.median(drs):.1f}° | {lock}/{len(drs)} | "
                f"{res['scored']['wrong_room_rate']:.2f} |")

    out("### Current estimator (upright rotation prior ON)\n")
    table(ARMS)
    out("\n### Previous estimator (prior OFF) — same arms, for reference\n")
    table(ARMS_NOPRIOR)

    out("\n## The four findings\n")
    out("**1. PanoPin is what makes multi-room work.** FGPL's own global search puts 18/22 panos "
        "in the WRONG ROOM (13.3 m median, 1/5 rooms covered) on same-shape rooms. PanoPin's "
        "seeds take that to 0.96 m and 5/5. This also refutes the 'give FGPL more search area' "
        "idea: global mode IS that idea's maximum, and it is catastrophic.\n")
    out("**2. The deployment config is survey-grade.** One seed per room (D32 room-anchored) "
        "gives **5/5 rooms, 0 wrong rooms, 0 rotation flips, 4.5 cm median**. The 22-seed arm was "
        "an unnaturally hard configuration: 22 seeds subdivide 5 rooms until each pano's Voronoi "
        "cell is starved of the 3D lines the rotation search needs.\n")
    out("**3. Pano crowding causes the 180° flips — mostly.** Controlled test "
        "(`manhattan_aliased_solo`): 4 previously-aliased panos, IDENTICAL seed positions, only "
        "the number of competing panos changed (22 -> 4). **3/4 now lock** (+`127fc8df` in the "
        "anchored arm = 4/5 overall), with geometry rising 3-15x (e.g. `80e1f6ae` 99->302 dense "
        "lines: 179.4° -> 0.3°).\n")

    out("**4. An upright rotation prior removes a whole class of impossible poses.** FGPL's 24 "
        "rotation candidates are the full octahedral group, so 20 tip or invert the camera — "
        "impossible for tripod capture. Of 12 aliased poses, 8 had the camera ON ITS SIDE and 2 "
        "were UPSIDE-DOWN. Constraining candidates to upright (`upright_prior`, opt-in, shipped "
        "on scan2measure branch `feat/upright-rotation-prior`) improved the 22-seed arm's "
        "translation median 0.960 -> 0.084 m with zero regressions and a ~10x faster XDF "
        "search. **It does not solve the flip**: the 4 surviving candidates differ only by yaw, "
        "and 7/22 panos still alias. Read those medians as 'which side of a bimodal split most "
        "panos are on', not as 'errors shrank'.\n")

    out("### Caveats — state these aloud\n")
    out("- **Not universal:** `d0834679` still flips at 179.7° with 481 dense lines (15x more). "
        "Some panos genuinely alias; de-crowding is necessary, not sufficient.")
    out("- **Rotation lock does NOT imply cm position when cells are large.** `f0e54fcd` locks at "
        "1.1° but sits 1.80 m off. Small cells pin translation while starving rotation; large "
        "cells free rotation but loosen translation. It is a trade-off, not a monotonic win.")
    out("- **The anchored arm poses 5 of 22 panos** — it is the answer for per-ROOM coverage, "
        "not for posing every pano. Those are different deliverables.")
    out("- **Weak panos stay weak:** the solo arm (previously-aliased panos) medians 0.923 m even "
        "with a full room, vs the anchored arm's 0.045 m (highest-confidence pano per room).")
    out("- Area_3, n=22, one scene. `tau=0.10` does NOT transfer to this pool (admits 21/22 incl. "
        "3 wrong-room); threshold-free room-anchored seeding (5/5 correct) is the claim that holds.")

    out("\n## Actionable: FGPL already computes a rotation-confidence signal it discards\n")
    out("`multiroom_pose_estimation.py:471` sorts candidates by `(-n_tight, avg_dist)` and takes "
        "[0], throwing away the runner-up. The **margin between the best and second-best DISTINCT "
        "rotation hypothesis** separates good from aliased poses at **42/44**, certifying 21/23 "
        "good poses with ZERO false positives (good margins span 3-56, aliased 0-11; several "
        "aliased poses have margin **0** — an exact tie broken arbitrarily).")
    out("\nIt measures the right thing (was the rotation choice decisive?) and, unlike an absolute "
        "`n_tight` threshold, is computed WITHIN a pano so it does not inherit the geometry-scale "
        "calibration trap that `tau` did. Carrying `rot_idx` into the `candidates` dict (already "
        "on `top_poses[i]`) and emitting the margin into `result_entry` costs no extra compute. "
        "Threshold value (13) is tuned on this scene and needs checking elsewhere.")

    p = paths.HERE / "MANHATTAN_SUMMARY.md"
    p.write_text("\n".join(L) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
