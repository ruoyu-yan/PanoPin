# PanoPin → FGPL seed ablation — results

**Date:** 2026-07-10 · **Spec:** `docs/specs/2026-07-10-panopin-fgpl-seed-ablation-design.md` · **Plan:** `docs/plans/2026-07-10-panopin-fgpl-seed-ablation.md` · **Decision:** `docs/DECISIONS.md` D25.

Feeds PanoPin's color-based coarse seed to the modified FGPL pose estimator
(`scan2measure-webframework/src/pose_estimation/multiroom_pose_estimation.py`) and measures FGPL's final
pose accuracy vs S3DIS GT. Run entirely in the **raw S3DIS frame** (identity metadata → no
floorplan/SAM3/density front-end). Subset: `office_1, office_4, office_5, office_6, office_7, hallway_3`
(same-shape offices + a loss-sink corridor), 12 in-frame panos, ≤2/room.

## The 5-arm ablation (n=12)

| Arm | Seed | Trans median | **Trans median (right-room, n=8)** | Trans mean | Trans max | Rot median | Wrong-room | Runtime |
|-----|------|-------------:|-----------------------------------:|-----------:|----------:|-----------:|-----------:|--------:|
| **oracle** | GT position | 0.789 m | **0.725 m** | 0.758 | 1.579 | 90.1° | 8% (1/12) | 557 s |
| **p1** | CPO color position | 0.946 m | **0.762 m** | 7.678 | 24.454 | 105.1° | 33% (4/12) | 536 s |
| **p2** | p1 + grid narrowing (±2 m) | 0.946 m | 0.762 m | 7.679 | 24.454 | 105.1° | 33% | 524 s |
| **p3** | p2 + CPO rotation prior | 1.807 m | 1.651 m | 8.091 | 24.454 | 179.3° | 42% (5/12) | 421 s |
| **wrong_room** | sibling-room centroid | 7.213 m | 7.807 m | 10.168 | 19.172 | 90.0° | 100% | 483 s |

*Rotation scored after converting FGPL output by `C = [[0,0,1],[-1,0,0],[0,-1,0]]` (`Rpᵀ·C` = camera→world).*

## Headline

**Where CPO picks the right room (8/12), the color POSITION seed is as good as ground truth:** P1 median
**0.762 m** vs oracle **0.725 m** — FGPL localizes *identically* whether seeded by PanoPin or by GT. The
entire P1-vs-oracle gap on the full set (0.946 vs 0.789) is the **4 wrong-room misses** (all → `hallway_3`,
the loss-sink corridor; ~22 m each). → **the color position seed works; the bottleneck is room-assignment
recall** (67% here, `cpo_seeds` uses RAW min-loss with no v1 calibration).

## Reading each arm

- **P1 (color position seed):** validated. When the room is right, ≈ oracle. The 4 failures are exactly the
  4 CPO room-assignment misses. → the drop-in for the jigsaw's `camera_position` is sound.
- **P2 (translation-grid narrowing):** **no effect** (identical to P1). The Voronoi partition already
  constrains the search enough; narrowing the grid around the seed adds nothing. **Drop it.**
- **P3 (CPO rotation prior):** **hurts** (median 1.807 m; even right-room subset 1.65 m vs P1's 0.762).
  Not because rotation priors are bad, but because **CPO's rotation is convention-unresolved** (~120° off
  GT in *both* camera→world and world→camera), so the yaw prior is garbage and forcing it degrades FGPL.
  **Don't use CPO rotation until its convention is solved.**
- **wrong_room (floor):** 100% wrong-room, ~7 m — the catastrophic failure the seed must prevent.

## Room-assignment context (why the 33% wrong-room)

CPO raw min-loss room recall@1 = **8/12 (67%)**; the 4 misses all went to `hallway_3` (loss-sink, D21).
PanoPin **v1 calibration** (`src/panopin/calibrate.py`) lifts whole-area recall 33%→58% and its confidence
gate flags loss-sink misses — i.e. it directly targets these 4 failures. `cpo_seeds.py` currently uses
**raw** min-loss (no calibration), so wiring v1 calibration into the seed is the clear next step.

## FGPL rotation convention (by-product finding)

FGPL emits rotation as `Rp = C @ R_wc` with `C = [[0,0,1],[-1,0,0],[0,-1,0]]` (equirect signed-perm — the
same convention the Point_360 project solved). `Rpᵀ · C` = camera→world, accurate to **0.4–0.5°** on
precisely-localized oracle panos. But FGPL's rotation is *also* frequently ~90° Manhattan-aliased even with
a perfect position seed (oracle rot median 90°) — a *good* rotation prior could help; CPO can't provide one.

## Recommendations

1. **Adopt P1** (color position seed) as the jigsaw replacement — simplest handoff, near-oracle when the
   room is right, and ~8 min/arm vs the jigsaw's ~300 s coarse cost (which also scales combinatorially).
2. **Wire v1 calibration** (`calibrate.assign`) into `cpo_seeds` room assignment and re-run — expected to
   cut the 4 wrong-room misses (the whole P1↔oracle gap).
3. **Drop P2 and P3.** Grid narrowing is a no-op; the CPO rotation prior is convention-broken and harmful.
4. Rotation guidance is only worth revisiting after resolving CPO's rotation convention.

## Calibration re-run (2026-07-13) — wiring v1 into the seed, NEGATIVE result

Executed the "next step" above: `cpo_seeds` now caches every room's pose + the v1 minmax-calibrated room
(`panopin.calibrate`, D24); new `p1_cal` arm seeds FGPL from the **calibrated** room; p2/p3 dropped (D25).
Fresh self-consistent re-run (cpo_seeds 25 min + 4-arm ablation). **Calibration did NOT boost FGPL — it
slightly hurt it on this n=12 subset.**

| Arm | Seed | Trans median | Trans median (right-room) | Trans mean | Trans max | Wrong-room | Runtime |
|-----|------|-------------:|--------------------------:|-----------:|----------:|-----------:|--------:|
| **oracle** | GT position | 0.789 m | — | 0.758 | 1.579 | 8% (1/12) | 466 s |
| **p1** | CPO raw min-loss | 1.171 m | 0.921 (8) | 8.191 | 24.426 | 33% (4/12) | 446 s |
| **p1_cal** | CPO v1-calibrated | 1.839 m | 1.158 (8) | 7.966 | 24.426 | 42% (5/12) | 433 s |
| **wrong_room** | sibling centroid | 7.004 m | — | 10.047 | 19.172 | 100% | 434 s |

Room recall (cpo_seeds): **raw 8/12 = calibrated 8/12** — calibration TRADES misses, not removes them.
Per-pano, it changed 4 seeds: **fixed 2** loss-sink misses (870 office_4 22.0→2.3 m; 0e3 office_7
24.2→3.5 m) but **broke 2** correct panos into siblings (7e48 office_1→office_5 1.0→14.9 m; decc
hallway_3→office_5 0.9→22.1 m). The sibling errors it introduces are as large as the loss-sink errors it
removes; the shared multi-pano Voronoi also nudged unchanged panos ~+1 m.

**Why the offline 10/12 projection was wrong:** it ran on the OLD committed loss cache; CPO is not
bit-reproducible (D1) and a fresh cache gave 8/12. Noise is large at n=12 — p1's own median drifted
0.946 (2026-07-10) → 1.171 (this run) with no code change. **Under-powered; do not conclude from n=12.**
Salvageable piece = the confidence **gate** (3/3 confident correct at 25% coverage; correctly withholds
both breaks) → use ABSTAIN-not-mis-seed, not wholesale room swap. See DECISIONS **D26**.

## Follow-up (2026-07-13) — robust-gate comparison → see `ROBUST_RESULTS.md` (D27)

Tested whether a robust per-point residual score beats the minmax gate on CPO's mean loss (precision/coverage,
seed-only-when-confident). **Robustness did NOT help** (best robust ties the raw-mean baseline, 9/12 vs 9/12).
**Unexpected lead:** a raw *unweighted* residual mean lifts the confidently-correct prefix 3→9 vs CPO's deployed
match_color+weighted loss. Numbers + method in `experiments/fgpl_seed/ROBUST_RESULTS.md`; decision in D27.

## Reproduce

```
conda run -n panopin     python -m experiments.fgpl_seed.gate_oracle   # Phase-0 oracle gate (~10 min)
conda run -n panopin-gpu python -m experiments.fgpl_seed.cpo_seeds      # per-room poses + v1 calibration -> work/seeds/cpo_cache.json (~25 min)
conda run -n panopin     python -m experiments.fgpl_seed.run_all        # 4-arm ablation (oracle/p1/p1_cal/wrong_room) -> work/results/ablation.json (~30 min)
```
Estimator runs in `panopin-gpu` (Ada cu118); build tools in `scan_env`; CPO in `panopin-gpu`. The dropped
P2/P3 FGPL narrowing edit is `patches/fgpl_narrowing.patch` (flag-gated, default-off; apply into
scan2measure). `work/` is gitignored — the numbers above are the committed record.
