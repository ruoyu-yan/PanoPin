# FGPL⊕PanoPin candidate fusion — F1/F2 results (NEGATIVE at fixed poses)

**Date:** 2026-07-14 · **Plan:** `docs/plans/2026-07-14-candidate-fusion-f1-f2.md` · **Spec:**
`docs/specs/2026-07-14-candidate-fusion-design.md` · **Pool:** `work/seeds/fusion_pool_colored.json`
(12 panos × 30 FGPL candidates, global-mode; colored via `residuals_at_pose`, Rp fed as-is, D-2026-07-14 convention).

## Result (n=12, dev subset; metric = pano→room recall of the selected candidate)

| selector | room recall | note |
|----------|:-----------:|------|
| geometry (argmax `n_tight`) | 4/12 | FGPL's own best pick |
| **F1 verify-select (argmin color)** | **2/12** | color alone selects |
| F2 RRF (k=60) | 2/12 | rank fusion geom+color |
| F2 Borda | 2/12 | rank-sum |
| F2 tie-break (geom top, color breaks, margin 20) | 3/12 | conservative |
| — reference: color-only via CPO `localize_pair` (refined) | 8/12 | cpo_cache, D25/D28 |
| — ceiling: GT-room present in the pool at all | 7/12 | Structure-Y coverage |

**Every fusion variant is ≤ geometry-alone (4/12) and far below refined per-room CPO (8/12). The
fixed-pose color-verify fusion is REFUTED on this subset.**

## Why it fails — reproduces D18/D20 (color needs pose refinement)

1. **No genuine color lock at fixed FGPL poses.** The best color residual across ALL 360 candidates is
   **0.323** (median 0.427); the Task-1 genuine-lock floor at the *true* GT pose is **~0.12–0.18**. FGPL's
   coarse candidate poses — especially their **Manhattan-aliased rotations** (RESULTS.md: even an
   oracle-seeded FGPL rotation is ~90° off) — never let the true room's color residual drop to a lock.
   **Direct rotation-isolation evidence** (3 covered panos, at the *exact GT position*): GTpos+GTrot =
   0.21/0.17/0.24 (locks), but GTpos+**FGPLrot** = 0.43/0.38/0.36 — swapping in FGPL's aliased rotation
   *alone* (position held at GT) doubles the residual and destroys the lock. So the culprit is FGPL's
   rotation, not position and not the convention (GT rotation locks → the Rp-as-is convention is correct).
2. **Color-loss-sink degeneracy returns.** With no genuine lock anywhere, one room's cloud gives the
   lowest residual for almost everyone: **argmin-color picks `office_5` for 7 of 12 panos regardless of
   the true room.** (Task 5 independently: only 4 of 6 rooms ever win nearest-centroid.) This is the same
   pathology as v1 D21 (raw color loss favors a fixed loss-sink), now at fixed poses.
3. **This is a KNOWN result (D18/D20).** The abandoned "cheap Tier-1" scorer failed for exactly this
   reason: *"the true room's low loss only emerges AFTER per-room Adam pose refinement; there is no cheap
   coarse proxy."* Scoring FGPL's unrefined candidate poses IS that refuted cheap proxy. CPO's 8/12 comes
   from `localize_pair`, which **refines** each pose with Adam before scoring.
4. **The InLoc pattern doesn't transfer.** InLoc's render-and-compare *verifier* is image/descriptor
   SIMILARITY, which tolerates small pose error; our verifier is a **per-point color residual**, which is
   pose-fragile and collapses without refinement. So "geometry proposes, color verifies" works in the
   literature but not with this pose-fragile color score at fixed poses.

## Two stacked limitations (the fixed-pose one is the binding one)

- **(a) Pool coverage = 7/12** (Structure-Y): global-mode geometry never surfaces the true room in its
  top-30 for both office_1, one office_4, both hallway_3. Per-room generation (Structure X) would lift
  this to 12/12 — but it would NOT fix (b).
- **(b) Fixed-pose color degeneracy** (the real blocker): even on the 7 covered panos, color at the fixed
  candidate poses does not pick the right room (F1 gets 2/12 overall; the covered-subset behavior is
  dominated by the office_5 sink). Fixing coverage alone cannot rescue F1.

## Implication / levers (decision needed before continuing)

The naive "verify at fixed pose" fusion (F1/F2) is refuted. Real ways forward:
1. **F4 — joint color+geometry refinement:** refine each (or the top few) candidate's pose with color in
   the loss BEFORE comparing. This restores the genuine lock (the D18/D20 fix). Cost: an Adam refine per
   candidate — which converges toward per-room CPO. This is now ESSENTIAL, not optional.
2. **Geometry-shortlist → per-room CPO refine:** use geometry only to prune which rooms to consider, then
   run refined CPO (`localize_pair`) per shortlisted room and pick by refined color. Avoids the fixed-pose
   problem; more expensive; close to `cpo_seeds` but geometry-filtered.
3. **Accept per-room CPO (8/12) as the room selector** and use geometry/FGPL only for the *pose* once the
   room is chosen (color POSITION seed ≈ oracle when room right, D25).

## Reproduce
```
conda run -n panopin-gpu python -m experiments.fgpl_seed.fusion_pool     # ~70 min: 12x30 candidate pool
conda run -n panopin-gpu python -m experiments.fgpl_seed.fusion_color    # ~5 min: color-score the pool
# selection recall computed offline from work/seeds/fusion_pool_colored.json (see this file's table)
```
Caveat: n=12 (D26/D29 discipline). But a 2/12 vs 8/12 gap is far beyond the run-to-run noise; the
fixed-pose degeneracy is a mechanism, not noise.
