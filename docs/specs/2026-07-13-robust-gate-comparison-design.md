# Design — precision-first comparison: minmax gate (a) vs robust-loss score (c)

**Date:** 2026-07-13 · **Branch:** `feat/fgpl-seed-ablation` · **Follows:** DECISIONS D24 (minmax
calibration + gate), D26 (calibration-as-seed-selector net-hurt FGPL at n=12; the gate is the
salvageable piece) · **Results feed:** `experiments/fgpl_seed/RESULTS.md`.

## 1. Goal and success criterion

The deployment rule is fixed (user, 2026-07-13): **PanoPin seeds FGPL only when confident; otherwise it
abstains and FGPL falls back to its own slow coarse path.** A *wrong* seed is worse than *no* seed — a
right-room seed localizes to ~1 m, a loss-sink/sibling seed blows up to ~22 m with no FGPL recovery (D25/D26).

So the deliverable is a **precision/coverage** result at the room-assignment level, not a full-set median:

- **Precision** = of the panos PanoPin marks confident, the fraction whose assigned room == GT room.
- **Coverage** = |confident panos| / n.
- **Winner = the method that reaches the highest coverage while holding 100% precision** (hand FGPL the
  most panos without ever handing a wrong room).

Two methods are compared, sharing ONE gate and differing only in the per-room *score* fed to it:
- **(a)** minmax gate on CPO's existing **mean** loss.
- **(c)** minmax gate on a **robust per-point** statistic of the color residuals.

Non-goal: deciding the single lowest-full-set-error seed policy (that would favor coverage over precision,
which the user explicitly rejected). Non-goal: scaling to 76 panos (premature until a lever beats the noise).

## 2. Background — why the mean loss fails, and why a robust stat might not

CPO's room score is `rgb_loss = mean(‖sample_rgb − cloud_rgb‖)` over sampled points
(`third_party/cpo/cpo/sampling_loss.py:203`, and the weighted-masked-mean variant at :275/:278). Two known
pathologies are both *outlier* effects on that mean:
- **Loss-sink (D21):** a fixed set of small/corridor rooms score a mediocre mean for *every* pano (no strong
  match, but no terrible mismatch either), burying the true room. On the 6-room subset this is `hallway_3`.
- **Window/occlusion (D23):** a genuine pano whose view is ~40% window/blank has a high own-room mean because
  a large minority of points mismatch hard — inflating its true room's loss.

Hypothesis: a **robust** statistic separates these better than the mean —
- the **true room** has a strong low-residual core (most points match) with a minority of high-residual
  outliers (windows/occlusion) → a low-percentile / trimmed statistic rewards the core;
- a **loss-sink** is mediocre *everywhere* → no strong low core → a low statistic penalizes it.

This is a hypothesis, not a given. D26 already showed minmax at n=12 is noisy and can trade misses; the
robust score could do the same. The precision/coverage comparison is exactly what tests it.

## 3. Shared gate (already built — `src/panopin/calibrate.py`)

`calibrate.assign(score_matrix)` where `score_matrix = {pano: {room: score}}` (lower = better) returns
`{pano: Assignment(room, confidence, is_confident)}`:
- per-room minmax-normalize each score against that room's leave-one-out [min,max] over the other panos
  (fair, no GT — D5);
- `room` = min-score room; `confidence` = its normalized score (lower = more confident);
- `is_confident = confidence <= conf_threshold`.

Both methods call this SAME function; only `score_matrix` differs. Precision/coverage is produced by
sweeping `conf_threshold` over the resulting confidences (see §6).

## 4. Approach (a) — minmax gate on the mean loss (no new compute)

The score matrix is the existing `cache[pano]["per_room"]` (CPO mean loss for all 6 rooms), already in
`work/seeds/cpo_cache.json`. Derive its precision/coverage curve **offline** from the cache + GT. This is the
D26 gate, now read as a curve rather than a single threshold. (Reference point from the last run: threshold
−0.3 → coverage 3/12, precision 3/3.)

## 5. Approach (c) — robust per-point score

### 5.1 Residual capture (the only new GPU work, ≈3 min)
Add `residuals_at_pose(cfg, pano_path, cloud_path, t, R) -> np.ndarray` to the CPO layer. It replicates
`sampling_loss`'s sampling **at a fixed, given pose** (the room's already-refined `(t,R)` from the cache) and
returns the **raw per-point color residual vector** `‖sample_rgb[mask] − cloud_rgb[mask]‖` *before* the mean —
composing CPO primitives only, **no edit to `third_party/cpo`** (D9). Because the pose is fixed and given,
this is a single sampling forward (no histogram search, no Adam) → cheap.

Reuse the cached per-room poses (6 rooms × 12 panos already stored in `cache[pano]["poses"][room]`), so no
re-localization is needed. A capture script iterates the 72 (pano, room) pairs and stores, per pair, a
**101-point percentile grid** (p0..p100) of the residual vector as JSON (`work/seeds/residuals.json`) — exact
for median and any percentile, sufficient for trimmed-means over a percentile sub-range, tiny, no pickle.

**Fidelity validation (gate before trusting robust stats):** a check asserts that `mean(residuals_at_pose)`
equals CPO's own `cpo.sampling_loss` scalar at the **same** pose to 1e-4 — an EXACT match, because both are
the unweighted mean of `‖sample_rgb[mask] − cloud_rgb[mask]‖` over the same sampled points, so the replication
is proven bit-faithful. (Refined during planning from an earlier "match the cached loss within ±0.02" idea:
the cached loss comes from `refine_pose_sampling_loss`, which is a *weighted* masked mean via the `SamplingLoss`
class needing the expensive score-maps, so it is not the right fidelity target. The robust statistic operates
on these raw unweighted residuals.) If the check fails, the replication is wrong and must be fixed before any
robust conclusion.

### 5.2 Robust scores (offline, no GPU) — `src/panopin/robust_score.py` (NEW, fair/no-GT)
Pure functions: `robust_scores(residual_grids, stat, **params) -> {pano: {room: score}}`, where `stat ∈
{median, trimmed_mean, low_percentile}`:
- `median` = p50;
- `low_percentile` = p_q (sweep q ∈ {10, 20, 25});
- `trimmed_mean` = mean of the percentile grid over [0, 100−k] (drop the top-k% high-residual outliers;
  sweep k ∈ {10, 20, 30}).
Reads only residual grids (no GT — D5 fair). The choice of `stat`/params is **swept offline** and picked by
precision/coverage; nothing is pre-committed.

### 5.3 Gate the robust scores
Feed each candidate robust score matrix through the SAME `calibrate.assign` → robust confidence + gate →
robust precision/coverage curve. Identical machinery to (a); only the underlying score changed.

## 6. Comparison and deliverable — `experiments/fgpl_seed/robust_analysis.py` (NEW, offline)
For (a) and each (c) variant:
1. build `score_matrix`; 2. sweep `conf_threshold`; 3. at each threshold record (coverage, precision) using
GT (experiments-side, allowed); 4. summarize each method by **coverage-at-100%-precision** and
**coverage-at-first-error**.

Deliverable = a table + the two-curve overlay written to `experiments/fgpl_seed/ROBUST_RESULTS.md`:
| method | score | cov@100%prec | cov@first-error | recall@1 (all) |
Winner = highest cov@100%-precision. Report honestly if (c) does not beat (a) (very possible at n=12); note
the n=12 noise caveat (D26) explicitly.

## 7. FGPL secondary check — DEFERRED (design call 2, confirmed)
The primary decision is settled offline. A confident-correct-room seed already gives FGPL ≈ oracle pose
(right-room 0.762 vs 0.725 m, D25) and minmax's confident subset was already measured (3/3, ~1 m) in the D26
run. Only if (c) makes materially MORE panos confident do we re-run FGPL to confirm those *extra* panos
localize well — a follow-up arm, not part of this spec.

## 8. Components and boundaries
| unit | location | purpose | deps |
|------|----------|---------|------|
| `residuals_at_pose` | `src/panopin/cpo_adapter.py` (+fn) | per-point residuals at a fixed pose | CPO primitives (no third_party edit) |
| `robust_score.py` | `src/panopin/` (NEW) | mean→robust re-score of residual grids; fair, no GT | numpy only |
| `calibrate.assign` | `src/panopin/` (reuse) | the shared gate | — |
| `robust_capture.py` | `experiments/fgpl_seed/` (NEW) | GPU: residual grids + fidelity check → `work/seeds/residuals.json` | adapter, cache, panopin-gpu |
| `robust_analysis.py` | `experiments/fgpl_seed/` (NEW) | offline: build/sweep/compare → `ROBUST_RESULTS.md` | robust_score, calibrate, GT |

## 9. Testing
- **Unit (`tests/test_robust_score.py`):** synthetic residual grids → assert median/low-percentile/
  trimmed-mean values; and a discrimination test — a "true-room" grid (strong low core + few high outliers)
  must score **below** a "loss-sink" grid (mediocre-everywhere) under `low_percentile` and `trimmed_mean`,
  where it would NOT under the plain mean. This encodes the §2 hypothesis as a test.
- **Fidelity (integration, GPU, in `robust_capture.py`):** weighted-masked-mean of captured residuals ==
  cached CPO loss within ±0.02 (D1). Hard-fail the capture if violated.
- **Existing suites stay green** (`tests/`, `experiments/fgpl_seed/tests/`).

## 10. Risks and mitigations
- **Residual replication is unfaithful** → robust stats meaningless. Mitigation: §5.1 fidelity gate before
  any conclusion.
- **Robust score does not beat the mean at n=12** (real possibility). Mitigation: this is a legitimate
  negative outcome — report it plainly with the noise caveat; it still tells us the gate lever, not the score,
  is where PanoPin stands, and points to (b) scaling or a different mechanism.
- **Cached poses are a bad realization** (Adam local min) → residuals high regardless. Accepted: the robust
  score reflects the same localization both methods see; this isolates *scoring*, not pose search (design call 1).

## 11. Reproduce (planned)
```
conda run -n panopin-gpu python -m experiments.fgpl_seed.robust_capture    # residual grids + fidelity (~3 min)
conda run -n panopin     python -m experiments.fgpl_seed.robust_analysis   # offline precision/coverage compare
```
No FGPL re-run in this spec. `work/` is gitignored; committed numbers live in `ROBUST_RESULTS.md`.

## 12. Resolved design decisions
1. **Re-rank cached localizations by a robust readout** (NOT re-run CPO with a robust Adam objective) —
   isolates the scoring question cheaply; a robust objective is a later move if the readout shows promise.
2. **FGPL secondary check deferred** (§7).
Both confirmed by the user, 2026-07-13.
