# Deployment-regime characterization + the low-percentile room score (2026-07-14)

**Question.** The user reframed deployment (2026-07-14): the real input is a **3–5 room point
cloud with EVERY room covered by ≥1 pano**, not 12 panos vs all 23 Area_3 rooms. So (Q1) what
recall do we actually get in that regime, and (Q2) is there a better per-room score than the
D28 raw-mean? Answered **entirely offline** by rescoring the cached whole-area residual grids
(12 panos × 23 rooms, 101 percentiles per pair) restricted to subsets of the 6 **covered** rooms.
No GPU. GT used only to score (D5).

**Design.** For candidate-set size k, enumerate all C(6,k) subsets of the 6 covered rooms; each
subset scores only the panos whose true room is in it. Every pano's true room lies in exactly
C(5,k−1) subsets → micro-recall == macro-recall (uniform pano weight); each pano is evaluated
across all ways its k−1 distractor rooms can be drawn. **Fidelity check:** at k=23 (full list)
this reproduces WHOLEAREA_RESULTS.md exactly — CPO-loss 6/12, raw-mean 8/12.

Reproduce (all in `panopin`, offline): `python -m experiments.fgpl_seed.deploy_regime`
(Q1–Q3), `… .deploy_regime_xcheck` (independent-cache cross-val), `… .deploy_regime_qpin`
(q-pin + rescue), `… .deploy_regime_gate` (Q4 calibration).

## Q1 — the deployment regime is far easier than whole-area
Recall vs candidate-set size k (all rooms covered), whole-area residual cache:

| score | k=2 | k=3 | k=4 | k=5 | k=6 | k=23 |
|-------|-----|-----|-----|-----|-----|------|
| CPO-loss | 92% | 88% | 87% | 85% | 83% | 50% |
| raw-mean | 90% | 88% | 87% | 85% | 83% | 67% |

At 3–5 covered rooms recall is **85–88%**, vs 67% at whole-area. The user's reframing pays off.

## Q2 — low-percentile is the best per-room score, and it CROSS-VALIDATES
Full statistic sweep (whole-area cache): median / trimmed-mean / low-percentile all reach **92%**
at k≤6 and 75–83% at k=23, beating raw-mean (83% / 67%). **But cross-validation on the
independent 6-room cache** (`residuals.json`, poses refined against the 6-room clouds) filters
these:

| statistic | 6-room k=6 | whole-area k=6 | generalises? |
|-----------|-----------|----------------|--------------|
| raw-mean | 67% | 83% | baseline |
| median | 67% | 92% | **no** (cache-specific) |
| trimmed-mean | 67% | 92% | **no** (cache-specific) |
| **low-pct q10–q25** | **75%** | **92%** | **YES — both caches** |

Only **low-percentile** wins on both. Its advantage **grows with room count** (tied at k=3,
+8 pts at k=6 on the 6-room cache) — the loss-sink signature. **q-pin:** identical recall for a
**wide plateau q∈[5,25]** (q=30 reverts) → not an n=12 knife-edge. Pinned **q=20**.

## Q3 — the remaining bottleneck is the hallway_3 loss-sink
At k=6, both raw-mean misses go to **hallway_3** by tiny margins (+0.006, +0.094). low-pct q20
**rescues office_5** (8/12→9/12 on the 6-room cache) but 3 deeper hallway_3 captures survive.
**Loss-sink capture by hallway_3 is THE dominant failure mode** — the next lever.

## Q4 — low-percentile needs NO calibration (fits the per-pano mandate)
minmax calibration (`calibrate`, the designed loss-sink fix) lifts the loss-sink-degenerate
CPO-loss a lot (whole-area 83→92%) and raw-mean modestly (6-room 67→75%), but adds **~nothing on
top of low-pct** (already ~92% / 75%). low-pct reaches the calibration ceiling with **plain
argmin, scoring each pano independently** — no cross-pano stats, works with few panos. Best
single config across both caches: **low-pct q20 + argmin** (92% whole-area, 75% 6-room, flat in k).

## Decision (→ D30) and what shipped
- **Adopt low-percentile (q=20) as the deployable room-assignment score** — `robust_score.
  low_percentile_scores` (+ test). Dominates raw-mean/CPO at every k on both caches, no
  calibration needed, per-pano. Supersedes `raw_mean_scores` for assignment; median/trimmed
  **rejected** (cache-specific).
- **NOT yet rewired** into `select_room`/`cpo_seeds` (those rank by CPO scalar loss / raw-mean)
  — that deployable swap + a **larger-pano GPU validation** (n=12 is the standing caveat) is the
  recommended next step, plus attacking the hallway_3 loss-sink directly.
