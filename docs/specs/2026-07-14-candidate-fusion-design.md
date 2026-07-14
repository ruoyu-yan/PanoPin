# Design — fusing PanoPin color candidates with FGPL geometry candidates

**Date:** 2026-07-14 (rev. after literature review) · **Branch:** `feat/candidate-fusion` (off `main`) ·
**Literature grounding:** `docs/geometry-color-fusion-literature-review.md` (§6–§8 below revised to its
findings: verify-select over weighted blend; added joint refinement). · **Supersedes the active thread:** the D29
"confidence gate over raw-mean at whole-area scale" problem is *deprioritised* — see §1 (deployment
reframing). **Builds on:** D25 (color POSITION seed ≈ oracle when the room is right; FGPL handoff is
positional-only; `C = [[0,0,1],[-1,0,0],[0,-1,0]]` rotation convention), D27/D28 (raw per-point residual
mean is the best color score; `residuals_at_pose` scores any fixed pose cheaply). **Results will feed:**
a new `experiments/fgpl_seed/FUSION_RESULTS.md`.

## 1. Goal and the deployment reframing (why this replaces the old thread)

**User clarification (2026-07-14):** the real deployment is **a 3–5 room point cloud with panoramas
covering every room — no empty/uncovered rooms.** This changes the target:

- The whole-area **23-room** regime that produced the D21/D29 "loss-sink" pain was an S3DIS-eval
  artifact: 17 of those rooms had **no panos** (tiny corridors / WC / storage) and acted as degenerate
  distractors that fooled every room score. In deployment those rooms do not exist as uncovered
  distractors. So **the D29 open problem (a confidence gate that survives 23 rooms) is not the
  deployment problem** and is set aside, not solved.
- The realistic regime is closer to the **6-room dev subset** already built
  (`experiments/fgpl_seed/subset.py`: office_1/4/5/6/7 + hallway_3, 12 in-frame panos) — and often
  smaller. At 3–5 fully-covered rooms, searching *every* room per pano is ~1–2 min/pano (§5), so we no
  longer need to commit to one room up front to keep cost down.

**New goal.** Maximise **final FGPL pose accuracy vs S3DIS GT** (the metric that matters) by combining
two independent cues instead of chaining them, so each covers the other's blind spot. Secondary:
report room-assignment accuracy of the fused pick.

**Success criterion.** On the dev subset, the fused pipeline's **translation error vs GT** beats the
current geometry-only pipeline (`p1` in RESULTS.md: 0.95 m median, 33% wrong-room) and approaches the
oracle (0.79 m, 8% wrong-room) — specifically by cutting the wrong-room / wrong-rotation misses that are
the entire p1↔oracle gap. Non-goal: scaling to 76 panos, or the 23-room gate (both deprioritised above).

## 2. What the pipeline does *today* (verified against the code)

Read from `scan2measure-webframework/src/pose_estimation/`:

- **FGPL already produces a ranked candidate LIST per pano.** `pose_search.py:341`
  (`xdf_coarse_search_from_precomputed`) scores every (rotation × translation) hypothesis by a geometric
  **line-intersection inlier count** and returns `top_k` candidates (`multiroom_pose_estimation.py`
  `TOP_K=10`), each `{R, t, cost}`. `multiroom_pose_estimation.py:415-472` then ICP-refines all `top_k`
  and selects the winner **by geometry only**: `sort(key=(-n_tight, avg_dist))` (most tight inliers, then
  smallest angular residual). Color never enters.
- **PanoPin's signal is used early and then discarded.** The seed position only carves a **Voronoi
  region** (`load_panorama_positions` → `get_local_mask`, lines 104-143): it picks *which room's lines*
  FGPL may search. It is not in the cost. → an **early hard commitment**. A wrong-room seed confines
  FGPL to the wrong region with no recovery (the ~22 m loss-sink failures).
- **The two cues fail on opposite inputs.** Geometry (lines) is blind to *which* same-shape room (the
  reason PanoPin exists) **and** is Manhattan-**rotation-aliased** — even with a perfect position seed the
  oracle arm's rotation median is 90° (RESULTS.md §"FGPL rotation convention"). Color is blind on
  windows / blank walls / occlusion (D23) but is **not** rotation-aliased (a red wall is in exactly one
  direction) and **not** same-shape-blind (content differs). Independent failure modes ⇒ fusion should
  strictly dominate either cue alone.

**Consequence (the bonus):** fusing color into candidate *selection* can fix **two** current failure
classes at once — wrong-room (color disambiguates same-shape geometry) **and** wrong-rotation (color
penalises the 90°-aliased rotations geometry can't tell apart). Neither cue alone does both.

## 3. The cheap primitive that makes fusion practical

`src/panopin/cpo_adapter.residuals_at_pose(cfg, pano, cloud, t, R)` (already built + fidelity-gated,
D28) returns the per-point color residuals `‖sample_rgb − cloud_rgb‖` at a **fixed** pose — one
projection, **no search, no Adam**. Its raw mean is the D28-best color score. So scoring an
FGPL-produced candidate pose by color is cheap (§5).

**Frame/convention interface (must be exact — DERIVED empirically 2026-07-14, not guessed).** FGPL emits
`Rp = C @ R_wc` (`C = [[0,0,1],[-1,0,0],[0,-1,0]]`, D25). `residuals_at_pose` expects the **equirect-
convention** rotation `C @ R_wc` — CPO and FGPL share this convention (same lab) — so **feed `R = Rp`
AS-IS** (this corrects an earlier draft that said `Cᵀ·Rp`). For a GT pose, feed `R = C · R_cwᵀ`. `t =
t_world` as-is. Everything runs in the raw S3DIS frame (identity metadata, ~5 mm frame identity per
Point_360). The blocking R1 pre-check verified this: `fusion_convention_probe.py` swept all natural
compositions and `C·R_cwᵀ` alone landed at the CPO genuine-lock floor (~0.12–0.18) vs ~0.40–0.46 for the
rest; the gate then passed 12/12 (GT residual < sibling, median 0.184).

## 4. The two fusion structures

Both **keep every candidate room in play** (no early commitment) and pick the final pose using **both**
cues. They differ in how candidates are generated.

### Structure X — per-room search, then color+geometry pick *across* rooms (recommended)
```
for room in candidate_rooms:                       # all 3–5 rooms (deployment) / the 6 dev rooms
    pose_r, geom_r = FGPL.search_refine(pano, room_lines[room])   # geometry: best pose + n_tight/avg_dist
    color_r        = mean(residuals_at_pose(pano, room_cloud[room], pose_r))   # color at THAT pose
final = argmin over rooms of  combine(geom_r, color_r)            # §6 rank-based
```
Each room gets a *fair* geometric pose and a color score; the winner must satisfy **both** cues.

### Structure Y — multi-room pool, color re-ranks one mixed list
```
cands = FGPL.search(pano, pooled_lines[all rooms])   # top_k candidates, possibly from different rooms
for c in cands: c.color = mean(residuals_at_pose(pano, room_cloud[c.room], c))
final = argmin over cands of  combine(c.geom, c.color)
```
Closer to a literal "merge the two lists", but the pooled search reintroduces the cross-room false-minima
that local Voronoi filtering exists to remove, and depends on `top_k` actually containing a right-room
candidate.

## 5. Head-to-head comparison (the four axes requested)

| Axis | **X — per-room search** | **Y — pooled search + re-rank** |
|---|---|---|
| **Compute / pano** | N separate FGPL runs (N = #rooms) + N cheap color scores. Each FGPL run is Voronoi-cell-sized (~30–60 s). **N=3–5 ⇒ ~2–5 min/pano**; color adds ~N×(1–3 s). Embarrassingly parallel across rooms. | One FGPL run over the union of all rooms' lines (bigger translation grid, `top_k` color scores). One precompute, but the larger multi-room extent is slower per unit **and** noisier (cross-room minima); `top_k` must grow to stay safe. |
| **Robustness: wrong-room** | **High, by construction.** Right room always gets a candidate + color score; a wrong room is rejected if *either* cue disfavours it. No dependence on `top_k`. | **Conditional.** If geometry's `top_k` is swamped by a same-shape wrong room, the right room may never enter the list and color cannot rescue it. Mitigation = larger `top_k` / forced per-room diversity (drifts toward X). |
| **Robustness: window/blank pano (color unreliable)** | Geometry still ranks all rooms; when color is low-confidence (see §6 fallback) the combiner down-weights it → graceful fall back to geometry. Same panos as D23 hard-floor rely on geometry. | Same fallback available, but a color-driven re-rank of a geometry-limited list has less to work with. |
| **Setting the geom↔color weight** | Rank-based combine (§6) sidesteps unit mismatch (inlier *count* vs residual *meters*); one scalar `w` (or "color breaks geometric ties within margin") tuned on the dev subset. Per-room symmetry makes `w` interpretable. | Same combiner, but candidates are heterogeneous across rooms/rotations, so the rank distribution is messier to calibrate. |
| **What's measurable at n=12** | Paired, same-pano comparison vs (i) geometry-only `p1`, (ii) color-only PanoPin, (iii) oracle. Can add trials by resampling room-subsets of size 3–5 from the 6 dev rooms (many (pano, room-set) draws) → more power than 12 raw panos. | Same metrics, but more free knobs (`top_k`, pooled-grid) make attribution of any gain harder at n=12. |

## 6. The selection rule (revised to the literature)

The literature review changes the default here. The localization field's proven pattern (InLoc CVPR 2018,
MegaPose CoRL 2022, Patch-NetVLAD, hloc) is **not** a weighted blend of the two scores — it is **"geometry
proposes the ranked candidate pool; the render-and-compare score verifies/selects the winner"**, with the
color score often the *decisive* selector (it may override the geometric ranking). Structure X's per-room
FGPL search *is* the candidate generator; the fusion happens at selection. We test three selection rules,
in this order (the F-menu from the review §6):

- **F1 — verify-select (PRIMARY).** Among the per-room candidate poses, pick the one with the **lowest
  color residual** (`residuals_at_pose` raw mean, D28). Color alone decides; geometry only defined the
  pool. Targets wrong-room *and* rotation aliasing (color is neither same-shape-blind nor rotation-
  invariant). Precedent: InLoc, MegaPose.
- **F2 — rank/score blend (BRACKET).** Combine the geometric score (`n_tight`) and color score to check
  whether keeping geometry's vote guards window/blank panos where F1 over-commits. Use **rank fusion**
  (Reciprocal Rank Fusion, `score = Σ 1/(k+rank_i)`, k≈60, or Borda) — unit-free, no calibration — and/or
  a **tanh-normalized equal-weight sum** (negate the color residual first). **No learned/tuned weight**:
  at n=12 fixed rules beat trained combiners (Kuncheva 2004). The conservative end is **color as a pure
  tie-breaker** among geometry's top candidates within a margin.
- **F3 — reliability-weighted (DEFERRED).** Per-pano weight from a cheap reliability signal (color:
  residual spread / window fraction; geometry: inlier margin). Only if F1/F2 show reliability varies and
  matters. Precedent: Nandakumar quality-based fusion, mixture-of-experts.

**Confidence (folds in the old D29 gate):** the F1/F2 selection *margin* (gap between the best and
second-best candidate score) is the seed-or-abstain signal — soft/probabilistic selection (DSAC,
FAB-MAP), replacing the minmax gate that broke at scale.

## 6b. Joint color+geometry refinement (F4 — additive fine stage)

Independent of selection, add a colored-ICP-style **joint refinement** of the *continuous* pose after the
winner is chosen: minimize `E = w_g·E_line + w_c·E_color`, coarse-to-fine, with a robust kernel on each
term and the weight set from residual covariance (Gutiérrez-Gómez 2015) rather than guessed. `E_color` =
PICCOLO's differentiable sampling loss (already available); `E_line` = FGPL's line-alignment residual.
This targets the **90° rotation aliasing** that persists even with a perfect position seed (RESULTS.md
oracle rot median 90°) — the one failure neither cue fixes alone and FGPL's line-only refine cannot.
Precedent: Colored ICP (Park 2017), SVO cascade, Gutiérrez-Gómez covariance weighting.

## 7. Recommendation

**Structure X.** It (a) guarantees the right room is always evaluated (no `top_k` dependence), (b) reuses
existing tools — a per-room line map is one `build_linemap` call on a single-room PLY (`build_ply` already
builds from per-room `rows`), and the estimator's line-filtering path (`get_local_mask` /
`multiroom_pose_estimation.py`) already restricts a search to a region — (c) keeps the cross-room
false-minima suppression that local filtering was built for, and (d) is fully affordable in the 3–5 room
deployment regime. Y is the more literal "merge the lists" but reintroduces the exact problem
Voronoi filtering solved and risks the right room never entering the candidate list. Implement X; keep Y
as a documented alternative only if X's per-room cost proves prohibitive (it should not at N≤5).

## 8. Validation plan (n=12 dev subset as the realistic proxy)

**Test order F1 → F2 → F4** (user-approved 2026-07-14), staged so we learn between each.

- **Fixed reference arms** (all scored by `eval/metrics` vs GT, paired per pano):
  `geom_only` (current p1, FGPL geometry-selected), `color_only` (PanoPin min-residual room → its FGPL
  pose), `oracle` (GT position seed).
- **F1 arm** `fusion_verify_select`: Structure-X per-room candidate pool → lowest-color-residual pick.
  Primary question: does color-picks-winner beat `geom_only` and approach `oracle`?
- **F2 arm(s)** `fusion_rrf` / `fusion_tanhsum` / `fusion_tiebreak`: does keeping geometry's vote help vs
  F1, especially on the window/blank panos where color-alone should over-commit? No learned weight.
- **F4 arm** `fusion_joint_refine`: F1 winner + joint color+geometry refinement. Question: does it cut the
  rotation error (report rotation median, not just translation), and does it hold translation?
- **Primary metric:** translation-error median + wrong-room rate; **rotation median** promoted to primary
  for F4 (that is what it targets).
- **Power:** resample 3–5-room subsets from the 6 dev rooms for extra (pano, room-set) trials; still flag
  n as small (D26/D29 discipline — no over-claiming from a handful of panos).
- **Blocking pre-check (R1):** before any fusion number, verify the frame conversion `R_wc = Cᵀ·Rp` — an
  oracle-GT-posed pano must yield a *low* color residual. If not, fix the convention first.
- **Fairness (D5):** `src/panopin/*` reads only manifest/residuals; GT used only in the oracle/scoring
  arms under `experiments/`.

## 9. Risks and open questions

- **R1 — rotation-frame conversion (§3). RESOLVED 2026-07-14:** the color feed is `R = Rp` as-is (GT: `C·R_cwᵀ`),
  derived empirically (`fusion_convention_probe.py`); the blocking gate passed 12/12 (median GT residual 0.184).
- **R2 — FGPL rotation aliasing pollutes color scores.** If FGPL's *geometric* best pose per room is
  90°-rotated, its color residual is high even for the right room. Mitigation: color-score *all* of FGPL's
  `top_k` rotations per room and take the room's best — turns R2 from a risk into the §2 bonus.
- **R3 — per-room search cost at deploy scale.** N≤5 is fine; if a future scene has more rooms, fall back
  to "PanoPin top-N rooms only" (color pre-shortlist) before the per-room FGPL search — a natural throttle.
- **R4 — n=12 noise.** Same caveat that killed the D28 6-room lead at scale (D29). Mitigate with paired
  comparison + subset resampling; do not deploy on a single run.
- **Open:** does `refine_pose` need the seed at all once we search per room, or can we drop the Voronoi
  seed entirely and let color pick? (The experiment answers this.)

## 10. Non-goals

- The 23-room whole-area confidence gate (D29) — deprioritised by §1, not solved here.
- Scaling to 76 panos.
- Editing `third_party/cpo/` or the FGPL estimator's core (D9); the per-room driving lives in
  `experiments/fgpl_seed/`, composing existing primitives.
- Learned fusion weights — deterministic rank combine only (charter).
