# Cheap Tier-1 scorer + deterministic in-frame eval — design spec

**Date:** 2026-07-09 · **Status:** approved (user) · **Branch:** `feat/coarse-room-cpo`
**Supersedes for cost:** the D11/D16 concern that Tier-1 was not actually cheap.

## 1. Goal
Produce the real **pano→room accuracy number** for PanoPin on the S3DIS Area_3 **in-frame** panos, by
(a) making the two-tier funnel's Tier-1 genuinely cheap so a full 76×21 evaluation runs in minutes not
~10 h, and (b) pinning determinism so thin-margin selections are trustworthy. Success = a deterministic,
honestly-reported room accuracy that clears the 5.9 % random baseline (project T3), reported **in-frame
(76 panos)** and **all (85)** separately.

## 2. Background — what the recon established (2026-07-09)
Profiling: one `localize_pair` on CPU ≈ 25 s, dominated by `make_score_map_2d` (~6.6 s) + `make_score_map_3d`
(~13.4 s) = the "inlier detection". Recon of `third_party/cpo/` found:
- **The score maps are only multiplicative WEIGHTS.** `score_map_2d` → `img_weight`, `score_map_3d` →
  `pcd_weight`. They are consumed only inside `histogram_pose_search` (patch scoring, `utils.py:1491-1496`)
  and `refine_pose_sampling_loss` loss (`sampling_loss.py:261-271`). **Neither is structurally required**
  to produce candidate poses or a comparable loss.
- `histogram_pose_search` runs correctly with `img_weight=None` (falls back to latitude `sin_weight`,
  `utils.py:1491-1492`), so it needs **no** score maps.
- A **single-forward** `sampling_loss` (`sampling_loss.py:147-210`) computes an unweighted mean colour
  residual at a fixed pose with **no Adam, no weights** — the natural cheap per-room ranking score. Lower
  = better match; uniform/None weights give a *cleaner* cross-room scale (loss denominator is `mask.sum()`
  regardless of weights).
- Pose-pool size `9360 = 45 trans × 208 rot`; rotations `8/8/8 → 208`, `4/4/4 → 24` (measured). Shrinking
  the pool + splits cuts cost 1–2 orders of magnitude.
- **Determinism is ~free on CPU**: the path consumes **no torch RNG** (`torch.manual_seed` is a no-op).
  The only RNG is `read_txt_pcd`'s `np.random.permutation`, active only when `sample_rate>1` (seedable).
  The one op without a formal deterministic guarantee is `make_pano`'s `index_put_` with duplicate pixel
  indices (`utils.py:233-241`); observed ±0.02 loss wobble is most likely multi-threaded float reduction
  order → test `torch.set_num_threads(1)` + `np.random.seed`.

## 3. Architecture — four small pieces
Deterministic/training-free throughout (D1); pure CPO-primitive composition, no upstream edits (D9);
solver reads only the anonymized manifest (D5); harness stays stdlib-only (D6).

### 3.1 `score_room_cheap(cfg, pano_path, cloud_path) -> (t, R, loss)`  — new, in `src/panopin/cpo_adapter.py`
A stripped `localize_pair` for Tier-1:
1. `read_txt_pcd(cloud, sample_rate)`, load + resize pano (2048×1024) — same as `localize_pair`.
2. Build `init_dict = get_init_dict_cpo(cfg)` with a **small pose pool** via cfg overrides:
   `num_yaw=num_pitch=num_roll=4` (→ 24 rot), reduced `num_trans` (e.g. 10), `num_split_h=8,num_split_w=16`.
3. `input_trans, input_rot = histogram_pose_search(img, xyz, rgb, trans, rot, top_k_candidate=1,
   num_split_h, num_split_w, img_weight=None, sin_hist)` → best coarse pose. **No `make_score_map_2d/3d`.**
4. Score that single pose with a **single-forward** `sampling_loss(...)` (unweighted) → `loss`.
5. Return `(t, R, loss)` — same contract as `localize_pair`.
Implementation note: use whichever CPO primitive composes cleanest for step 3–4 — `histogram_pose_search` +
`sampling_loss`, or `sampling_loss_pose_search` (`utils.py:1566`) which searches a pose pool by that loss
directly. Decide during implementation by reading the exact signatures; keep the `(t,R,loss)` contract.

### 3.2 `select_room` rewire — `src/panopin/select_room.py`
Tier-1 loop calls `score_room_cheap(cfg1, ...)` for every candidate room (cfg1 = small-pool cfg); keep the
top-k lowest-loss survivors. Tier-2 loop keeps the existing full `localize_pair(cfg2, ...)` (inlier maps +
Adam, `num_iter=100`) on survivors → final min-loss `RoomResult`. `RoomResult` interface unchanged, so the
funnel tests' INTENT (pick min-loss, refine only top-k, clamp, empty→None) is preserved — but the mock
setup in `tests/test_select_room.py` must be reworked to mock **both** `score_room_cheap` (Tier-1) and
`localize_pair` (Tier-2) separately, instead of keying on `cfg.num_iter` as it does now.

### 3.3 Determinism — solver entry + a check
In `solve.py` (and smoke scripts): `torch.set_num_threads(1)` and `np.random.seed(seed)` at startup.
New `smoke/check_determinism.py`: run one real pair twice and assert **identical** loss. If threads-1 does
not fully pin it, record the residual variance and rely on the (wider, full-settings) Tier-2 margin —
but only after measuring. This is a verification gate, not an assumption.

### 3.4 Eval — `solve.py` (plan Task 6) + `eval/`
`solve.py` reads the anonymized manifest, runs `select_room` per pano, writes predictions JSON per
`eval/PREDICTIONS_SCHEMA.md`. Run over the 76 in-frame panos; score with `eval/score.py`. Report room
accuracy **in-frame (76)** and **all (85)**; list the 9 out-of-frame panos (D17) as excluded, not failures.

## 4. Data flow
manifest (uuid→pano, room→cloud) → for each pano: Tier-1 `score_room_cheap` over all 21 rooms → top-k →
Tier-2 `localize_pair` over top-k → min-loss room + coarse pose → predictions JSON → `eval/score.py` →
accuracy (in-frame / all).

## 5. Validation gates (before trusting the number)
- **G1 Tier-1 recall** — on ~5 labeled in-frame panos, the correct room must appear in Tier-1's top-k
  (default k=5). If not, widen the pool (more rot/trans) until it does. This is approach-B's main risk.
- **G2 Tier-1↔Tier-2 agreement** — cheap Tier-1's ranking broadly tracks full Tier-2 on office_3 (robust)
  and office_5 (thin). Sanity, not exactness.
- **G3 determinism** — `check_determinism.py` passes (identical loss over two runs) or the residual is
  measured and documented.
- **G4 cost** — measured Tier-1 wall-clock < ~2 s/room; full-run projection recorded.

## 6. Scope / YAGNI
IN: cheap Tier-1, determinism pinning, full in-frame eval + honest split reporting. OUT (future work):
out-of-frame recovery (the 9 panos, D17), GPU port, coarse-pose→FGPL hand-off (project T4), accuracy
tuning beyond clearing the baseline, all-areas generalization (T5).

## 7. Risks
- **R1 (main):** cheap Tier-1 too lossy → correct room drops out of top-k (G1 catches; mitigation = widen
  pool, still far cheaper than inlier detection).
- **R2:** determinism not fully pinned by threads-1 (`index_put_` order). Mitigation: measure; lean on
  Tier-2 full-settings margin; worst case document residual noise with the number.
- **R3:** `sampling_loss` single-forward signature differs from assumption → adapt to the real signature
  during implementation (read `sampling_loss.py`), keep the `(t,R,loss)` contract.
- **R4:** thin margins (office_5, +3.5 %) may still misassign some panos even deterministically — that is a
  real accuracy finding to report, not a bug.

## 8. References
- Recon report (this session) on `third_party/cpo/`: `utils.py:1152-1235,1268-1384,1422-1505`,
  `sampling_loss.py:13-144,147-210,226-280`, `dict_utils.py`, `config/stanford_cpo.ini`.
- DECISIONS D1, D5, D6, D9, D11, D14, D16, D17. Plan: `docs/plans/2026-07-08-coarse-room-cpo-v0.md`
  (Task 6/7 consume this). Harness: `eval/` + `eval/PREDICTIONS_SCHEMA.md`.
