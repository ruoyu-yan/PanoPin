# PanoPin — Handover (end of 2026-07-09)

Read this first, then `docs/PROGRESS.md` (current-state block) and `docs/DECISIONS.md` **D21–D24**.

## TL;DR
CPO render-and-compare **works but is slow and imperfect**. On the whole area (23 rooms) it needs
**calibration** to be usable, and even then ~1/3 of panos are unmatchable. **v1 is done and committed**
(calibrated assignment + a confidence gate that abstains). Two threads are open: (a) push the accuracy
ceiling with a robust loss (branch ready), (b) wire v1 into `solve.py` + validate beyond n=12.

## The numbers that matter (all honest / fair, n=12 whole-area sample)
- Raw whole-area recall@1 ≈ **33%** — a fixed set of small/corridor "loss-sink" rooms (hallway_5/6/4,
  WCs, storage) top-rank every pano (D21).
- **v1 minmax calibration → recall@1 ≈ 58%.** Confidence gate is a precision/coverage tradeoff:
  `-0.3` → 100% precision @ 33% coverage; `-0.19` → 86% @ 58% (D24).
- Hard floor: ~1/3 of panos **fail to self-localize** (own-room loss ~0.25 vs ~0.08) — window /
  blank-wall / occlusion-dominated views (D23). The gate correctly flags them. Not fixable by calibration.
- Prototype scale (3-room scenes, each pano vs its 3 rooms): recall@1 = 78%, ~1–2 min/pano.
- ⚠️ **D24 correction:** an earlier "clean 100%/58% confidence" result was a **GT leak** (baseline excluded
  rows by true-room label). Fair calibration baselines must be **leave-one-out on the PANO ONLY**.

## Repo state
- Branches (nothing pushed to GitHub; `main` = skeleton only):
  - **`feat/coarse-room-cpo`** — all work through v1 (HEAD 24b3c4a). ← currently checked out.
  - **`feat/robust-loss-ceiling`** — created for the ceiling-push, currently == coarse-room-cpo (empty).
- v1 code: `src/panopin/calibrate.py` (fair minmax + confidence gate), `tests/test_calibrate.py` (3 pass),
  `smoke/score_v1.py` (scores a loss matrix), `smoke/analyze_calibration.py` (fair formula sweep).
- **Reusable artifact:** `runs/calib_matrix.json` — the 12-pano × 23-room loss matrix. Re-analyze
  calibration/confidence formulas offline (no GPU) with `smoke/analyze_calibration.py` / `score_v1.py`.
- Envs: **`panopin`** (CPU, torch 1.10+cpu — reproducible, D10) for tests/harness; **`panopin-gpu`**
  (torch 2.0.1+cu118 + torch_scatter + CPO deps — RTX 4060, ~2× faster) for CPO runs. **Do NOT use scan_env.**

## Cost model (so you size runs correctly)
One `localize_pair` (1 pano vs 1 room) ≈ **25 s** on GPU (CPU ~40 s). Assigning a pano = 25 s × (num
candidate rooms). Prototype (3–5 rooms) ≈ 1–2 min/pano; whole-area (23 rooms) ≈ ~7 min/pano; full 85×23 ≈
~13 h. Speed is intrinsic (D20: CPO's inlier loop + Adam are launch-bound; GPU only ~2×). Don't try to
"cheapen" the per-room CPO — the cheap Tier-1 path is a proven dead end (D18/D19, Plan 2 abandoned).

## NEXT — two open threads
1. **Ceiling-push (branch `feat/robust-loss-ceiling`).** CPO's loss is a *mean* color residual; a minority
   of un-matchable pixels (window/occlusion) can dominate. Try a **robust loss** (trimmed-mean / median /
   low-percentile of per-point residuals) to rescue *minority*-window panos (test on office_7; won't help
   window-*dominated* office_8). Note: needs a GPU re-run capturing poses + per-point residuals (don't edit
   `third_party/cpo` — compute the robust statistic in our adapter at the returned pose). First test a few
   panos before scaling.
2. **Wire v1 into `solve.py`** (manifest → per-pano×room loss matrix → `calibrate.assign` → predictions with
   confidence/flag → `eval/score.py`). The matrix is the expensive part; support loading a cached matrix.
   Then **validate on more than n=12 panos** — the 58% / gate numbers are noisy.

## Gotchas / rules
- Fairness (D5): `src/panopin/*` reads only the manifest — no GT, no room labels. GT only in `smoke/`.
- D9: compose CPO primitives; do NOT edit `third_party/cpo/`.
- Calibration baselines: leave-one-out on the pano only (never exclude by room label — that's the D24 leak).
- CPO not bit-reproducible on CPU (~±0.02 loss); `panopin.determinism.pin()` sets single-thread + seed.
- Candidates = individual S3DIS room segments `Stanford3dDataset_v1.2/Area_3/<room>/<room>.txt`.
- Out-of-frame panos (D17): 9/85 have GT camera outside their room cloud → unlocalizable; report in-frame
  (76) vs all (85).

## Session ritual (start of tomorrow)
`cat docs/PROGRESS.md docs/DECISIONS.md docs/tasks.json`; `git log --oneline -15`;
`conda run -n panopin python -m pytest tests/ -q` (or `python tests/test_harness.py`); then pick a thread.
