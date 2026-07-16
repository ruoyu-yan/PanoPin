# PanoPin — Handover

> **This file was written 2026-07-10 and its instructions are STALE.** Its "next step" (wire v1
> calibration into the seed) was done and **refuted** (D26). Do **not** follow it. The current
> state lives in `docs/PROGRESS.md` (top block) + `docs/DECISIONS.md` (**D36** newest). The
> 2026-07-10 content is kept below only as history.

**Read order (2026-07-16):** `docs/PROGRESS.md` top block -> `docs/DECISIONS.md` **D36** (today),
then **D30-D35** (the deployment line) -> `docs/tasks.json` (**T9**, **T10** are the open work) ->
`experiments/fgpl_seed/MANHATTAN_PANOPIN_VS_FGPL.md` (the presentation doc).

## What PanoPin is
A fast, deterministic **coarse panorama->room seed** (which room + rough position) handed to
**FGPL** (the fine geometric localizer in `/home/ruoyu/scan2measure-webframework/`) so pose
estimation works on large, repetitive multi-room buildings where the thesis' shape-based jigsaw
fails. Method = **CPO render-and-compare on COLOR** (not shape).

## The one-line result (2026-07-16, D36)
On a strictly-Manhattan 5-room / 22-pano Area_3 scene, same map and estimator throughout:
**FGPL alone = 1/5 rooms, 82% wrong-room, 13.27 m median. +PanoPin = 5/5 rooms, 18%, 0.084 m.**
The rooms are near-identical in shape, so FGPL's geometry cannot tell them apart and colour can.
That is the whole argument.

## Facts a new session MUST know
- **Envs (never mix):** FGPL estimator + CPO -> `panopin-gpu`; FGPL build tools (baker/cluster/
  features) -> `scan_env`; PanoPin tests/harness -> `panopin`.
- **The hand-off is per-PANO** (D34): FGPL localizes each pano from its own seed, and consumes only
  `camera_position` from `demo6_alignment.json` (`room_label`/`rotation_deg` are ignored -- room_label
  appears only in a `print`). Seeding one pano per room poses ONE pano, not the room's others.
- **The seeds ARE the Voronoi partition.** More seeds -> smaller cells -> less 3D geometry per pano
  -> rotation aliasing. But the seed also **pins position**: removing the partition entirely is
  WORSE (`PERROOM_RESULTS.md` -- `fafa0629` locked rotation at 0.7 deg and landed 8.8 m down a
  hallway). Both refuted extremes are documented in D36; the untested middle is **T9**.
- **Absolute thresholds do not transfer.** `tau=0.10` (D34) admits 21/22 including 3 wrong-room panos
  on the Manhattan pool. Prefer threshold-free (room-anchored) or within-pano relative signals (the
  rotation margin, T10).
- **Read bimodal medians with care.** Poses either lock (~1 deg) or flip (~90-180 deg), nothing
  between; a median only reports which side most panos are on, NOT the error size.
- **Rotation flips are FGPL's, not PanoPin's** -- they happen with a GT seed too.
- `work/` is gitignored (514 MB of caches); committed numbers live in the `*_RESULTS.md` files.

---

# HISTORICAL — the 2026-07-10 handover (superseded; read for context only)

**Read order:** this file → `docs/PROGRESS.md` (top "Current state" block) → `docs/DECISIONS.md` **D25**
(today) + **D14–D24** (v1) → `experiments/fgpl_seed/RESULTS.md` (today's numbers) →
`.superpowers/sdd/progress.md` (today's task-by-task ledger).

## What PanoPin is
A fast, deterministic **coarse panorama→room seed** (which room + rough position) handed to **FGPL** (the
fine geometric localizer in `/home/ruoyu/scan2measure-webframework/`) so pose estimation works on large,
repetitive multi-room buildings where the thesis' shape-based jigsaw fails. Method = **CPO render-and-compare
on COLOR** (not shape). Success metric = pano→room accuracy on S3DIS Area_3 (random baseline 5.9%).

## Three workstreams and their state
1. **v1 (DONE, 2026-07-09)** — fair minmax calibration + confidence gate over CPO losses. Whole-area
   recall@1 33%→**58%** (n=12). On branch `feat/coarse-room-cpo`. Two threads still OPEN (see bottom).
2. **FGPL seed ablation (DONE TODAY, 2026-07-10)** — the main event; branch `feat/fgpl-seed-ablation`.
   Details below. **This is the active thread with the clearest next step.**
3. (background) ceiling-push robust loss — branch `feat/robust-loss-ceiling`, still empty.

---

## TODAY — PanoPin → FGPL seed ablation (branch `feat/fgpl-seed-ablation`, PUSHED)

**Question answered:** can PanoPin's color output replace FGPL's slow SAM3+IoU jigsaw and improve FGPL's
pose accuracy on same-shape multi-room S3DIS? **YES for the position seed.**

**Result (n=12, 6-room subset office_1/4/5/6/7 + hallway_3):**
- **Where CPO picks the right room (8/12), the color POSITION seed == ground truth:** P1 median **0.762 m**
  vs oracle **0.725 m** — FGPL localizes identically. The whole P1↔oracle gap on the full set (0.946 vs
  0.789) is the **4 wrong-room misses** (all → hallway_3 loss-sink, ~22 m each).
- **⇒ the color position seed WORKS; the bottleneck is ROOM RECALL** (67% here, raw min-loss).
- **P2** (translation-grid narrowing) = **no effect** (Voronoi already constrains). **P3** (CPO rotation
  prior) = **hurts** (CPO rotation is convention-broken). Both should be dropped.
- Whole-branch review = **SOUND** (no GT leak; scoring/frame correct).

### THE NEXT STEP (do this first tomorrow — ~1 hr, harness already exists)
**Wire v1 calibration into the room assignment in `experiments/fgpl_seed/cpo_seeds.py`.** It currently
assigns `room = min-loss room` (raw → 67% recall). v1's calibration lifts recall 33→58% and flags the
loss-sink misses — i.e. it directly targets the 4 failures that are the entire gap to the oracle.
Concretely:
1. In `cpo_seeds.py`, cache `(t, R, loss)` for **all 6 rooms** per pano (today it keeps only the min-loss
   winner's `t,R` + a `per_room` loss dict).
2. After the loop, build `loss_matrix = {pano: per_room}` and call `calibrate.assign(loss_matrix)`
   (`src/panopin/calibrate.py`) → calibrated room per pano.
3. Use the **calibrated** room's cached `(t,R)` as the P1 seed (instead of raw min-loss).
4. Re-run `run_all` (already computes the right-room split). Expect wrong-room rate to drop from 33%.
5. (Optional) apply the **confidence gate** to ABSTAIN on flagged panos — hand FGPL nothing rather than a
   wrong seed (the deployed pipeline can then skip or fall back for those).

### Reproduce (all commands from `/home/ruoyu/PanoPin`)
```
conda run -n panopin     python -m experiments.fgpl_seed.gate_oracle    # Phase-0 oracle gate (~10 min)
conda run -n panopin-gpu python -m experiments.fgpl_seed.cpo_seeds       # CPO seeds -> work/seeds/cpo_cache.json (~30 min)
conda run -n panopin     python -m experiments.fgpl_seed.run_all         # 5-arm ablation -> work/results/ablation.json (~40 min)
```
Line map + features are already built under `work/` (reused unless `--rebuild`). `work/` is gitignored;
committed numbers are in `RESULTS.md`.

### Key facts the next session MUST know (FGPL side)
- **Envs (never mix):** FGPL **estimator + CPO** → `panopin-gpu` (`paths.ESTIMATOR_ENV`, torch cu118,
  Ada-native). FGPL **build tools** (baker/cluster/features) → `scan_env`. PanoPin tests/harness → `panopin`.
  scan_env's cu116 predates Ada (RTX 4060) → CPU fallback; that's why we moved the estimator to panopin-gpu.
- **Estimator cost:** ~8 min/arm for 12 panos, because the **multi-pano Voronoi shrinks each pano's cell** →
  fewer translation candidates (~8–94 s/pano). A SINGLE-pano run searches the whole map (~325 s) and can
  land in the wrong same-shape room — so never gate/measure on a single pano. The XDF search is **CPU-bound
  numpy** (GPU only ~1.2×).
- **Raw S3DIS frame:** we run FGPL entirely in the raw S3DIS frame via an **identity `rotation_matrix`** in
  `metadata.json` → `aligned_meters_to_raw_3d` is a pass-through ([x,y]→[x,y,0]). No floorplan/SAM3/density
  front-end needed. **Gotcha:** the estimator does `density_img.shape` unconditionally
  (`multiroom_pose_estimation.py:259`) on a cv2-loaded density image — we emit a blank 256×256 `density.png`
  (viz-only; `seed_and_config._ensure_density_png`). Without it the estimator crashes before pose.
- **FGPL handoff is POSITIONAL:** the estimator consumes only `camera_position` from the seed
  (`demo6_alignment.json`); `room_label`/`rotation_deg` are ignored; region binding is a Voronoi over seed
  positions. So the seed's job is region selection — which is why room recall is the bottleneck.
- **FGPL rotation convention** (D25): FGPL outputs `Rp = C @ R_wc`, `C = [[0,0,1],[-1,0,0],[0,-1,0]]`
  (equirect signed-perm, same as Point_360 solved). `Rp.T @ C` = camera→world; 0.4–0.5° on precise panos.
  FGPL rotation is ALSO ~90° Manhattan-aliased even with a good seed. **CPO's rotation is unusable** (~120°
  off GT in both conventions) — that's why P3 fails; don't seed FGPL rotation from CPO.
- **The FGPL narrowing edit** (P2/P3, flag-gated default-off) lives on scan2measure branch
  `feat/panopin-seed-narrowing` (pushed) + as `experiments/fgpl_seed/patches/fgpl_narrowing.patch`.
  scan2measure is back on `main`; re-apply via `git checkout feat/panopin-seed-narrowing` or the patch to
  re-run P2/P3.
- **`experiments/fgpl_seed/` map:** `paths.py` (env/path consts) · `subset.py` (6 rooms→12 panos) ·
  `build_ply/build_linemap/build_features` (FGPL artifacts) · `seed_and_config.py` (identity metadata +
  per-arm seed/config) · `cpo_seeds.py` (CPO room assign + cache) · `run_arm.py` (estimator wrapper) ·
  `score.py` (vs GT via `eval/metrics`) · `gate_oracle.py` (Phase-0 gate) · `run_all.py` (5-arm ablation).

---

## v1 — two threads still OPEN (branch `feat/coarse-room-cpo`, PUSHED)
1. **Wire v1 into `solve.py`** (manifest → per-pano×room loss matrix → `calibrate.assign` → predictions with
   confidence/flag → `eval/score.py`); validate beyond n=12. (Overlaps with the FGPL next-step above —
   both need the calibrated assignment.)
2. **Ceiling-push (branch `feat/robust-loss-ceiling`, empty):** robust/masked loss (trimmed-mean / median /
   low-percentile per-point residuals) to rescue minority-window panos (test office_7; won't help
   window-dominated office_8). Needs a GPU re-run capturing per-point residuals; compute the robust stat in
   our adapter at the returned pose — do NOT edit `third_party/cpo/`.
- v1 code: `src/panopin/calibrate.py` (+ `tests/test_calibrate.py`), `smoke/{score_v1,analyze_calibration}.py`.
- Reusable: `runs/calib_matrix.json` (12×23 loss matrix — offline formula analysis, no GPU).

## Repo / GitHub state (as of end 2026-07-10 — now PUSHED)
- **PanoPin** (github.com/ruoyu-yan/PanoPin): `main` = skeleton; `feat/coarse-room-cpo` = v1 (pushed);
  **`feat/fgpl-seed-ablation` = today, HEAD `37cdabc`, pushed**; `feat/robust-loss-ceiling` = empty.
- **scan2measure-webframework** (github.com/ruoyu-yan/scan2measure-webframework): on `main`;
  `feat/panopin-seed-narrowing` (`56e6519`, pushed) holds the flag-gated FGPL edit. Its `main` has the
  user's own unrelated pending work (README, submodules, thesis-defense docs) — leave those alone.

## Standing gotchas / rules
- **Fairness (D5):** `src/panopin/*` reads only the manifest — no GT. In the FGPL experiment, GT is used
  ONLY for the oracle/wrong-room reference arms + scoring, all under `experiments/` (never in `src/panopin`).
- **D9:** compose CPO primitives; do NOT edit `third_party/cpo/`.
- **`panopin` package** is not pip-installed — scripts add `src/` to `sys.path` (see `cpo_seeds.py` header /
  the smoke scripts). A `-m experiments.fgpl_seed.X` run from repo root also needs `src/` on the path for
  `import panopin`.
- CPO not bit-reproducible on CPU (~±0.02 loss); `panopin.determinism.pin()` sets single-thread + seed.
- Out-of-frame panos (D17): 9/85 have GT camera outside their room cloud → unlocalizable; the FGPL subset
  uses in-frame panos only.

## Session ritual (start of tomorrow)
`cat docs/PROGRESS.md docs/DECISIONS.md docs/tasks.json`; `git log --oneline -15`;
`conda run -n panopin python -m pytest tests/ experiments/fgpl_seed/tests/ -q`; then pick a thread
(recommended: the FGPL next-step — wire v1 calibration into `cpo_seeds`).
