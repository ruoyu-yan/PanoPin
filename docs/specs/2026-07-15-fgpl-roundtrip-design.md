# Design — PanoPin↔FGPL validation round-trip (spec §8 fast-follow)

**Date:** 2026-07-15 · **Branch:** new `feat/fgpl-roundtrip` (off `main` @ 76dea5c) ·
**Builds on:** D34 (`src/panopin/fgpl_export.py` — the gated per-pano seed writer), D25 (the
`experiments/fgpl_seed/` ablation harness that already runs FGPL end-to-end; color POSITION seed ≈ oracle
when the room is right; FGPL rotation convention `C=[[0,0,1],[-1,0,0],[0,-1,0]]`), D30/D32/D33 (low-pct
score, room-anchored coverage). **Consumes:** FGPL `scan2measure-webframework/src/pose_estimation/multiroom_pose_estimation.py`.
**Deliverable:** `experiments/fgpl_seed/roundtrip.py` + `experiments/fgpl_seed/ROUNDTRIP_RESULTS.md`; a
decision note in `docs/DECISIONS.md`.

## 1. Goal

Run the D34 `fgpl_export` output through FGPL's estimator end-to-end on real Area_3 data and measure
refined-pose accuracy vs S3DIS GT — proving the two halves run together and that the deployment claim
(**every room gets an accurately-localized pano**) survives FGPL's fine refinement, not just PanoPin's
own color scoring. This is the "wire `seed_rooms` into the FGPL runner" step from D33/D34 §8, executed as
a measured experiment (validation, not production/Electron wiring — that is a separate later task).

**Success:** (a) `fgpl_export` drives FGPL with no errors and produces a `camera_pose.json` for every
admitted pano; (b) on correct-room panos the refined **translation median is close to the cached oracle**
(D25 reproduces for the better low-pct seed); (c) **per-room coverage = 6/6** — every subset room ends up
with ≥1 admitted pano whose refined FGPL pose lands in the correct room.

## 2. What already exists (verified — the harness is built)

The D25 ablation (`experiments/fgpl_seed/`) already runs FGPL end-to-end on scene `area3_seed_ablation`
(the 6-room subset office_1/4/5/6/7 + hallway_3, 12 in-frame panos). Reused verbatim:

- **FGPL-estimator data (built):** `work/linemap/3d_line_map.pkl`, per-pano 2D features `work/features/<uuid>_v2/`,
  merged cloud `work/clouds/area3_seed_ablation.ply`.
- **`seed_and_config.py`:** `write_identity_metadata()` (raw==aligned frame), `write_config(arm, rows, seed, line_map, md, feat, panos)`
  (pano_names = `[r["pano_name"] for r in rows]`), `room_centroids(rows)`.
- **`run_arm.run_arm(cfg, rows)`:** subprocess → FGPL estimator (in `panopin-gpu`) → per-pano `camera_pose.json`
  → `{pano: {translation, rotation}|None}`. ~330–390 s/pano (XDF search is CPU-bound).
- **`score.score_arm(poses, gt, rows, centroids)`:** translation/rotation errors + wrong-room rate vs S3DIS GT.
- **`run_all.py`:** the orchestration template — `rows = subset.build_subset()`, `gt = s3dis_gt.load_gt("Area_3", …)`,
  `cpo = cpo_cache.json`, rotation conversion `_fgpl_rot_to_cw(R) = R.T @ C`, comparison table.
- **Cached inputs:** `work/seeds/residuals.json` (per-(pano,room) 101-percentile grids), `work/seeds/cpo_cache.json`
  (`cpo[pano] = {t, R, loss, room, poses:{room:{t,R}}}`).
- **Cached reference poses (no re-run):** `work/poses/oracle/` (GT-seed upper bound) and `work/poses/p1/`
  (raw min-loss all-pano, the D25 baseline).

Scene choice is **forced by data availability**: largeval's 8 rooms have CPO scores but no line-map/features,
so the round-trip must use this 6-room subset. It is a valid multi-room all-covered demo.

## 3. The one new thing — a `fgpl_export` arm

A new arm identical to the ablation arms except the seed comes from the D34 module instead of `sc.write_seed`:

1. `grids = json.load(work/seeds/residuals.json)`; `scores = robust_score.low_percentile_scores(grids)` (D30, q=20).
2. `poses = {p: {r: (v["t"], v["R"]) for r, v in cpo[p]["poses"].items()} for p in cpo}` (adapt the cache's
   `{"t","R"}` dicts to the `(t, R)` tuples `build_matches` expects).
3. `room_order = sorted({r["room"] for r in rows})`.
4. `admitted = fgpl_export.export_alignment(scores, poses, room_order, md_path, work/seeds/fgpl_export.json, tau=0.10)`
   — the real deployment path (per-pano gate + coverage backstop + frame conversion; identity metadata ⇒ the
   conversion is a validated no-op here, exercising the guard's happy path).
5. `admitted_rows = [r for r in rows if r["pano_name"] in admitted]`.
6. `cfg = sc.write_config("fgpl_export", admitted_rows, seed_path, line_map, md, feat, panos)` — so
   `cfg["pano_names"] == admitted` (the deployment contract; each seeded pano has a match, no KeyError).
7. `poses = run_arm.run_arm(cfg, admitted_rows)` → refined FGPL poses (~10 panos × ~6 min ≈ **~60 min GPU**,
   `panopin-gpu`). Convert rotation via `_fgpl_rot_to_cw`.
8. `score.score_arm(poses_cw, gt, admitted_rows, cents)`.

**Deployment-faithful gating:** FGPL runs on only the admitted (~10) panos — the weak-lock ~2 are abstained,
never seeded. Because the Voronoi partition (`get_local_mask`) is built over the *seeded set*, this is the
genuine deployment behavior and cannot be derived by filtering a larger run (see §6 non-goals).

## 4. Metrics + deliverable

`ROUNDTRIP_RESULTS.md` + `work/results/roundtrip.json` + a printed table:

| arm | n_localized | per-room coverage | trans median | trans median (right-room) | rot median | wrong-room |
|---|---|---|---|---|---|---|
| **fgpl_export** (gated, NEW run) | ~10/12 | **6/6 target** | measure | measure | measure | measure |
| oracle (cached, GT-seed) | 12/12 | 6/6 | reference | reference | reference | ~0 |
| p1 (cached, raw min-loss) | 12/12 | reference | reference | reference | reference | ~33% (D25) |

- **Per-room coverage** (the headline deployment metric): for each of the 6 subset rooms, count admitted
  panos assigned to it whose **refined** FGPL pose has nearest-centroid == that room; `covered` = rooms with
  ≥1. Target 6/6.
- **Right-room translation slice** (the load-bearing D25 number): median translation error over admitted
  panos whose fgpl_export seed room == true room — should be ≈ oracle.
- oracle/p1 are scored on the admitted pano set from their cached poses for **reference context** (see the
  caveat in §6: they ran with a 12-pano Voronoi, so this is directional, not a controlled ablation).

## 5. New code (small) + fairness

`experiments/fgpl_seed/roundtrip.py` — mirrors `run_all.py`'s setup, runs the single new arm (§3), computes
the metrics (§4), loads and scores the cached oracle/p1 on the admitted set, prints the table, writes the
results file + `ROUNDTRIP_RESULTS.md`. Small helpers may live inline or in `seed_and_config`.

**Fairness (D5):** the seed is built only from cached color scores/poses — no GT. GT enters only in `score_arm`
and the coverage/centroid computation (scoring is allowed GT). `fgpl_export` itself never reads GT.

## 6. Verification, non-goals, caveats

**Verification:**
- **GPU-free pre-check (before the ~60-min run):** build `work/seeds/fgpl_export.json`, load it through FGPL's
  own `load_panorama_positions` (identity metadata), assert the recovered positions equal the cached `t[:2]`
  for every admitted pano and that `admitted` covers every room. Reuses the D34 smoke pattern; catches a
  malformed seed before paying for the estimator run.
- The FGPL run itself is the integration proof — show the scored table.

**Non-goals:**
- **Ungated variant deferred.** Seeding all 12 panos would need a *full second* FGPL run (the Voronoi
  partition is over the seeded set, so a gated run's results cannot be filtered out of an all-12 run), and
  the cached `p1` arm already illustrates all-12 non-gated seeding. Optional follow-up, not v1.
- Not largeval's 8 rooms (no FGPL-estimator data). Not production/Electron pipeline wiring (Option B, later).
- Not re-running oracle/p1 (use cached poses). Not exercising R≠I frame conversion (identity metadata here;
  D34 unit tests already cover R≠I with the round-trip guard).

**Caveats to state in the results:** (a) oracle/p1 cached poses ran with a 12-pano Voronoi vs the gated arm's
~10-pano Voronoi — reference context, not a controlled comparison; (b) Area_3 subset only (cross-area open);
(c) the gate `tau=0.10` is the D34 Area_3-tuned value — this round-trip also functions as the deferred D34
"offline precision check," now measured through FGPL rather than on the score matrix alone.

## 7. Success criterion

`roundtrip.py` runs green, the GPU-free pre-check passes, and `ROUNDTRIP_RESULTS.md` reports: fgpl_export
drove FGPL to a `camera_pose.json` for every admitted pano; per-room coverage (target 6/6); right-room
translation median vs the cached oracle; and the wrong-room rate. This closes spec §8's plumbing question —
PanoPin and FGPL run together, and the deployment claim is measured on real refined poses.
