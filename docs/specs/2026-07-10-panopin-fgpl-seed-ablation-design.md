# PanoPin → FGPL seed ablation — design

**Date:** 2026-07-10 · **Status:** design (approved for planning) · **Repo:** PanoPin (feeds FGPL).
**Supersedes/extends:** `docs/specs/2026-07-09-fgpl-handoff-scope.md` (T4). This design commits to
*replacing* the jigsaw (not just handing off to it) and to an experiment that measures FGPL's final
pose accuracy under PanoPin seeding.

---

## 1. Motivation

The Scan2BIM pipeline's fine pose estimator is a modified **FGPL** (fully geometric, line-based) in
`/home/ruoyu/scan2measure-webframework/`. Its **coarse** step today is a "jigsaw": SAM3 room
segmentation + IoU polygon matching (`src/floorplan/align_polygons_demo6.py`). Two problems, both of
which PanoPin's color-based room assignment (v1) addresses:

- **Speed.** In the shipped TMB validation run (`data/validation/subset1 - corridor + office/results.md`)
  the jigsaw stack is **~300 s of the 406.8 s pipeline** (Polygon Matching alone = 191.8 s; Pose
  Estimation = 54.6 s). Its polygon matching is *combinatorial* (M^N assignments,
  `align_polygons_demo6.py:142-147`) so it worsens on larger scenes; PanoPin/CPO is linear in
  panos × rooms.
- **Accuracy on same-shape rooms.** The jigsaw decides the room by **shape IoU** with no appearance
  prior (`align_polygons_demo6.py:521`); on congruent rooms the confidence margin collapses (the
  shipped TMB example won by only 0.16) and a geometrically-identical **wrong room** can win. FGPL's
  catastrophic failure mode is **wrong room**, not imprecise position — a wrong room means the entire
  spatial partition (below) is wrong, so the fine pose is wrong regardless of geometry quality.

PanoPin v1 assigns the room by **color render-and-compare** (CPO), which disambiguates exactly the
same-shape case shape cannot, and adds a **confidence gate** the jigsaw lacks (it can abstain instead
of silently seeding a wrong room).

## 2. Objective & hypothesis

**Objective.** Replace the jigsaw with PanoPin's color-based coarse seed and measure whether — and
with how much guidance — it makes FGPL's **final pose** more accurate on a multi-room scene containing
same-shape rooms (S3DIS Area_3). **FGPL's geometric loss is never modified**; PanoPin only *seeds and
narrows* FGPL's search.

**Hypothesis.** On same-shape rooms, a color-derived seed lands FGPL in the correct room where
shape-based coarse selection fails; and increasing guidance (position → +narrowed grid → +rotation
prior) moves final-pose accuracy toward the oracle (GT-seed) ceiling.

**Non-goals.** No change to FGPL's cost function; no colorization-quality metric (Delta-E) in this
experiment; no whole-Area_3 run (first cut is a subset); no literal jigsaw baseline on S3DIS (see §7).

## 3. How FGPL consumes a seed (established facts, with citations)

Verified by code recon of `/home/ruoyu/scan2measure-webframework/` (2026-07-10):

- **The estimator is config-driven.** `src/pose_estimation/multiroom_pose_estimation.py` `main()`
  (`:149-167`) overrides scene name + all paths from `--config <json>` (`point_cloud_name`,
  `pano_names`, `pkl_3d_path`, `alignment_path`, `metadata_path`, `point_cloud_path`,
  `features_2d_dir`, `pano_dir`, `output_dir`, `use_local_filtering`). **No code edits needed** to
  point it at Area_3. (The single-room `pose_estimation_pipeline.py` is hardcoded — do not use it.)
- **The seed interface is positional.** The fine step reads `demo6_alignment.json` via
  `load_panorama_positions` (`:104-124`) and consumes **only `camera_position`** numerically;
  `room_label` is merely printed (`:121`), `rotation_deg`/`room_idx`/`score` are ignored. The room→
  region binding is re-derived by a **Voronoi partition** over the shared 3D line map
  (`compute_voronoi_assignment :127-131`, `get_local_mask :134-143`, applied `:225-232`, `:307-326`),
  with `OVERLAP_MARGIN = 2.0 m` (`:66`) and a documented ~3 m position tolerance.
- **Frame conversion is bypassable.** `aligned_meters_to_raw_3d` (`:93-101`) applies
  `metadata['rotation_matrix'].T` to convert the jigsaw's aligned-meters position into the raw-3D
  frame of the line map. If PanoPin supplies a **raw-3D** `camera_position`, this must be made a
  pass-through (identity), which removes the dependency on `metadata['rotation_matrix']` (absent for
  Area_3 density metadata → would otherwise KeyError).
- **Search internals (for P2/P3 narrowing).** Translation grid ≈ 1700 candidates
  (`pose_search.py:117`); rotation = 24 Manhattan candidates
  (`build_rotation_candidates`, `pose_search.py:72-110`); coarse XDF inlier-count scoring
  (`pose_search.py:424-433`); top-K=10 pose shortlist → sphere-ICP refine (`pose_refine.py`), best-of-K
  by tight-inlier count (`multiroom_pose_estimation.py:453-471`). All geometric; **no color**.
- **Color is discarded at bake.** `src/geometry_3d/point_cloud_geometry_baker_V4.py:71` reads only
  `pcd.points`; FGPL scores against a *derived line map*, not the cloud. (Irrelevant to us — PanoPin
  consumes the color, FGPL stays geometric.)

## 4. Key design decision — run entirely in the raw S3DIS frame

PanoPin/CPO, the S3DIS room clouds, and the S3DIS pose GT are all in the **same raw S3DIS frame**
(identity ↔ pano/pose, ~5 mm, per Point_360; PanoPin `CLAUDE.md`). By building FGPL's line map from the
raw S3DIS clouds, feeding PanoPin's raw-3D position as the seed (with `aligned_meters_to_raw_3d` as
identity), and scoring FGPL's output against the raw-3D GT pose, **the entire floorplan/density/SAM3
front-end is unnecessary** — including the missing `rotation_matrix`. This is the enabler that makes
the S3DIS setup a small, well-bounded build rather than a port of the whole jigsaw front-end.

## 5. Subset (first cut — adjustable)

- **Rooms (6):** `office_1, office_4, office_5, office_6, office_7, hallway_3`.
  - Contains the documented same-shape confusion pairs **4↔6** and **5↔7** (PanoPin prototype set
    `A_offices`, `smoke/prototype_sets_timing.py`) — the case where shape-based coarse selection fails
    and color must do the work. `hallway_3` adds a corridor/"loss-sink" for realism.
- **Panos:** the GT panos of those rooms, capped ~2/room → ≈ 10 panos, all from PanoPin's **in-frame**
  GT set (pose GT available; out-of-frame panos D17 excluded).
- **Line map:** one shared `3d_line_map.pkl` built from the **combined** 6-room cloud (Voronoi then
  partitions it per pano).

## 6. Phase 0 — Enabler (with a hard go/no-go gate)

Build the three raw-frame FGPL artifacts for the subset:

1. **3D line map.** Combine the 6 S3DIS room clouds (`Point_360/data/s3dis/.../Area_3/<room>/<room>.txt`,
   `X Y Z R G B`) into one cloud → `point_cloud_geometry_baker_V4.py` → `cluster_3d_lines.py` →
   `3d_line_map.pkl` (+ `room_geometry.pkl`). Color is dropped by the baker (fine).
2. **2D features.** `image_feature_extractionV2.py` per subset pano → `fgpl_features.json`
   (~3 s/pano). Uses PanoPin's GT pano JPGs.
3. **Seed adapter + raw-frame config.** A PanoPin-side writer that emits a `demo6_alignment.json`-format
   file with per-pano raw-3D `camera_position`; an Area_3 `--config` JSON with the subset
   `point_cloud_name`/`pano_names` + path overrides + a flag/identity path so
   `aligned_meters_to_raw_3d` is a pass-through.

**GATE (go/no-go):** run the estimator on ≥1 subset pano with an **oracle (GT) seed**; require final
translation error **< 0.5 m**. If it fails → **stop and diagnose** (likely S3DIS line-map quality)
before building the ablation. This isolates "does FGPL even work on S3DIS clouds" from "does PanoPin's
seed help".

## 7. Phase 1 — Ablation (only after the gate passes)

Each arm runs on all subset panos; the seed is the only thing that varies (except P2/P3 which also set
search-narrowing config).

| Arm | Seed given to FGPL | Narrowing | Tests |
|-----|--------------------|-----------|-------|
| **Oracle** | GT camera XY (raw-3D) | none | FGPL ceiling — is the seed the bottleneck? |
| **P1** | PanoPin CPO position (`t`→XY) | none (full search) | Does the color room-pick alone suffice? |
| **P2** | PanoPin CPO position | translation grid restricted to ~1.5–2 m around seed | Does tightening translation help? |
| **P3** | PanoPin CPO position | P2 + rotation search seeded/narrowed by CPO's `R` (nearest Manhattan candidate / yaw prior) | Does CPO's rotation resolve symmetry? |
| **Wrong-room** | centroid of a *different* same-shape room (e.g. seed office_4's pano at office_6) | none | Floor — the catastrophic failure PanoPin prevents |

**Why no literal jigsaw arm:** running the real SAM3+IoU jigsaw on S3DIS would resurrect the
density/SAM3/alignment front-end this design deliberately deletes, and its wrong-room failure is
already what the **Wrong-room** arm demonstrates. The jigsaw's *speed* cost is already quantified on
TMB (§1). A literal head-to-head can be added later if a direct number is wanted.

**Narrowing mechanics (P2/P3)** are implemented as config/params consumed by the estimator's existing
search, not as changes to the cost:
- P2: bound `generate_translation_grid` extent to a radius around the seed (the grid is already
  region-bounded by the Voronoi slice; P2 tightens it further around the point).
- P3: restrict/reorder the 24 `build_rotation_candidates` by proximity to CPO's `R` (or inject CPO yaw
  as the initialization for refine). Exact hook chosen at plan time; must not alter the scoring.

## 8. Metric & success criteria

Reuse PanoPin's existing harness (`eval/metrics.py`: translation error in m, rotation error in deg vs
GT). Per arm, report:

- **Median & mean translation error (m)** and **rotation error (deg)**.
- **Wrong-room rate** — fraction of panos whose final pose falls outside the true room's bounds.
- **Runtime** (CPO seed time + FGPL time), for the speed story.

**Success signal (qualitative, n≈10 — treat as directional, not significant):** P1/P2/P3 approach the
Oracle and clearly beat the Wrong-room floor; the ablation identifies the sweet-spot guidance level
(is position alone enough, or is grid/rotation narrowing needed?). A clean result is *either* "P1
already matches Oracle" (color room-pick is sufficient; simplest handoff wins) *or* "accuracy climbs
P1→P2→P3" (more guidance needed; quantifies how much).

## 9. Components & interfaces (isolation)

- **`experiments/fgpl_seed/` (new, PanoPin repo)** — orchestration only; depends on `src/panopin`
  (CPO adapter, calibrate) + the FGPL repo via subprocess/config. Units:
  - `build_linemap.py` — combine subset clouds, invoke baker+cluster (wraps FGPL tools). Output:
    `3d_line_map.pkl`. Depends on: FGPL geometry tools, subset cloud paths.
  - `build_features.py` — invoke `image_feature_extractionV2` per pano. Output: `fgpl_features.json`.
  - `write_seed.py` — {arm, pano} → `demo6_alignment.json`-format raw-3D seed (Oracle from GT;
    P1/P2/P3 from CPO `localize_pair`; Wrong-room from a chosen sibling room). Output: seed JSON +
    narrowing config. Depends on: `src/panopin.cpo_adapter`, PanoPin GT (Oracle/Wrong-room only —
    allowed here because this is experiment scaffolding, not the fair solver; see fairness note).
  - `run_arm.py` — invoke `multiroom_pose_estimation.py --config <arm>.json`; collect `camera_pose.json`.
  - `score.py` — final poses vs GT via `eval/metrics.py`; emit per-arm table.
- **FGPL repo:** touched only via generated config JSONs + a small pass-through for the raw-3D seed
  (identity `aligned_meters_to_raw_3d`). Prefer a config flag over editing the estimator; if a minimal
  edit is unavoidable, keep it behind the flag and document it.

**Fairness note (D5).** PanoPin's *fair solver* (`src/panopin`) reads only the manifest. This
experiment's Oracle and Wrong-room arms intentionally use GT to construct *reference* seeds — that is
scaffolding for measurement, not part of the deployed method, and lives in `experiments/`, never in
`src/panopin`. P1/P2/P3 (the method under test) use only CPO output, no GT.

## 10. Cost, risks, open items

- **Cost (first cut, GPU `panopin-gpu`):** CPO seeds = ~6 rooms × ~10 panos × ~25 s ≈ **~25 min**
  (P1/P2/P3 reuse the same CPO `(t,R)` — compute once). FGPL ≈ 55 s × 10 panos × 5 arms ≈ **~45 min**
  (Oracle/Wrong-room/P1 share settings; P2/P3 differ only in narrowing). Line-map + feature build:
  minutes. Tractable in a session.
- **Risks:**
  1. **Phase-0 gate fails** — S3DIS room clouds are noisier than TMB; the C++ line detector may yield a
     poor map. Mitigation: the gate is first; diagnose (subsample, params) before the ablation.
  2. **CPO rotation reliability (P3)** — never validated in PanoPin (only position was). P3 *is* the
     test; a null/negative P3 result is still informative.
  3. **Frame identity** — verify GT-pose frame == cloud frame == line-map frame on one pano before
     trusting scores (a fixed offset would masquerade as method error).
  4. **CPO nondeterminism** (~±0.02 loss, torch) — pin `panopin.determinism`; treat n≈10 as directional.
- **Open items (decide at plan/implementation time):** exact P2 radius; exact P3 rotation hook; whether
  `hallway_3` stays in or a second same-shape pair replaces it; pano cap per room.

## 11. References

- FGPL handoff scope: `docs/specs/2026-07-09-fgpl-handoff-scope.md`.
- FGPL estimator: `scan2measure-webframework/src/pose_estimation/multiroom_pose_estimation.py`;
  search `pose_search.py`, `pose_refine.py`; features `src/features_2d/image_feature_extractionV2.py`;
  geometry `src/geometry_3d/{point_cloud_geometry_baker_V4,cluster_3d_lines}.py`; jigsaw
  `src/floorplan/align_polygons_demo6.py`.
- PanoPin v1: `src/panopin/{cpo_adapter,calibrate}.py`; prototype sets `smoke/prototype_sets_timing.py`;
  decisions D14–D24 (`docs/DECISIONS.md`).
