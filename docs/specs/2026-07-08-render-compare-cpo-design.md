# PanoPin design spec — coarse room-selection via CPO render-and-compare

**Date:** 2026-07-08 · **Status:** proposed (awaiting review) · **Task:** `tasks.json` T2
**Builds on:** [options note](2026-07-08-coarse-pano-to-room-options.md) · **Decisions:** D1–D7 + D8–D11 (below)

## 1. Goal
Assign each 360° equirectangular RGB panorama to the correct **room** among candidate colored point
clouds, and emit a **coarse camera pose** (translation + yaw) as a seed for the line-based fine
localizer **FGPL**. Primary metric: room-assignment accuracy on S3DIS Area_3 (beat the 5.9% random
floor); secondary: coarse-pose error. Repetitive/same-shape rooms are the target hard case.

## 2. Approach — build upon CPO
PanoPin is a thin **color-based coarse room selector** that reuses CPO
(`82magnolia/panoramic-localization`, Apache-2.0 — same repo as FGPL). CPO scores color-consistency
between a pano and a colored cloud in a rendering-free, training-free, deterministic way; we run it
**per candidate room**, rank rooms by CPO's color-consistency cost, and hand the winning room + its
coarse pose to FGPL. This keys on room-specific *content* (posters, furniture, windows), which
disambiguates same-shape rooms where shape/line geometry cannot — see the options note for the full
rationale and the rejected alternatives.

**One-line framing:** FGPL already does "score rooms by a scalar cost → keep top-k → refine," keyed on
*line-distance* cost (`fgpl/xdf_precompute.infer_room`). PanoPin replaces that selector with CPO's
*color* cost.

## 3. Architecture & data flow
```
manifest.json (anonymized: panos by uuid + candidate room .txt clouds)
      │  (solver reads ONLY this — fairness D5)
      ▼
for each pano uuid:
   load pano → resize 2048×1024, RGB∈[0,1]                     [cpo utils]
   TIER 1 (cheap, all rooms): for each candidate room
       read_txt_pcd(room.txt) → xyz, rgb                        [cpo data_utils]
       coarse pose search (score-map + histogram, NO Adam)      [cpo utils.histogram_pose_search]
       cheap room score (histogram-intersection at coarse pose) [cpo color_utils]
   rank rooms by cheap score → keep TOP-K (e.g. 3–5)
   TIER 2 (refine, top-k only): for each surviving room
       refine_pose_sampling_loss → (t, R, loss)                 [cpo sampling_loss]
   pick room = argmin loss;  coarse pose = that room's (t, yaw)
      ▼
predictions.json  { room (required), coarse_pose{t,[R]} }       → eval/score.py
                                                                  → (T4) demo6_alignment.json → FGPL
```

## 4. Components (isolation & interfaces)
Vendored CPO lives read-only under `third_party/cpo/`; PanoPin's own code under `src/panopin/`.

**Reused from CPO (unmodified):**
- `data_utils.read_txt_pcd(path) -> (xyz, rgb)` — loads our `X Y Z R G B` clouds; RGB→[0,1].
- `utils.make_score_map_2d/3d`, `utils.histogram_pose_search(...) -> (trans, rot)` — coarse candidates.
- `utils.make_pano(xyz, rgb, resolution)` — rendering-free splat to equirect (for the cheap score).
- `color_utils.histogram*`, `color_utils.color_match` — histograms + CDF exposure match (change-robustness).
- `cpo/sampling_loss.refine_pose_sampling_loss(...) -> [t(3×1), R(3×3), loss]` — Adam refine + cost.
- `parse_utils.parse_ini(config/stanford_cpo.ini)` — build the `cfg` namedtuple (keep `dataset=stanford`).

**PanoPin's own modules:**
- `src/panopin/cpo_adapter.py` — reproduces CPO's single-localization orchestration (score-maps →
  `histogram_pose_search` → `refine_pose_sampling_loss`) but **returns `(t, R, loss)`** for one
  (pano, cloud) pair. Rationale: CPO's `localize_single.localize` discards these (returns `None`); we
  compose the underlying primitives instead of forking its entry point. Also exposes a cheap
  `coarse_score(pano, xyz, rgb)` (Tier-1, no Adam).
- `src/panopin/select_room.py` — the two-tier funnel over candidate rooms → best room + coarse pose.
- `src/panopin/solve.py` (CLI) — reads the manifest, runs `select_room` per pano, writes predictions
  JSON per `eval/PREDICTIONS_SCHEMA.md`. Reads **only** the manifest (fairness).
- `src/panopin/fgpl_handoff.py` (T4) — convert world-frame `t` → `demo6_alignment.json` `camera_position`
  (aligned-meters via a `metadata.json`); documented, not on the v0 critical path.

## 5. Coarse pose & FGPL hand-off
CPO returns `t (3×1)` (camera translation, world frame = Original S3DIS = pano frame, D4) and `R (3×3)`;
yaw = Z-axis component (`R = Rz(yaw)·Ry(pitch)·Rx(roll)`). Emit `coarse_pose.t=[x,y,z]`, optional
`R`. For the real FGPL integration (T4), `fgpl_handoff.py` maps `t` → the aligned-meters
`camera_position:[x,y]` FGPL reads (rotation is re-searched by FGPL, so yaw is optional there). CPO's
`(t,R)` can also seed FGPL's `refine_from_sphere_icp` directly — noted for later, out of v0 scope.

## 6. Environment (D10)
Dedicated `panopin` conda env (self-contained, matches the parent project's fresh-env pattern):
`python=3.8`, `torch==1.10.* (CPU wheel)`, `numpy~=1.23`, `opencv-python`, `pandas`, `scipy`,
`scikit-learn`, `Pillow`. **Dropped:** `tensorflow-cpu` (dead import), `pylsd-nova`/`einops` (line-method
only), `open3d` (only an unused top-level import in `data_utils` — stripped in the vendored copy).
Harness (`eval/`) stays stdlib-only (D6); this env is the solver's only.

## 7. Testing (nothing "done" without a runnable check — protocol #4)
- **Unit (synthetic, deterministic):** build a synthetic colored box-room with a distinctive colored
  patch on one wall; render a pano from a known pose via `make_pano`; assert the adapter recovers a
  near-GT pose (low loss). **Same-shape test:** two identical boxes differing only in patch
  color/position → selector must pick the right one. *This is the falsifiable test of the user's core
  concern.*
- **Smoke (real, reproduce CPO):** run the adapter on ONE Area_3 room + its GT pano (GT used here for
  dev validation only, never in the solver) → localizes near GT pose, in the paper's ballpark.
  De-risks vendoring + env before building the room loop. (Mirrors the parent project's smoke-first rule.)
- **Integration (T3):** solver over Area_3 from the anonymized manifest → `eval/score.py` room
  accuracy **≫ 0.06**.
- **Diagnostic:** per-room confusion focused on the ~7 identical-shape pairs (office_9↔10, WC_1↔2, …) —
  does color actually separate them? Reported honestly in PROGRESS.

**Accuracy context:** CPO reports 0.83 @ 0.05 m/5° on Stanford2D3D (full pose); our room metric is
coarser and should sit higher. FGPL's repetitive-office splits (0.64–0.68) are a later alignment target.

## 8. Milestones (map to tasks.json)
- **M0 — smoke:** vendor CPO + build `panopin` env + reproduce CPO on 1 room. Evidence: pose+loss vs GT.
- **M1 → T3:** two-tier room selector + solver CLI (manifest-only) → room accuracy ≫ 0.06. Evidence: `score.py`.
- **M2 → T4:** emit coarse pose; report pose error; write + document `fgpl_handoff.py`. Evidence: `score.py` pose metrics.
- **M3 → T5:** all 85 panos; same-shape diagnostic; tune Tier-1 top-k; log failures in PROGRESS.

## 9. Risks & mitigations (from the code recon)
1. **CPU cost** (score-maps + 6×100 Adam per room × 21 rooms) → the Tier-1 cheap rank limits Adam to
   top-k rooms; voxel-subsample clouds once per room; cache Tier-1 poses. Measure M1 wall-clock; if
   still too slow, reduce `top_k_candidate`/`num_iter` in a `panopin` config override.
2. **torch 1.10 pin is CUDA-tagged** → install the CPU wheel; pin `numpy~=1.23` to avoid `np.int`
   deprecations in old code (the color path avoids `read_line`, but pinning is safest).
3. **INI-config coupling** (`cfg` namedtuple, `dataset=stanford` hard-check) → reuse `stanford_cpo.ini`,
   add a small `panopin` override for pano paths / tunables; drive CPO via primitives, not the batch CLI
   (which hardcodes `./data/stanford` paths).
4. **`match_color=True`** rewrites query colors to the cloud distribution before scoring → keep it ON and
   identical across all candidate rooms so per-room costs are comparable.
5. **2048×1024 pano + z-up, `atan2(y,x)` frame assumption** → verify our S3DIS panos/clouds share it in
   M0 (they should — it's the Stanford2D3D reference convention).
6. **Same-shape twins identical in content too** → best-effort; report; escalate to a feature/depth
   fallback only if the bar isn't cleared, logged in DECISIONS.

## 10. Reuse / licensing
Vendor a minimal CPO subset into `third_party/cpo/` (the `.py` primitives above + `config/stanford_cpo.ini`),
**preserving `LICENSE` (Apache-2.0)** and a `third_party/cpo/ATTRIBUTION.md` noting the upstream commit,
authors, and our one change (stripped unused imports; added a returning orchestration in *our* code, not
theirs). Cite CPO (ECCV 2022) in the eventual paper.

## 11. New decisions recorded (see DECISIONS.md)
- **D8** — render-and-compare (CPO), not shape, is the decider (repetitive-room rationale).
- **D9** — build upon CPO by **composing its primitives** in our own returning orchestrator (no fork/patch of its entry point).
- **D10** — dedicated `panopin` CPU env (torch 1.10 CPU, py3.8), not reuse of `scan_env`.
- **D11** — two-tier funnel (cheap spatial rank → Adam-refine top-k) for CPU tractability.

## 12. Open questions (resolve in the plan / M1)
- Exact Tier-1 cheap score (histogram-intersection at coarse pose vs coarse score-map cost) — pick empirically in M1.
- Tier-1 `top-k` and pose-grid density — tune for the accuracy/latency trade in M1/M3.
- Camera-height handling for candidate poses — CPO searches translation; confirm its trans grid covers S3DIS camera heights in M0.
