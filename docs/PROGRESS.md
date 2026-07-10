# PanoPin — Progress log

_Newest first. Update at the END of every session: what changed, what's next, where you stopped._

## Current state (2026-07-10 — FGPL seed ablation DONE; color POSITION seed ~= oracle when room is right)
- **NEW LINE: PanoPin->FGPL seed ablation** (branch feat/fgpl-seed-ablation; spec+plan docs/{specs,plans}/2026-07-10-*; code experiments/fgpl_seed/). Replace FGPL's slow SAM3+IoU jigsaw with PanoPin's color seed; measure FGPL pose accuracy on a same-shape multi-room S3DIS subset. Executed subagent-driven (SDD ledger .superpowers/sdd/progress.md has full task history).
- **Setup:** ran FGPL entirely in the RAW S3DIS frame (identity metadata -> NO floorplan/SAM3/density front-end). 6-room subset (office_1/4/5/6/7 + hallway_3), 12 panos. Built 3d_line_map (2215 lines / 3159 intersections), fgpl_features, raw-3D seed adapter. Estimator in panopin-gpu (Ada cu118; scan_env cu116 = CPU fallback). Multi-pano Voronoi -> ~8-40s/pano (vs 325s single-pano whole-map).
- **ORACLE GATE:** median trans 0.789m, 12/12 localized, 1/12 wrong-room -> FGPL localizes correctly given a good seed (well within FGPL's ~3m coarse tolerance; my 0.5m gate bar was miscalibrated). D25.
- **5-ARM ABLATION (n=12), trans median / wrong-room:** oracle 0.789 / 8% ; p1(color pos) 0.946 / 33% ; p2(+grid) 0.946 / 33% (grid = NO effect) ; p3(+CPO yaw) 1.807 / 42% (HURTS) ; wrong_room 7.213 / 100%.
- **HEADLINE:** where CPO picks the right room (8/12), its POSITION seed is as good as GT: median 0.762m vs oracle 0.725m -> FGPL localizes IDENTICALLY. The entire P1-vs-oracle gap = the 4 wrong-room misses (~22m, all -> hallway_3 loss-sink). So the color POSITION seed WORKS; the bottleneck is ROOM RECALL (67% raw min-loss). **v1 calibration (33->58% recall) is the exact lever** and cpo_seeds uses RAW min-loss (no calibrate) -> NEXT = wire v1 calibrate into the seed + re-run. Drop P2/P3.
- **FGPL rotation convention (D25):** FGPL outputs Rp = C @ R_wc, C=[[0,0,1],[-1,0,0],[0,-1,0]] (same signed-perm Point_360 solved); Rp.T@C = camera->world (0.5deg on precise panos). FGPL rotation also ~90deg Manhattan-aliased even with a good seed. CPO rotation UNUSABLE (~120deg off GT both conventions) -> P3 garbage.
- All local, nothing pushed. Runtime whole ablation ~40min.

## Current state (2026-07-09, latest — method validated; speed characterized; NEXT = T4 FGPL hand-off)
- **Method VALIDATED (D14):** CPO assigns the right room for in-frame panos. But the **cheap Tier-1
  speedup is ABANDONED** (D18/D19): no cheap/coarse proxy preserves accuracy — the per-room Adam pose
  refine is load-bearing (corridors/small rooms match many coarse poses; the true room's low loss only
  emerges after refinement). Coverage-normalization refuted; reduced-inlier fails.
- **Speed is intrinsic (~25 s/room):** GPU gives only ~2× (D20) — CPO's inlier detection (~9360-pose
  Python loop) + Adam refine (600 steps) are launch-bound, not GPU-able. **Assigning ONE pano = ~25 s ×
  (num candidate rooms).** So: 3–5 room prototype scene ≈ **~1–2 min/pano** (tractable); whole Area_3
  (~23 rooms) ≈ **~7 min/pano** → full 76-pano eval ~9–18 h.
- **Accuracy at prototype scale (NEW, `smoke/prototype_sets_timing.py`):** 3 sets of 3 rooms, each pano vs
  its set's 3 room segments on GPU → **recall@1 = 14/18 (78%)**, ~75–86 s/pano. Misses are the
  characteristic hard cases: same-shape office pairs (office_4→office_6, office_5→office_7) and a
  **degenerate "loss-sink" room WC_1** (wrongly won twice) — same phenomenon as the whole-area hallway wins.
- **Envs:** `panopin` (CPU, torch 1.10+cpu — reproducible, D10); **`panopin-gpu`** (torch 2.0.1+cu118 +
  torch_scatter 2.1.2+pt20cu118 + CPO deps — runs on the RTX 4060; ~2× faster; use for GPU runs). scan_env
  is NOT ours — do not use.
- Out-of-frame panos (D17): 9/85 have GT camera outside their room cloud → unlocalizable; report accuracy
  in-frame (76) vs all (85). Determinism: CPO not bit-reproducible on CPU (~±0.02 loss); deferred (D1).
- Candidates = individual S3DIS room segments (`Stanford3dDataset_v1.2/Area_3/<room>/<room>.txt`); pose
  search within a room is bounded by that segment's own extent (quantile trans init).
- **T4 FGPL hand-off SCOPED (D-spec 2026-07-09-fgpl-handoff-scope):** FGPL needs per pano only {top-1
  room_label, rough 2D position ~3 m}; no rotation/6-DoF. Production FGPL is top-1 (top-k `infer_room` was
  deprecated). Chose **B: hand off a shortlist + re-enable FGPL top-k** (PanoPin narrows, FGPL disambiguates).
- **Whole-area reality (D21):** vs all 23 rooms, raw recall@1 collapses to ~17-33% — a FIXED set of
  small/corridor "loss-sink" rooms (hallway_5/6/4, WCs, storage) top-rank every pano. (Prototype's 78% was
  easy — it excluded them.)
- **v1 FINALIZED — calibration + confidence gate (D22→D24):** `src/panopin/calibrate.py`. **Fair (no-GT)
  minmax calibration** lifts whole-area recall@1 to **58%** (n=12) vs 33% raw; a tunable **confidence gate**
  gives a precision/coverage tradeoff (−0.3 → 100% precision @ 33% coverage) and reliably FLAGS the hard-fail
  panos. **CORRECTION (D24):** an earlier "percentile → clean 100%/58%" result was a GT LEAK; fair numbers are
  the modest ones above. Unit-tested (tests/test_calibrate.py, 3 pass); scored via smoke/score_v1.py on the
  cached 12×23 loss matrix (runs/calib_matrix.json).
- **Hard floor (D23, stands):** ~1/3 of panos FAIL to self-localize (own-room loss ~0.25 vs ~0.08) —
  window/blank-wall/occlusion-dominated views (office_8: 40% window + chair occlusion) — uncatchable by
  calibration or shortlist; correctly flagged by the confidence gate.
- **NEXT:** (a) wire v1 into solve.py (manifest→matrix→calibrate→predictions) + validate on more panos than
  n=12; (b) **ceiling-push on a NEW branch** — robust/masked loss to rescue minority-window panos (D23 lever).
- **Envs:** `panopin` (CPU repro, D10); `panopin-gpu` (torch 2.0.1+cu118, RTX 4060, ~2× — GPU runs). scan_env
  NOT ours. Determinism: CPO not bit-reproducible on CPU (~±0.02). Baseline to beat: 5.9%.
- Plans: v0 `docs/plans/2026-07-08-coarse-room-cpo-v0.md`; Plan 2 (cheap Tier-1) ABANDONED. Decisions D14–D24.

## Session log

### 2026-07-09 (latest) — cheap Tier-1 gate failure → GPU → speed characterized → prototype timing
- Wrote spec+plan for a cheap Tier-1 (skip the ~20 s inlier detection); executed subagent-driven. Tasks 1–2
  done (determinism helper 286da6d..e845fe3; `score_room_cheap` e8a14b4..f12b43e, both reviewed clean).
- **Task-3 GATE FAILED (D18):** cheap Tier-1 recall@5 = 0/5 — degenerate; small/corridor clouds win.
  Chased the fix: reduced-inlier 1/3, Adam-without-inlier 2/3, coverage-normalization REFUTED. Root cause =
  per-room Adam pose refine is required (D18/D19). **Plan 2 abandoned.**
- **GPU (D20):** built `panopin-gpu` (torch 2.0.1+cu118) — runs on the RTX 4060 (Ada). office_3 rank 1
  (loss 0.081, recall preserved) but **22 s/room, only ~1.8× CPU** — CPO's loops are launch-bound, not
  GPU-able. (scan_env cu116 segfaulted on Ada; reverted the torch_scatter I'd added there.)
- **Prototype timing (user request):** clarified units (~25 s is per ROOM, not per pano; assign = ×N
  candidate rooms). Ran 3 sets of 3 rooms, each pano vs its set's 3 segments on GPU: ~8 min/set (~75–86 s/
  pano), **recall@1 = 14/18 (78%)**. Misses = same-shape offices + WC_1 loss-sink.
- **T4 scoped** (`docs/specs/2026-07-09-fgpl-handoff-scope.md`): FGPL needs {top-1 room, ~3 m 2D position};
  chose option B (shortlist + FGPL top-k). **Whole-area sizing (D21):** raw recall@1 collapses to ~17–33%
  (loss-sink rooms dominate). **Calibration (D22→D24):** fair minmax → recall@1 58%; confidence gate is a
  precision/coverage tradeoff. **Caught + fixed a GT LEAK (D24)** in my own calibration analysis (the earlier
  "clean 100%/58%" was invalid). **Diagnosed the hard floor (D23):** ~1/3 panos fail to self-localize
  (window/occlusion). **v1 committed:** `src/panopin/calibrate.py` + tests + `smoke/score_v1.py`.
- **STOPPED for the day at v1.** Handover written (`docs/HANDOVER.md`). Ceiling-push branch
  `feat/robust-loss-ceiling` created (empty, ready). Two open threads: robust-loss ceiling-push; v1→solve.py
  wiring + validate beyond n=12. Nothing pushed to GitHub.

### 2026-07-09 (later) — Task 4 (M0) + Task 5 (funnel) + the out-of-frame discovery
- **Task 4 (M0), committed b579efe.** Frame pre-checked cheaply (office_3 camera inside cloud bbox, Z≈1.4 m,
  RGB 0–255). `localize_pair` self-localizes office_3 pano→cloud at **0.039 m**, loss ~0.12, ~26 s CPU
  (sample_rate=30, top_k=1, num_iter=20). Seeded reruns: 0.086/0.135 m — all cm-level. Seeding np.random
  pins subsampling but NOT torch (not bit-reproducible; ~±0.02 loss).
- **Discrimination probe, committed 8ffd15e.** office_3 pano vs {office_1,2,6,9, hallway_1}: correct room
  wins, loss 0.1185 vs 0.21–0.26, **+75% margin**, only room with a correct pose (0.095 m). → D14.
- **Probe #2 (office_9) FAILED — and that led to the big finding.** office_9 didn't even self-localize
  (4.4 m; 5.8 m at full budget). Cause: **office_9's GT camera is OUTSIDE its cloud** (bbox check). Swept
  all 85 with `smoke/check_frame_alignment.py`: **76/85 in-frame (89%), 9 out-of-frame** across
  lounge_2/office_10/office_9/office_2. → D17. Method is fine; those panos are a GT/frame limitation.
- **Task 5** `select_room` two-tier funnel + 4 mock funnel-logic tests (green) + `smoke/select_room_real.py`.
  Replaced the plan's known-broken synthetic same-shape test (D15); added a plan callout.
- Recorded D14–D17. **STOPPED** at Task 5 done. Next: Task 6 (solve.py), then Task 7 with the D16 cost
  fix + D17 accuracy split. A third in-frame discrimination confirmation (office_5) was queued.
  Nothing pushed to GitHub.

### 2026-07-08/09 — implementation session: v0 plan + subagent-driven Tasks 1–3
- Wrote the v0 implementation plan (`docs/plans/2026-07-08-coarse-room-cpo-v0.md`, 7 tasks) and executed
  it subagent-driven (fresh implementer + reviewer per task, ledger at `.superpowers/sdd/progress.md`).
- **Task 1** vendor CPO + `panopin` env — done (needed a D12 doc fix: open3d couldn't be dropped).
- **Task 2** `cpo_config` (load_cfg + TIER1/TIER2) — done, review Approved.
- **Task 3** `localize_pair` + synthetic fixtures — core reviewed-correct and committed. Its synthetic
  discrimination test proved UNRELIABLE (CPO can't localize flat boxes; wrong room won across 4 configs
  + match_color on/off) → marked `xfail`; real validation deferred to Task 4. See **D13**. ~1h spent
  diagnosing; payoff = the D13 insight (rotational-alignment failure mode + fixture-fidelity limit).
- **STOPPED** for the day at plan Task 3 done. Resume at Task 4 (M0 real-data smoke). Nothing pushed to GitHub.

### 2026-07-08 — T1 survey + T2 design (render-and-compare / CPO)
- Cloned the GitHub repo into the local PanoPin dir and pushed the skeleton (branch `main`).
- Ran the session ritual; T1 survey via 3 research agents + a deep-research run: the pano→colored-cloud
  line is PICCOLO/CPO/LDL/FGPL (`82magnolia`, Apache-2.0, all training-free, benchmarked on Stanford2D3D).
- Area_3 EDA: geometry is separable *on Area_3* but reproduces the thesis' same-shape blind spot; global
  color is degenerate → chose spatially-resolved appearance matching (CPO), demoting shape.
- User rejected shape matching (== the thesis) → confirmed render-and-compare; user approved building upon CPO.
- Code recon of CPO: CPU-feasible, training-free, `read_txt_pcd` matches our `X Y Z R G B` clouds exactly,
  room score = min `refine_pose_sampling_loss` loss; a two-tier funnel is needed for CPU cost. Wrote the
  options note + design spec; added decisions D8–D11.
- **STOPPED:** awaiting user review of the design spec before `writing-plans` / implementation.

### 2026-07-08 — project setup (setup session — done by the parent Scan2BIM session, not the PanoPin agent)
- Spun PanoPin out of the Feyzullah meeting (see Point_360 `roadmap.md` §"Scope update — 2026-07-08").
- Researched autonomous-agent best practices from 7 verified Anthropic sources →
  `docs/best-practices-references.md`; distilled the protocol into `CLAUDE.md`.
- Built and **verified against real Area_3 GT**:
  - `config/datasets.json` — data paths (GT on /mnt/d, room clouds on ext4).
  - `eval/s3dis_gt.py` — loads 85 panos, room + pose GT, 23 candidate rooms (21 with panos). ✓ runs.
  - `eval/metrics.py` — room accuracy + coarse translation/rotation error. ✓ self-test.
  - `eval/score.py` — CLI scorer. ✓ scored the baseline (5.9%).
  - `eval/make_manifest.py` — anonymized solver input (UUID-only panos + candidate clouds). ✓ 85 linked.
  - `eval/baselines/random_room.py` — the floor. ✓ wrote predictions.
  - `tests/test_harness.py` — ✓ green. `src/panopin/` — empty package stub.
- Git initialized (no commits yet — first commit is the agent's / user's call).
- **STOPPED:** handed off. First real task = T1/T2 (research routes, then brainstorm the method).
  Remember: design-before-implement (protocol #2), and never read room/pose from GT paths (fairness).
