# PanoPin — Progress log

_Newest first. Update at the END of every session: what changed, what's next, where you stopped._

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
- **NEXT = T4:** define the FGPL hand-off — what the coarse seed must deliver (top-1 room? top-k? coarse
  pose?). That decides whether 78% recall@1 is already enough (seed role → move on) or accuracy needs work
  on the same-shape-office / degenerate-room cases. Baseline to beat unchanged: 5.9%.
- Plans: `docs/plans/2026-07-08-coarse-room-cpo-v0.md` (v0), `docs/plans/2026-07-09-cheap-tier1-eval.md`
  (Plan 2, ABANDONED). Decisions D14–D20. SDD ledger has the full P2 gate/GPU trail.

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
- **STOPPED** here. Housekeeping done; NEXT = T4 (FGPL hand-off scope). Nothing pushed to GitHub.

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
