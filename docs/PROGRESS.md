# PanoPin — Progress log

_Newest first. Update at the END of every session: what changed, what's next, where you stopped._

## Current state (2026-07-09, later — Task 4 + Task 5 done, big data finding)
- **Plan Tasks 1–5 of 7 done on `feat/coarse-room-cpo`. RESUME AT Task 6 (solve.py CLI).** Read
  DECISIONS **D14–D17** first — the method is validated on real data, but a real GT limitation surfaced.
- **D13 RESOLVED for in-frame panos (D14):** on REAL Area_3, CPO self-localizes the office_3 pano to its
  cloud at **0.039 m** (cm-level, frame identity confirmed) and **discriminates** it against 4 near-
  identical offices + a hallway by **+0.089 loss (+75% margin)** — content, not shape, picks the room.
  This is what the synthetic box test (D13) could not show. Evidence: `smoke/reproduce_cpo_one_room.py`,
  `smoke/discriminate_one_pano.py`.
- **NEW BLOCKER — out-of-frame panos (D17):** `smoke/check_frame_alignment.py` found **9/85 panos (11%)
  whose GT camera sits OUTSIDE its room cloud** (lounge_2 2.0 m, office_10 1.5 m, office_9 0.9 m,
  office_2 0.7 m). CPO can't localize those (searches only within the cloud) → they cap accuracy at
  ~89% and their pose metric is invalid. It's a DATA/GT issue, not a method failure. (office_9's earlier
  discrimination FAIL was this, not CPO.) Task 7 must report accuracy split **in-frame (76) vs all (85)**.
- **Cost caveat (D16):** the two-tier funnel is correct but not yet cheap — the fixed inlier score-map
  detection (~20 s/room) dominates BOTH tiers, so a full Area_3 run is ~10 h as-is. Task 7 needs a
  lighter Tier-1 scorer before scaling.
- Task 5 (`select_room` two-tier funnel) built; funnel logic unit-tested by MOCKING localize_pair (the
  synthetic same-shape test is known-broken, D15) — 4 tests green. Real-data validation via
  `smoke/select_room_real.py`.
- Determinism: CPO is NOT bit-reproducible on CPU (torch scatter/Adam, ~±0.02 loss); deferred (D1).
- Harness floor unchanged: random baseline = 5.9% room accuracy (Task 7 / project T3 target).
- Plan: `docs/plans/2026-07-08-coarse-room-cpo-v0.md`. SDD ledger updated (Tasks 1–5 done).

## Session log

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
