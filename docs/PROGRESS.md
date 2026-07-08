# PanoPin — Progress log

_Newest first. Update at the END of every session: what changed, what's next, where you stopped._

## Current state (2026-07-09)
- **Design + plan DONE & approved. Implementation (subagent-driven) IN PROGRESS: plan Tasks 1–3 of 7
  complete on branch `feat/coarse-room-cpo`. RESUME AT plan Task 4 (M0 real-data smoke).**
- Built: vendored CPO (Apache-2.0) + `panopin` conda env (py3.8, torch 1.10 CPU); `cpo_config.py`
  (`load_cfg` + `TIER1`/`TIER2`); `cpo_adapter.localize_pair(cfg, pano, cloud) -> (t, R, loss)` composing
  CPO primitives (reviewed-correct); synthetic fixtures. Solver CLI NOT built yet (that's Task 6).
- **TOP OPEN RISK (D13):** the synthetic discrimination test is `xfail` — CPO can't localize flat
  synthetic boxes (self-match loss ~0.22), so "content disambiguates same-shape rooms" is UNVALIDATED.
  First thing next session = **Task 4 M0 smoke on REAL Area_3 data**: does CPO localize a real pano to
  its real room (low loss, near-GT pose)? If the margin is inadequate on the ~7 identical pairs, revisit
  the method before Task 7.
- Harness floor unchanged: random baseline = 5.9% room accuracy — the number to beat (Task 7 / project T3).
- Plan: `docs/plans/2026-07-08-coarse-room-cpo-v0.md` (7 tasks). SDD ledger: `.superpowers/sdd/progress.md`
  (says Tasks 1–3 done). To resume: `superpowers:subagent-driven-development` → start Task 4.

## Session log

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
