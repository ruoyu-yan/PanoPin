# PanoPin — Progress log

_Newest first. Update at the END of every session: what changed, what's next, where you stopped._

## Current state (2026-07-08)
- **Scaffold + evaluation harness created and verified on real S3DIS Area_3. Method design NOT
  started — no solver exists yet.**
- The loop runs end-to-end: `make_manifest` → baseline → `score`. **Random baseline = 5.9% room
  accuracy** (5/85; chance = 1/23 ≈ 4.3%). That is the floor to beat.
- Next: **T1** (survey approaches) → **T2** (brainstorm + design spec) → **T3** (first solver that
  beats random). See `docs/tasks.json`.

## Session log

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
