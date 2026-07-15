# PanoPin↔FGPL Validation Round-trip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the D34 `fgpl_export` seed through FGPL's estimator end-to-end on the 6-room `area3_seed_ablation` subset, and measure refined-pose accuracy vs S3DIS GT + per-room coverage — proving PanoPin and FGPL run together and the deployment claim survives FGPL refinement.

**Architecture:** One new experiment script `experiments/fgpl_seed/roundtrip.py` that mirrors `run_all.py`'s setup but swaps the seed source to `panopin.fgpl_export`. It reuses the already-built harness verbatim (`subset`, `seed_and_config`, `run_arm`, `score`, the `work/` line-map/features/cloud, cached `oracle`/`p1` poses). Offline logic (seed build, pre-check, coverage metric) is verified GPU-free first; the single ~60-min FGPL estimator run is the last step.

**Tech Stack:** Python 3.8. Offline pieces run in the `panopin` conda env; `run_arm` internally spawns the FGPL estimator in `panopin-gpu`. numpy, stdlib json, pytest.

## Global Constraints

- **Scene = the 6-room subset** `area3_seed_ablation` (office_1/4/5/6/7 + hallway_3, 12 in-frame panos) — the only scene with FGPL-estimator data built. Do not use largeval's 8 rooms.
- **Fairness (D5):** the seed is built ONLY from cached color scores/poses (`work/seeds/residuals.json`, `work/seeds/cpo_cache.json`) — never GT. GT (`s3dis_gt`) enters only in scoring and centroid/coverage computation.
- **Gated deployment arm:** seed FGPL via `fgpl_export.export_alignment(..., tau=0.10)`; run FGPL on only the `admitted` panos (`cfg["pano_names"] == admitted`). The ungated variant is a deferred non-goal.
- **Identity metadata** (`sc.write_identity_metadata()`): raw==aligned frame, so `fgpl_export`'s frame conversion is a validated no-op here.
- **Reuse verbatim, do not reimplement:** `subset.build_subset()`, `seed_and_config.{write_identity_metadata,write_config,room_centroids}`, `run_arm.run_arm`, `score.{score_arm,_nearest_room}`, `eval.s3dis_gt`, `eval.metrics`.
- **FGPL rotation convention:** FGPL outputs `C @ R_wc` with `C = [[0,0,1],[-1,0,0],[0,-1,0]]`; convert to camera→world before scoring via `_fgpl_rot_to_cw(R) = np.array(R).T @ C` (same as `run_all.py`).
- **Verified data shapes:** rows fields `[pano_name, uuid, room, pano_jpg, cloud_txt]`; `cpo_cache[pano] = {t,R,loss,room,poses:{room:{t,R}}, room_cal, conf, is_confident}`; `residuals.json = {pano:{room:[101 floats]}}`; `gt[pano] = {room, location, rt, R_cw}`; pano ids match across all three.
- **Per-room coverage metric:** denominator = the 6 subset rooms; a room `r` is *covered* iff ≥1 admitted pano whose **true** room is `r` has a refined FGPL pose whose nearest centroid == `r`.
- **Commit** after each task on branch `feat/fgpl-roundtrip`.

---

### Task 1: Seed building + GPU-free pre-check through FGPL's real loader

**Files:**
- Create: `experiments/fgpl_seed/roundtrip.py`

**Interfaces:**
- Consumes: `panopin.fgpl_export.export_alignment`, `panopin.robust_score.low_percentile_scores`, `experiments.fgpl_seed.paths`, `seed_and_config` (for metadata path in the check command).
- Produces: `load_scores_poses() -> (scores, poses)`; `build_export_seed(rows, md_path, tau=0.10) -> (seed_path, admitted)`; `precheck_seed(seed_path, md_path, admitted, rows) -> (n_seed, sorted_rooms)`; `_fgpl_rot_to_cw(R) -> list`.

- [ ] **Step 1: Write the module with the seed-build + pre-check functions**

```python
# experiments/fgpl_seed/roundtrip.py
"""PanoPin<->FGPL validation round-trip (spec 2026-07-15-fgpl-roundtrip). Seed FGPL from the
D34 fgpl_export module, run the estimator on the 6-room area3_seed_ablation subset, score
refined poses vs S3DIS GT + per-room coverage. Reference: cached oracle/p1 (work/poses/).

Offline pieces (seed build, pre-check, coverage) run in `panopin`; run_arm internally spawns
the FGPL estimator in `panopin-gpu`. Run the full round-trip:
    conda run -n panopin python -m experiments.fgpl_seed.roundtrip
Fair (D5): the seed is built only from cached color scores/poses, never GT."""
import json
import sys
import numpy as np

from experiments.fgpl_seed import subset, seed_and_config as sc, run_arm, score, paths
from eval import s3dis_gt
from panopin import robust_score, fgpl_export

C = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # FGPL equirect signed-perm
SEEDS = paths.WORK / "seeds"


def _fgpl_rot_to_cw(R):
    """FGPL output rotation (C @ R_wc) -> camera->world, for scoring vs GT R_cw."""
    return (np.array(R).T @ C).tolist()


def load_scores_poses():
    """Cached low-pct score matrix + per-(pano,room) (t,R) poses for the subset (no GT)."""
    grids = json.load(open(SEEDS / "residuals.json"))
    scores = robust_score.low_percentile_scores(grids)
    cpo = json.load(open(SEEDS / "cpo_cache.json"))
    poses = {p: {r: (v["t"], v["R"]) for r, v in cpo[p]["poses"].items()} for p in cpo}
    return scores, poses


def build_export_seed(rows, md_path, tau=0.10):
    """Deployment seed via the D34 module -> (seed_path, admitted pano names)."""
    scores, poses = load_scores_poses()
    room_order = sorted({r["room"] for r in rows})
    seed_path = SEEDS / "fgpl_export.json"
    admitted = fgpl_export.export_alignment(scores, poses, room_order, md_path, seed_path, tau=tau)
    return seed_path, admitted


def precheck_seed(seed_path, md_path, admitted, rows):
    """GPU-free: load the seed through FGPL's OWN load_panorama_positions; assert positions
    recover to the cached t[:2] and every subset room is seeded. Returns (n_seed, rooms)."""
    sys.path.insert(0, str(paths.FGPL_ROOT / "src" / "pose_estimation"))
    from multiroom_pose_estimation import load_panorama_positions
    positions = load_panorama_positions(str(seed_path), str(md_path), list(admitted))
    cpo = json.load(open(SEEDS / "cpo_cache.json"))
    matches = json.load(open(seed_path))["matches"]
    for m in matches:
        exp = np.array(cpo[m["pano_name"]]["poses"][m["room_label"]]["t"])[:2]
        assert np.allclose(positions[m["pano_name"]], exp, atol=1e-6), m["pano_name"]
    seeded_rooms = {m["room_label"] for m in matches}
    all_rooms = {r["room"] for r in rows}
    assert seeded_rooms == all_rooms, (sorted(seeded_rooms), sorted(all_rooms))
    return len(matches), sorted(all_rooms)
```

- [ ] **Step 2: Run the offline pre-check and verify it passes**

Run:
```bash
conda run -n panopin python -c "
from experiments.fgpl_seed import roundtrip as R, subset, seed_and_config as sc
rows = subset.build_subset(); md = sc.write_identity_metadata()
sp, adm = R.build_export_seed(rows, md)
n, rooms = R.precheck_seed(sp, md, adm, rows)
print('admitted', len(adm), '/ 12 ; rooms seeded', rooms)"
```
Expected: no `AssertionError`; prints something like `admitted 10 / 12 ; rooms seeded ['hallway_3', 'office_1', 'office_4', 'office_5', 'office_6', 'office_7']` (admitted count may be 9–11; all 6 rooms MUST be present). If it errors, stop and report — do not proceed to the GPU run.

- [ ] **Step 3: Commit**

```bash
git add experiments/fgpl_seed/roundtrip.py
git commit -m "feat(roundtrip): fgpl_export seed build + GPU-free FGPL-loader pre-check"
```

---

### Task 2: Per-room coverage metric + cached-arm loader (+ unit test)

**Files:**
- Modify: `experiments/fgpl_seed/roundtrip.py`
- Create: `tests/test_roundtrip.py`

**Interfaces:**
- Consumes: `score._nearest_room`, `paths.WORK`.
- Produces: `per_room_coverage(poses_cw, admitted_rows, all_rooms, cents) -> (n_covered, n_rooms, covered_dict)`; `load_cached_arm(arm, rows) -> {pano: {translation, rotation}|None}`.

- [ ] **Step 1: Write the failing unit test**

```python
# tests/test_roundtrip.py
"""Unit test for the round-trip's per-room coverage metric (pure logic, no GPU/cache)."""
from experiments.fgpl_seed.roundtrip import per_room_coverage


def test_per_room_coverage_counts_true_room_landings():
    # centroids: A at x=0, B at x=10. A pano's pose "lands" in the room of its nearest centroid.
    cents = {"A": [0.0, 0.0, 0.0], "B": [10.0, 0.0, 0.0]}
    admitted_rows = [
        {"pano_name": "a1", "room": "A"},   # lands near A -> covers A
        {"pano_name": "b1", "room": "B"},   # lands near A (wrong) -> does NOT cover B
    ]
    poses_cw = {
        "a1": {"translation": [0.5, 0.0, 0.0], "rotation": None},
        "b1": {"translation": [1.0, 0.0, 0.0], "rotation": None},
    }
    n_cov, n_rooms, covered = per_room_coverage(poses_cw, admitted_rows, ["A", "B"], cents)
    assert n_rooms == 2
    assert covered == {"A": True, "B": False}
    assert n_cov == 1


def test_per_room_coverage_ignores_none_poses():
    cents = {"A": [0.0, 0.0, 0.0]}
    admitted_rows = [{"pano_name": "a1", "room": "A"}]
    n_cov, n_rooms, covered = per_room_coverage({"a1": None}, admitted_rows, ["A"], cents)
    assert n_cov == 0 and covered == {"A": False}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_roundtrip.py -v`
Expected: FAIL with `ImportError: cannot import name 'per_room_coverage'`.

- [ ] **Step 3: Implement the functions**

```python
# append to experiments/fgpl_seed/roundtrip.py
def per_room_coverage(poses_cw, admitted_rows, all_rooms, cents):
    """Rooms whose refined FGPL pose lands in the correct room. A room r is covered iff >=1
    admitted pano whose TRUE room is r has a pose whose nearest centroid == r. Denominator =
    all_rooms (the full subset room set). Returns (n_covered, n_rooms, {room: bool})."""
    true_room = {r["pano_name"]: r["room"] for r in admitted_rows}
    covered = {r: False for r in all_rooms}
    for u, p in poses_cw.items():
        if p is None:
            continue
        tr = true_room[u]
        if score._nearest_room(p["translation"], cents) == tr:
            covered[tr] = True
    return sum(covered.values()), len(all_rooms), covered


def load_cached_arm(arm, rows):
    """Load a previously-run arm's per-pano FGPL poses from work/poses/<arm>/ (no re-run)."""
    base = paths.WORK / "poses" / arm
    out = {}
    for r in rows:
        cp = base / r["pano_name"] / "camera_pose.json"
        if cp.exists():
            d = json.load(open(cp))
            out[r["pano_name"]] = {"translation": d["translation"], "rotation": d["rotation"]}
        else:
            out[r["pano_name"]] = None
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n panopin python -m pytest tests/test_roundtrip.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/fgpl_seed/roundtrip.py tests/test_roundtrip.py
git commit -m "feat(roundtrip): per-room coverage metric + cached-arm loader + unit test"
```

---

### Task 3: `main()` — the FGPL run, scoring, and results (the ~60-min GPU step)

**Files:**
- Modify: `experiments/fgpl_seed/roundtrip.py`
- Create: `experiments/fgpl_seed/ROUNDTRIP_RESULTS.md` (written by the run)

**Interfaces:**
- Consumes: everything from Tasks 1–2 + `sc.write_config`, `run_arm.run_arm`, `score.score_arm`, `s3dis_gt`, `eval.metrics` (via `score_arm`).
- Produces: `main()`; `work/results/roundtrip.json`; `ROUNDTRIP_RESULTS.md`.

- [ ] **Step 1: Implement `main()`**

```python
# append to experiments/fgpl_seed/roundtrip.py
def _score_cw(poses, gt, rows, cents):
    poses_cw = {u: (None if p is None else
                    {"translation": p["translation"], "rotation": _fgpl_rot_to_cw(p["rotation"])})
                for u, p in poses.items()}
    return poses_cw, score.score_arm(poses_cw, gt, rows, cents)


def main():
    rows = subset.build_subset()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    cents = sc.room_centroids(rows)
    md = sc.write_identity_metadata()
    line_map = paths.WORK / "linemap" / "3d_line_map.pkl"
    feat, panos = paths.WORK / "features", paths.WORK / "panos"
    all_rooms = sorted({r["room"] for r in rows})

    # --- build + pre-check the fgpl_export seed (offline) ---
    seed_path, admitted = build_export_seed(rows, md)
    n_seed, _ = precheck_seed(seed_path, md, admitted, rows)
    admitted_rows = [r for r in rows if r["pano_name"] in admitted]
    seed_room = {m["pano_name"]: m["room_label"]
                 for m in json.load(open(seed_path))["matches"]}
    print(f"[precheck] {n_seed} seeds admitted; rooms {all_rooms}", flush=True)

    # --- run FGPL on the admitted panos (the expensive step) ---
    cfg = sc.write_config("fgpl_export", admitted_rows, seed_path, line_map, md, feat, panos)
    poses = run_arm.run_arm(cfg, admitted_rows)
    poses_cw, s = _score_cw(poses, gt, admitted_rows, cents)
    n_cov, n_rooms, covered = per_room_coverage(poses_cw, admitted_rows, all_rooms, cents)

    # right-room slice: median trans over admitted panos whose SEED room == true room
    pu = s["translation"]["per_uuid"]
    right = [r["pano_name"] for r in admitted_rows
             if seed_room.get(r["pano_name"]) == r["room"]]
    import statistics
    rr = [pu[u] for u in right if u in pu]
    trans_right = statistics.median(rr) if rr else None

    # --- cached reference arms, scored on the SAME admitted set ---
    refs = {}
    for arm in ("oracle", "p1"):
        _, rs = _score_cw(load_cached_arm(arm, admitted_rows), gt, admitted_rows, cents)
        refs[arm] = rs

    results = {"fgpl_export": s, "coverage": {"n_covered": n_cov, "n_rooms": n_rooms,
               "covered": covered}, "trans_median_right_room": trans_right,
               "n_right_room": len(rr), "n_admitted": len(admitted_rows),
               "references": refs}
    out = paths.subdir("results") / "roundtrip.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    _write_report(results, all_rooms)
    print(f"\n[roundtrip] coverage {n_cov}/{n_rooms}  "
          f"trans_med={s['translation']['median']:.3f}  "
          f"right-room_med={trans_right if trans_right is None else round(trans_right,3)}  "
          f"wrong_room={s['wrong_room_rate']}\nwrote {out}", flush=True)


def _write_report(res, all_rooms):
    s = res["fgpl_export"]; t, r = s["translation"], s["rotation"]
    cov = res["coverage"]
    def f(x, p=3):
        return "n/a" if x is None else f"{x:.{p}f}"
    lines = [
        "# PanoPin<->FGPL validation round-trip results",
        "",
        f"Scene: area3_seed_ablation (6-room subset, 12 in-frame panos). "
        f"fgpl_export arm gated at tau=0.10; FGPL estimator run on the {res['n_admitted']} "
        f"admitted panos. Fair: seed from cached color scores only (no GT).",
        "",
        f"**Per-room coverage: {cov['n_covered']}/{cov['n_rooms']}** "
        f"(a room counts iff >=1 admitted pano of that TRUE room refines into it). "
        f"Per room: {cov['covered']}.",
        "",
        "| arm | n_localized | trans median | trans median (right-room) | trans mean | trans max | rot median | wrong-room |",
        "|-----|-------------|--------------|---------------------------|------------|-----------|-----------|-----------|",
        f"| fgpl_export (gated) | {s['n_localized']}/{res['n_admitted']} | {f(t['median'])} | "
        f"{f(res['trans_median_right_room'])} ({res['n_right_room']}) | {f(t['mean'])} | "
        f"{f(t['max'])} | {f(r['median'],1)} | {f(s['wrong_room_rate'],2)} |",
    ]
    for arm in ("oracle", "p1"):
        rs = res["references"][arm]; rt, rr2 = rs["translation"], rs["rotation"]
        lines.append(
            f"| {arm} (cached, ref) | {rs['n_localized']}/{res['n_admitted']} | {f(rt['median'])} "
            f"| n/a | {f(rt['mean'])} | {f(rt['max'])} | {f(rr2['median'],1)} | "
            f"{f(rs['wrong_room_rate'],2)} |")
    lines += [
        "",
        "Caveats: oracle/p1 cached poses ran with a 12-pano Voronoi vs this arm's "
        f"{res['n_admitted']}-pano Voronoi (reference context, not a controlled ablation); "
        "Area_3 subset only; tau=0.10 is the D34 Area_3-tuned value (this run also serves as "
        "its through-FGPL precision check).",
    ]
    (paths.HERE / "ROUNDTRIP_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full round-trip (long — ~60 min GPU)**

Run: `conda run -n panopin python -m experiments.fgpl_seed.roundtrip`
Expected: prints `[precheck] N seeds admitted; rooms [...]`, then FGPL `RUN:` logs per admitted pano (~330–390 s each), then a final `[roundtrip] coverage X/6 trans_med=… right-room_med=… wrong_room=…` line and `wrote …/work/results/roundtrip.json`. `ROUNDTRIP_RESULTS.md` is written. Success = a `camera_pose.json` produced for every admitted pano (`n_localized == n_admitted`) and the table populated. Capture the full printed table + coverage line.

- [ ] **Step 3: Commit (script + results)**

```bash
git add experiments/fgpl_seed/roundtrip.py experiments/fgpl_seed/ROUNDTRIP_RESULTS.md experiments/fgpl_seed/work/results/roundtrip.json
git commit -m "feat(roundtrip): FGPL end-to-end run + coverage/accuracy results vs GT"
```
(If `work/` is gitignored, commit only the script + `ROUNDTRIP_RESULTS.md`; note the json path in the report.)

---

### Task 4: Documentation — DECISIONS + PROGRESS

**Files:**
- Modify: `docs/DECISIONS.md`, `docs/PROGRESS.md`

**Interfaces:**
- Consumes: the results from Task 3.
- Produces: durable record per the project protocol.

- [ ] **Step 1: Append a DECISIONS entry (D35)**

Append `## D35 — PanoPin↔FGPL validation round-trip (§8): <one-line result> (2026-07-15)` after D34, same bullet format. Record: the round-trip method (fgpl_export seed → FGPL estimator on the 6-room subset → score vs GT), the **measured** numbers from `ROUNDTRIP_RESULTS.md` (per-room coverage X/6, right-room translation median vs oracle, wrong-room rate, n_admitted), that this doubles as the deferred D34 `tau=0.10` through-FGPL precision check, the caveats (Voronoi-set reference caveat; Area_3 subset only), and the NEXT step (production/Electron wiring — Option B). Do NOT invent numbers — copy them from the results file produced in Task 3.

- [ ] **Step 2: Update PROGRESS.md**

Append a session entry: what shipped (`experiments/fgpl_seed/roundtrip.py`, `tests/test_roundtrip.py`, `ROUNDTRIP_RESULTS.md`), the measured headline (coverage + accuracy), test evidence (`pytest tests/test_roundtrip.py` green + the round-trip run), where it stopped, NEXT (Option B production wiring). Note branch `feat/fgpl-roundtrip`, whether pushed.

- [ ] **Step 3: Commit**

```bash
git add docs/DECISIONS.md docs/PROGRESS.md
git commit -m "docs(roundtrip): record D35 — PanoPin<->FGPL round-trip measured through FGPL"
```

---

## Self-Review

**Spec coverage:** §1 goal (end-to-end run + accuracy + coverage) → Tasks 1+3. §2 reuse of built harness → Global Constraints + all tasks call the existing helpers. §3 fgpl_export arm (steps 1–8) → Task 1 (seed) + Task 3 (run/score). §4 metrics/table (coverage, right-room slice, cached refs) → Task 2 (coverage fn) + Task 3 (table/report). §5 new code + fairness → Tasks 1–3, fairness in Global Constraints. §6 verification (GPU-free pre-check, run) → Task 1 pre-check + Task 3 run; non-goals (ungated, largeval, production) explicitly excluded. §7 success criterion → Task 3 Step 2 expected output. No gaps.

**Placeholder scan:** No TBD/TODO. Task 4 deliberately defers concrete numbers to the run output (correct — the plan must not invent measured results) and instructs copying them from `ROUNDTRIP_RESULTS.md`; all code steps show complete code.

**Type consistency:** `_fgpl_rot_to_cw`, `build_export_seed(rows, md_path, tau)`, `precheck_seed(seed_path, md_path, admitted, rows)`, `per_room_coverage(poses_cw, admitted_rows, all_rooms, cents)`, `load_cached_arm(arm, rows)` used identically where referenced. `poses_cw` shape `{pano: {translation, rotation}|None}` consistent across `_score_cw`, `per_room_coverage`, `score.score_arm`. `export_alignment(scores, poses, room_order, md_path, seed_path, tau=)` matches the shipped D34 signature. Cache/GT field names match the verified shapes in Global Constraints.
