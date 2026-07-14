# Candidate Fusion (F1 verify-select + F2 blend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether letting FGPL geometry propose a multi-room candidate-pose pool and having PanoPin's color render-and-compare score *select* the winner (F1), optionally blended with the geometric score (F2), beats geometry-only selection on the 6-room dev subset.

**Architecture:** Run the FGPL estimator once per arm in **global mode** (whole-scene search, so its top-k pool spans the same-shape rooms) with a flag-gated dump of the refined candidate pool. Offline, color-score every candidate pose with `cpo_adapter.residuals_at_pose` (frame-converted), then apply pure, unit-tested selection rules in a new `src/panopin/fusion.py`. Score each arm vs S3DIS GT with the existing harness. Fusion follows the literature's "geometry proposes, render-and-compare verifies/selects" pattern (InLoc); see `docs/specs/2026-07-14-candidate-fusion-design.md` §6 and `docs/geometry-color-fusion-literature-review.md`.

**Tech Stack:** Python 3.8 (`panopin` / `panopin-gpu` conda envs); numpy, torch, cv2; the vendored CPO primitives (`third_party/cpo`, via `panopin.cpo_adapter`); the FGPL estimator + pose-search primitives in `/home/ruoyu/scan2measure-webframework/src/pose_estimation/`; the stdlib-only `eval/` harness.

## Global Constraints

- **Fairness (D5):** `src/panopin/*` reads only scores/residuals — never GT, room labels, or pano filenames. GT is used ONLY in the oracle/scoring arms under `experiments/`.
- **No third_party edits (D9):** compose CPO primitives; never edit `third_party/cpo/`. The one FGPL-estimator edit (Task 3) is additive + flag-gated + default-off, and lives in the scan2measure repo, not in `src/panopin`.
- **No learned fusion (charter + Kuncheva 2004):** all selection rules are deterministic; at n=12 fixed rules beat trained combiners.
- **Determinism:** call `panopin.determinism.pin()` before any CPO call (single-thread + seed); CPO is not bit-reproducible on CPU (~±0.02, D1).
- **Envs (never mix):** estimator + CPO/`residuals_at_pose` → `panopin-gpu` (`paths.ESTIMATOR_ENV`/`paths.GPU_ENV`); PanoPin unit tests → `panopin`.
- **Frame conventions (fixed, D25):** FGPL emits rotation `Rp = C @ R_wc`, `C = [[0,0,1],[-1,0,0],[0,-1,0]]`. To color-score an FGPL pose, `residuals_at_pose` needs world→camera `R_wc = Cᵀ · Rp`. To score vs GT, camera→world `R_cw = Rpᵀ · C` (that is `run_all._fgpl_rot_to_cw`). A GT pose has `R_cw` directly, so its `R_wc = R_cwᵀ`.
- **`panopin` not pip-installed:** experiment scripts prepend `src/` and repo root to `sys.path` (mirror `cpo_seeds.py:12-14`).
- **`work/` is gitignored:** commit only code + the committed `FUSION_RESULTS.md`.

---

## File Structure

**New (PanoPin):**
- `src/panopin/fusion.py` — pure selection rules over a per-pano candidate list (keys `geom` higher=better, `color` lower=better): `verify_select` (F1), `rrf_select` / `borda_select` / `norm_sum_select` / `tiebreak_select` (F2), `selection_margin` (confidence). Reads only scores.
- `tests/test_fusion.py` — unit tests for all of the above (fast, no GPU).
- `experiments/fgpl_seed/fusion_framecheck.py` — Phase-0 blocking R1 check (GT pose → low color residual; wrong pose → higher).
- `experiments/fgpl_seed/fusion_pool.py` — run estimator global-mode + dump → per-pano candidate pool → `work/seeds/fusion_pool.json`; logs room coverage.
- `experiments/fgpl_seed/fusion_color.py` — color-score every candidate (`residuals_at_pose`, `R_wc=Cᵀ·Rp`, nearest-room cloud) → `work/seeds/fusion_pool_colored.json`.
- `experiments/fgpl_seed/fusion_run.py` — build arms (geom_only / color_only / oracle / fusion_f1 / fusion_f2*) → score vs GT → `work/results/fusion.json` + committed `FUSION_RESULTS.md`.
- `experiments/fgpl_seed/tests/test_fusion_plumbing.py` — offline tests for pool/color/arm plumbing on a tiny synthetic fixture (no GPU).

**Modified:**
- `scan2measure-webframework/src/pose_estimation/multiroom_pose_estimation.py` — additive flag-gated candidate dump + config-driven `top_k` (Task 3), on a scan2measure feature branch.
- `experiments/fgpl_seed/seed_and_config.py` — `write_config` gains `use_local` + `extra` params.
- `experiments/fgpl_seed/run_arm.py` — add `run_arm_pool` that also reads `candidates.json`.

**Deferred to its own plan:** F4 (joint color+geometry refinement) — a distinct differentiable-optimizer subsystem with an open weighting design; plan it after F1/F2 land (see final section).

---

## Phase 0 — Blocking frame-conversion pre-check (R1)

### Task 1: Prove the color scorer + frame conventions on GT poses

**Files:**
- Create: `experiments/fgpl_seed/fusion_framecheck.py`

**Interfaces:**
- Consumes: `subset.build_subset()` rows (`pano_name, room, pano_jpg, cloud_txt`); `s3dis_gt.load_gt` (`{uuid: {location, R_cw, room}}`); `cpo_config.load_cfg`; `cpo_adapter.residuals_at_pose(cfg, pano_path, cloud_path, t, R)`.
- Produces: `work/seeds/framecheck.json` = `{pano: {gt_color, wrong_color}}`; a pass/fail print.

- [ ] **Step 1: Write the check script**

```python
"""BLOCKING R1 pre-check: does residuals_at_pose + the frame conventions give a LOW
color residual at the GT pose and a HIGHER one at a wrong (sibling-room) pose? If this
fails, the color scorer or the R_wc convention is wrong and NO fusion number is valid.
Run in panopin-gpu. -> work/seeds/framecheck.json."""
import os, sys, json, statistics
import numpy as np
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, seed_and_config as sc, paths
from eval import s3dis_gt
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import residuals_at_pose


def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    clouds = {r["room"]: r["cloud_txt"] for r in rows}
    out = {}
    for r in rows:
        name, room = r["pano_name"], r["room"]
        loc = gt[name]["location"]
        R_cw = np.array(gt[name]["R_cw"], float)
        R_wc = R_cw.T                                   # world->camera for residuals_at_pose
        gt_res = float(residuals_at_pose(cfg, r["pano_jpg"], clouds[room], loc, R_wc).mean())
        sib = sc.SIBLING[room]
        wrong_res = float(residuals_at_pose(cfg, r["pano_jpg"], clouds[sib], loc, R_wc).mean())
        out[name] = {"gt_color": gt_res, "wrong_color": wrong_res, "room": room, "sibling": sib}
        print(f"{name} {room:11s} gt={gt_res:.4f}  wrong({sib})={wrong_res:.4f}  "
              f"{'OK' if gt_res < wrong_res else 'X'}", flush=True)
    p = paths.subdir("seeds") / "framecheck.json"
    json.dump(out, open(p, "w"), indent=2)
    n_ok = sum(1 for v in out.values() if v["gt_color"] < v["wrong_color"])
    med_gt = statistics.median(v["gt_color"] for v in out.values())
    print(f"\nGT<wrong on {n_ok}/{len(out)} panos; median GT residual={med_gt:.4f}. Wrote {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin-gpu python -m experiments.fgpl_seed.fusion_framecheck`
Expected: prints one line per pano; **most panos show `gt < wrong` (`OK`)** and a median GT residual clearly below the sibling residual. (A handful of window/blank panos — D23 — may not; that is expected, not a failure. Failure = GT residual is NOT systematically below wrong, or is high everywhere → the `R_wc` convention or `residuals_at_pose` is wrong.)

- [ ] **Step 3: Gate decision**

If GT residuals are NOT systematically below wrong-room residuals, STOP and fix the convention before any further task (per project error policy: undo, re-derive `R_wc`, do not patch incrementally). If OK, proceed.

- [ ] **Step 4: Commit**

```bash
cd /home/ruoyu/PanoPin
git add experiments/fgpl_seed/fusion_framecheck.py
git commit -m "feat(fusion): R1 blocking frame-conversion pre-check (GT pose -> low color residual)"
```

---

## Phase 1 — F1 verify-select

### Task 2: Pure `verify_select` (F1) + tests

**Files:**
- Create: `src/panopin/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Produces: `fusion.verify_select(candidates) -> int` where `candidates` is a list of dicts each with `color` (float, lower=better); returns the index of the minimum-color candidate. Later tasks add `geom` (float, higher=better) and the F2 rules.

- [ ] **Step 1: Write the failing test**

```python
from panopin import fusion

def test_verify_select_picks_min_color():
    cands = [{"geom": 100, "color": 0.5}, {"geom": 90, "color": 0.2}, {"geom": 80, "color": 0.4}]
    assert fusion.verify_select(cands) == 1

def test_verify_select_single():
    assert fusion.verify_select([{"color": 0.9}]) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_fusion.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'panopin.fusion'`).

- [ ] **Step 3: Implement**

```python
"""Pure per-pano candidate-selection rules for FGPL⊕PanoPin fusion (fair: reads only
scores, no GT — D5). Candidate = dict with 'geom' (line-inlier count, higher=better) and
'color' (mean color residual, lower=better). F1 = verify-select (InLoc pattern: color
alone selects). F2 = rank/score blends. See docs/specs/2026-07-14-candidate-fusion-design.md."""


def verify_select(candidates):
    """F1: pick the lowest-color candidate. Geometry only defined the pool."""
    return min(range(len(candidates)), key=lambda i: candidates[i]["color"])
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_fusion.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/panopin/fusion.py tests/test_fusion.py
git commit -m "feat(fusion): verify_select (F1) + tests"
```

### Task 3: Instrument the FGPL estimator (flag-gated candidate dump + config top_k)

**Files:**
- Modify: `/home/ruoyu/scan2measure-webframework/src/pose_estimation/multiroom_pose_estimation.py` (after the candidate sort ~line 471; and `top_k` at ~line 406)

**Interfaces:**
- Produces: when config has `"dump_candidates": true`, writes `<output_dir>/<pano>/candidates.json` = a ranked list of `{"t":[3], "R":[3][3], "n_tight":int, "avg_dist":float, "cost":float}`. When config has `"top_k": N`, uses N instead of the `TOP_K=10` default. Default behavior (both absent) is unchanged.

- [ ] **Step 1: Create a scan2measure feature branch**

```bash
cd /home/ruoyu/scan2measure-webframework
git checkout -b feat/panopin-candidate-dump
```

- [ ] **Step 2: Make `top_k` config-driven**

In `main()`, just after `use_local = cfg.get("use_local_filtering", USE_LOCAL_FILTERING)` (line ~154), add:

```python
    top_k_cfg = int(cfg.get("top_k", TOP_K))
    dump_candidates = bool(cfg.get("dump_candidates", False))
```

Then in the B6 call (line ~406) change `top_k=TOP_K` to `top_k=top_k_cfg`.

- [ ] **Step 3: Dump the ranked candidate pool (additive)**

Immediately AFTER `candidates.sort(key=lambda c: (-c['n_tight'], c['avg_dist']))` (line ~471) and before `best = candidates[0]`, add:

```python
        if dump_candidates:
            pano_output_dir.mkdir(parents=True, exist_ok=True)
            _dump = [{"t": [float(x) for x in c["t"]],
                      "R": [[float(x) for x in row] for row in c["R"]],
                      "n_tight": int(c["n_tight"]), "avg_dist": float(c["avg_dist"]),
                      "cost": float(c["cost"])} for c in candidates]
            with open(pano_output_dir / "candidates.json", "w") as _cf:
                json.dump(_dump, _cf)
```

(`json` is already imported at line 19; `pano_output_dir` is defined at line 301.)

- [ ] **Step 4: Smoke-run the dump on ONE pano (global mode)**

This is the runnable verification (a full unit test needs the GPU + minutes). Reuse an existing PanoPin config but with `use_local_filtering=false`, `dump_candidates=true`, `top_k=30`, and a single pano — the plumbing to build such a config lands in Task 4; for now confirm the edit imports cleanly:

Run: `cd /home/ruoyu/scan2measure-webframework && conda run -n panopin-gpu python -c "import ast; ast.parse(open('src/pose_estimation/multiroom_pose_estimation.py').read()); print('parse OK')"`
Expected: `parse OK` (the full run happens in Task 4's Step 3).

- [ ] **Step 5: Commit (scan2measure)**

```bash
cd /home/ruoyu/scan2measure-webframework
git add src/pose_estimation/multiroom_pose_estimation.py
git commit -m "feat(pose): flag-gated candidate-pool dump + config-driven top_k (default off)"
```

### Task 4: Generate the candidate pool (global mode)

**Files:**
- Modify: `experiments/fgpl_seed/seed_and_config.py` (`write_config` signature), `experiments/fgpl_seed/run_arm.py` (add `run_arm_pool`)
- Create: `experiments/fgpl_seed/fusion_pool.py`

**Interfaces:**
- Consumes: `sc.write_identity_metadata`, `sc.write_seed`, the modified `sc.write_config(..., use_local=False, extra={"dump_candidates": True, "top_k": 30})`; `run_arm.run_arm_pool(config_path, rows)`.
- Produces: `work/seeds/fusion_pool.json` = `{pano_name: {"best": {"t","R"}, "candidates": [ {"t","R","n_tight","avg_dist","cost"} ]}}`; prints per-pano distinct-room coverage.

- [ ] **Step 1: Extend `write_config` with `use_local` + `extra`**

In `seed_and_config.py`, change the signature and body:

```python
def write_config(arm, rows, seed_path, line_map, metadata_path, feat_dir, pano_dir,
                 narrowing=None, use_local=True, extra=None):
    _ensure_density_png()
    cfg = {
        "point_cloud_name": paths.SCENE,
        "pano_names": [r["pano_name"] for r in rows],
        "use_local_filtering": use_local,
        "pkl_3d_path": str(line_map),
        "alignment_path": str(seed_path),
        "metadata_path": str(metadata_path),
        "density_image_path": str(_density_png_path()),
        "point_cloud_path": str(paths.WORK / "clouds" / f"{paths.SCENE}.ply"),
        "features_2d_dir": str(feat_dir),
        "pano_dir": str(pano_dir),
        "output_dir": str(paths.WORK / "poses" / arm),
    }
    if narrowing:
        cfg.update(narrowing)
    if extra:
        cfg.update(extra)
    p = _configs_dir() / f"pose_{arm}.json"
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    return p
```

(Existing callers pass neither `use_local` nor `extra` → defaults preserve today's behavior.)

- [ ] **Step 2: Add `run_arm_pool` to `run_arm.py`**

```python
def run_arm_pool(config_path, rows):
    """Like run_arm but also collect each pano's dumped candidate pool (candidates.json).
    Returns {pano_name: {"best": {translation,rotation}|None, "candidates": [ ... ]}}."""
    import json, subprocess, os
    from experiments.fgpl_seed import paths
    with open(config_path) as f:
        cfg = json.load(f)
    out_base = cfg["output_dir"]
    cmd = ["conda", "run", "--no-capture-output", "-n", paths.ESTIMATOR_ENV,
           "python", str(paths.ESTIMATOR), "--config", str(config_path)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(paths.FGPL_ROOT), env=dict(os.environ), check=False)
    res = {}
    for r in rows:
        d = os.path.join(out_base, r["pano_name"])
        cp, cand = os.path.join(d, "camera_pose.json"), os.path.join(d, "candidates.json")
        best = None
        if os.path.exists(cp):
            j = json.load(open(cp)); best = {"translation": j["translation"], "rotation": j["rotation"]}
        cands = json.load(open(cand)) if os.path.exists(cand) else []
        res[r["pano_name"]] = {"best": best, "candidates": cands}
    return res
```

- [ ] **Step 3: Write `fusion_pool.py`**

```python
"""Generate FGPL's multi-room candidate pool per pano by running the estimator in GLOBAL
mode (whole-scene search -> top-k spans same-shape rooms) with the flag-gated candidate
dump (Task 3). Global mode intentionally lets cross-room false minima populate the pool —
that IS the pool color must disambiguate (InLoc pattern). Run in panopin-gpu.
-> work/seeds/fusion_pool.json (+ per-pano distinct-room coverage)."""
import os, sys, json
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, seed_and_config as sc, run_arm, paths
from eval import s3dis_gt

TOP_K = 30


def _nearest_room(t_xyz, centroids):
    return min(centroids, key=lambda room: sum((t_xyz[i]-centroids[room][i])**2 for i in range(3)))


def main():
    rows = subset.build_subset()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    cents = sc.room_centroids(rows)
    md = sc.write_identity_metadata()
    line_map = paths.WORK / "linemap" / "3d_line_map.pkl"
    feat, panos = paths.WORK / "features", paths.WORK / "panos"
    # Seed content is irrelevant in global mode (the estimator only reads camera_position
    # under use_local); write a trivial alignment so the config path resolves.
    paths.subdir("seeds")
    dummy_seed = paths.WORK / "seeds" / "pool_dummy.json"
    json.dump({"metadata": {}, "matches": []}, open(dummy_seed, "w"))
    cfg = sc.write_config("fusion_pool", rows, dummy_seed, line_map, md, feat, panos,
                          use_local=False, extra={"dump_candidates": True, "top_k": TOP_K})
    res = run_arm.run_arm_pool(cfg, rows)

    pool = {}
    for r in rows:
        entry = res[r["pano_name"]]
        cands = entry["candidates"]
        rooms = {_nearest_room(c["t"], cents) for c in cands}
        gt_in = r["room"] in rooms
        pool[r["pano_name"]] = {"best": entry["best"], "candidates": cands}
        print(f"{r['pano_name']} true={r['room']:11s} n_cand={len(cands):2d} "
              f"rooms={len(rooms)} gt_in_pool={gt_in} {'OK' if gt_in else 'MISS'}", flush=True)

    out = paths.subdir("seeds") / "fusion_pool.json"
    json.dump(pool, open(out, "w"))
    gt_cov = sum(1 for r in rows if r["room"] in {_nearest_room(c["t"], cents)
                 for c in pool[r["pano_name"]]["candidates"]})
    print(f"\nGT-room in pool: {gt_cov}/{len(rows)}   wrote {out}")
```

- [ ] **Step 4: Run pool generation (~70 min GPU)**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin-gpu python -m experiments.fgpl_seed.fusion_pool`
Expected: one line per pano; **the pool spans ≥2 rooms for most panos and `GT-room in pool` is high (≳9/12).** If `GT-room in pool` is low (< ~7/12), global-mode geometry is concentrating the pool — trigger the Task 4b fallback below before proceeding.

- [ ] **Step 5: Commit**

```bash
git add experiments/fgpl_seed/seed_and_config.py experiments/fgpl_seed/run_arm.py experiments/fgpl_seed/fusion_pool.py
git commit -m "feat(fusion): global-mode candidate-pool generation + coverage diagnostic"
```

### Task 4b (contingency — only if Task 4 Step 4 coverage is low): per-room pool (Structure X)

**Files:** extend `fusion_pool.py` with a `--per-room` mode.

- [ ] **Step 1:** If and only if GT-room-in-pool < ~7/12, add a per-room path: for each room, run the estimator restricted to that room (single-room line map built via `build_linemap` on a single-room PLY, or `use_local=True` with a seed at the room centroid), take that room's top candidates, and union across rooms into the pool. This guarantees every room contributes (Structure X, spec §7). Cost ~2–6 h. Document the switch in `FUSION_RESULTS.md`.
- [ ] **Step 2:** Re-run; confirm coverage now high; commit.

*(If Task 4 coverage was adequate, skip 4b entirely and note "global-mode coverage sufficient" in results.)*

### Task 5: Color-score every candidate

**Files:**
- Create: `experiments/fgpl_seed/fusion_color.py`

**Interfaces:**
- Consumes: `work/seeds/fusion_pool.json`; `residuals_at_pose`; `C` (frame const); nearest-room cloud from `subset` + `room_centroids`.
- Produces: `work/seeds/fusion_pool_colored.json` = pool with each candidate augmented `"color": float` (mean residual at `R_wc = Cᵀ·R`, scored against the candidate's nearest-room cloud).

- [ ] **Step 1: Write `fusion_color.py`**

```python
"""Color-score every FGPL candidate pose: color = mean(residuals_at_pose) at R_wc = C^T @ R
(FGPL Rp -> world->camera) against the candidate's nearest-room cloud. Fair: reads only the
pool + clouds, no GT (D5). Run in panopin-gpu. -> work/seeds/fusion_pool_colored.json."""
import os, sys, json
import numpy as np
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, seed_and_config as sc, paths
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import residuals_at_pose

C = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


def _nearest_room(t, cents):
    return min(cents, key=lambda r: sum((t[i]-cents[r][i])**2 for i in range(3)))


def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    clouds = {r["room"]: r["cloud_txt"] for r in rows}
    cents = sc.room_centroids(rows)
    pool = json.load(open(paths.WORK / "seeds" / "fusion_pool.json"))
    row_by_name = {r["pano_name"]: r for r in rows}

    for name, entry in pool.items():
        pano_jpg = row_by_name[name]["pano_jpg"]
        for c in entry["candidates"]:
            room = _nearest_room(c["t"], cents)
            R_wc = (C.T @ np.array(c["R"], float)).tolist()
            res = residuals_at_pose(cfg, pano_jpg, clouds[room], c["t"], R_wc)
            c["color"] = float(res.mean())
            c["room"] = room
            c["geom"] = float(c["n_tight"])          # alias the geom score the selectors read
        print(f"{name}: scored {len(entry['candidates'])} candidates", flush=True)

    out = paths.subdir("seeds") / "fusion_pool_colored.json"
    json.dump(pool, open(out, "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run (~10 min GPU)**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin-gpu python -m experiments.fgpl_seed.fusion_color`
Expected: one line per pano; writes `fusion_pool_colored.json`. Spot-check: `python -c "import json;d=json.load(open('experiments/fgpl_seed/work/seeds/fusion_pool_colored.json'));print(list(d.values())[0]['candidates'][0].keys())"` shows `t R n_tight avg_dist cost color room`.

- [ ] **Step 3: Commit**

```bash
git add experiments/fgpl_seed/fusion_color.py
git commit -m "feat(fusion): color-score every candidate (residuals_at_pose, R_wc=C^T@R)"
```

### Task 6: F1 arm + scoring + results

**Files:**
- Create: `experiments/fgpl_seed/fusion_run.py`, `experiments/fgpl_seed/tests/test_fusion_plumbing.py`

**Interfaces:**
- Consumes: `fusion_pool_colored.json`; `fusion.verify_select`; `score.score_arm`; `s3dis_gt.load_gt`; `run_all._fgpl_rot_to_cw` frame conversion for scoring.
- Produces: `work/results/fusion.json` + committed `experiments/fgpl_seed/FUSION_RESULTS.md`; arms `geom_only`, `color_only`, `oracle`, `fusion_f1`.

- [ ] **Step 1: Write a plumbing test (offline, synthetic fixture)**

```python
import os, sys
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "src"))
from experiments.fgpl_seed import fusion_run

def test_pick_pose_uses_selector_index():
    cands = [{"t":[0,0,0],"R":[[1,0,0],[0,1,0],[0,0,1]],"geom":5,"color":0.9},
             {"t":[9,9,9],"R":[[1,0,0],[0,1,0],[0,0,1]],"geom":3,"color":0.1}]
    # verify_select -> index 1 (min color)
    t, R = fusion_run.pick_pose(cands, "fusion_f1")
    assert t == [9,9,9]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest experiments/fgpl_seed/tests/test_fusion_plumbing.py -q`
Expected: FAIL (`fusion_run` has no `pick_pose`).

- [ ] **Step 3: Write `fusion_run.py`**

```python
"""Build fusion arms from the colored candidate pool, score vs GT, emit FUSION_RESULTS.md.
Arms: geom_only (estimator best), color_only (PanoPin min-residual room's own pose),
oracle (GT pos seed), fusion_f1 (verify-select over the pool). Fair: selection reads only
scores (D5); GT only in oracle/scoring. Rotations converted C-convention -> camera->world
for scoring. Offline (reuses cached pools). Run in panopin (no GPU needed)."""
import os, sys, json, statistics
import numpy as np
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, seed_and_config as sc, score, paths
from experiments.fgpl_seed.run_all import _fgpl_rot_to_cw
from eval import s3dis_gt
from panopin import fusion

SELECTORS = {"fusion_f1": lambda cands: fusion.verify_select(cands)}


def pick_pose(candidates, arm):
    """Return (t, R) for the arm's selector over a colored candidate list."""
    i = SELECTORS[arm](candidates)
    return candidates[i]["t"], candidates[i]["R"]


def _arm_poses(arm, pool, rows, gt, cpo):
    poses = {}
    for r in rows:
        name = r["pano_name"]
        if arm == "geom_only":
            b = pool[name]["best"]; poses[name] = None if b is None else b
        elif arm == "oracle":
            # Upper bound: the GT pose itself. GT R_cw is already camera->world; store it
            # directly and let _to_cw pass it through unchanged.
            poses[name] = {"translation": gt[name]["location"], "rotation": gt[name]["R_cw"]}
        elif arm == "color_only":
            room = cpo[name]["room"]; p = cpo[name]["poses"][room]
            poses[name] = {"translation": p["t"], "rotation": p["R"]}
        else:  # fusion arms
            cands = pool[name]["candidates"]
            if not cands:
                poses[name] = None; continue
            t, R = pick_pose(cands, arm)
            poses[name] = {"translation": t, "rotation": R}
    return poses


def _to_cw(arm, poses):
    """Convert each arm's rotation to camera->world for scoring vs GT R_cw.
      - oracle: already camera->world -> pass through.
      - color_only: CPO localize_pair returns world->camera (R_wc) -> transpose.
      - geom_only / fusion_*: FGPL C-convention (Rp = C @ R_wc) -> _fgpl_rot_to_cw (Rp^T @ C)."""
    out = {}
    for u, p in poses.items():
        if p is None:
            out[u] = None; continue
        if arm == "oracle":
            out[u] = p
        elif arm == "color_only":
            out[u] = {"translation": p["translation"], "rotation": np.array(p["rotation"]).T.tolist()}
        else:
            out[u] = {"translation": p["translation"], "rotation": _fgpl_rot_to_cw(p["rotation"])}
    return out


def main():
    rows = subset.build_subset()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    cents = sc.room_centroids(rows)
    cpo = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    pool = json.load(open(paths.WORK / "seeds" / "fusion_pool_colored.json"))
    arms = ["oracle", "geom_only", "color_only", "fusion_f1"]
    results = {}
    for arm in arms:
        poses = _arm_poses(arm, pool, rows, gt, cpo)
        results[arm] = score.score_arm(_to_cw(arm, poses), gt, rows, cents)
    out = paths.subdir("results") / "fusion.json"
    json.dump(results, open(out, "w"), indent=2)
    lines = ["# FGPL⊕PanoPin candidate fusion — F1 (n=%d)\n" % len(rows),
             "| arm | trans median | trans mean | rot median | wrong-room |",
             "|-----|-------------:|-----------:|-----------:|-----------:|"]
    for arm in arms:
        s = results[arm]; t, rr = s["translation"], s["rotation"]
        lines.append(f"| {arm} | {t['median']:.3f} | {t['mean']:.3f} | {rr['median']:.1f} | {s['wrong_room_rate']} |")
    md = "\n".join(lines) + "\n"
    print(md)
    open(os.path.join(_HERE, "FUSION_RESULTS.md"), "w").write(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the plumbing test**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest experiments/fgpl_seed/tests/test_fusion_plumbing.py -q`
Expected: PASS.

- [ ] **Step 5: Run F1 end-to-end + read the table**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m experiments.fgpl_seed.fusion_run`
Expected: prints the arm table + writes `FUSION_RESULTS.md`. **Read it:** the load-bearing question is whether `fusion_f1` beats `geom_only` on wrong-room rate and translation median, and approaches `oracle`. Record the numbers; do NOT over-claim at n=12 (D26/D29 discipline).

- [ ] **Step 6: Commit**

```bash
git add experiments/fgpl_seed/fusion_run.py experiments/fgpl_seed/tests/test_fusion_plumbing.py experiments/fgpl_seed/FUSION_RESULTS.md
git commit -m "feat(fusion): F1 verify-select arm + scoring + FUSION_RESULTS"
```

---

## Phase 2 — F2 blend (bracket F1)

### Task 7: F2 selection rules + tests

**Files:**
- Modify: `src/panopin/fusion.py`, `tests/test_fusion.py`

**Interfaces:**
- Produces: `fusion.rrf_select(candidates, k=60)`, `fusion.borda_select(candidates)`, `fusion.norm_sum_select(candidates, w=0.5)`, `fusion.tiebreak_select(candidates, margin)`, `fusion.selection_margin(candidates, key, lower_better=True)`. All take the same candidate dicts (`geom` higher=better, `color` lower=better) and return an index (margin returns a float).

- [ ] **Step 1: Write failing tests**

```python
def test_rrf_prefers_agreement():
    # c1 is 2nd on geom but best on color -> reciprocal-rank sum highest
    cands = [{"geom": 100, "color": 0.9}, {"geom": 90, "color": 0.1}, {"geom": 80, "color": 0.5}]
    assert fusion.rrf_select(cands, k=60) == 1

def test_borda_min_ranksum():
    cands = [{"geom": 100, "color": 0.9}, {"geom": 90, "color": 0.1}, {"geom": 80, "color": 0.5}]
    # geom ranks 0,1,2 ; color ranks 2,0,1 ; sums 2,1,3 -> idx1
    assert fusion.borda_select(cands) == 1

def test_norm_sum_w1_is_geom_only():
    cands = [{"geom": 100, "color": 0.9}, {"geom": 90, "color": 0.1}]
    assert fusion.norm_sum_select(cands, w=1.0) == 0   # all geom weight -> highest geom

def test_tiebreak_breaks_by_color_within_margin():
    cands = [{"geom": 100, "color": 0.9}, {"geom": 98, "color": 0.1}, {"geom": 50, "color": 0.05}]
    # within margin 5 of best geom(100): idx0,1 ; break by color -> idx1 (0.1 < 0.9)
    assert fusion.tiebreak_select(cands, margin=5) == 1

def test_selection_margin_gap():
    cands = [{"color": 0.1}, {"color": 0.4}, {"color": 0.9}]
    assert abs(fusion.selection_margin(cands, "color") - 0.3) < 1e-9
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_fusion.py -q`
Expected: FAIL (functions undefined).

- [ ] **Step 3: Implement (append to `fusion.py`)**

```python
def _ranks(values, ascending):
    """Rank each value (0 = best). ascending=True => smaller value is rank 0."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=not ascending)
    ranks = [0] * len(values)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def _geom_color_ranks(candidates):
    rg = _ranks([c["geom"] for c in candidates], ascending=False)   # highest geom = rank 0
    rc = _ranks([c["color"] for c in candidates], ascending=True)   # lowest color = rank 0
    return rg, rc


def rrf_select(candidates, k=60):
    """F2: Reciprocal Rank Fusion over geom + color ranks; pick the max RRF."""
    rg, rc = _geom_color_ranks(candidates)
    rrf = [1.0 / (k + rg[i]) + 1.0 / (k + rc[i]) for i in range(len(candidates))]
    return max(range(len(candidates)), key=lambda i: rrf[i])


def borda_select(candidates):
    """F2: pick the minimum rank-sum (Borda)."""
    rg, rc = _geom_color_ranks(candidates)
    return min(range(len(candidates)), key=lambda i: rg[i] + rc[i])


def _zscore(vals):
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    s = var ** 0.5 or 1.0
    return [(v - m) / s for v in vals]


def norm_sum_select(candidates, w=0.5):
    """F2: z-score-normalized weighted sum; geom higher=better, color negated. Pick max.
    w=1 -> geom only; w=0 -> color only (== verify_select up to ties)."""
    g = _zscore([c["geom"] for c in candidates])
    c = _zscore([cd["color"] for cd in candidates])
    fused = [w * g[i] + (1 - w) * (-c[i]) for i in range(len(candidates))]
    return max(range(len(candidates)), key=lambda i: fused[i])


def tiebreak_select(candidates, margin):
    """F2 (conservative): among candidates within `margin` geom of the best, pick lowest color."""
    best_geom = max(cd["geom"] for cd in candidates)
    pool = [i for i, cd in enumerate(candidates) if cd["geom"] >= best_geom - margin]
    return min(pool, key=lambda i: candidates[i]["color"])


def selection_margin(candidates, key, lower_better=True):
    """Confidence = gap between the best and second-best `key` value (∞ if <2 candidates)."""
    vals = sorted((cd[key] for cd in candidates), reverse=not lower_better)
    return float("inf") if len(vals) < 2 else abs(vals[0] - vals[1])
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_fusion.py -q`
Expected: PASS (all fusion tests).

- [ ] **Step 5: Commit**

```bash
git add src/panopin/fusion.py tests/test_fusion.py
git commit -m "feat(fusion): F2 rules — RRF, Borda, norm-sum, tiebreak, selection_margin + tests"
```

### Task 8: Add F2 arms to the run + compare

**Files:**
- Modify: `experiments/fgpl_seed/fusion_run.py`

**Interfaces:**
- Consumes: the Task-7 selectors. Produces: extra arms `fusion_rrf`, `fusion_borda`, `fusion_normsum`, `fusion_tiebreak` in `fusion.json` + `FUSION_RESULTS.md`, plus a per-arm confidence-margin column.

- [ ] **Step 1: Register the F2 selectors**

In `fusion_run.py`, extend `SELECTORS`:

```python
SELECTORS = {
    "fusion_f1":       lambda cands: fusion.verify_select(cands),
    "fusion_rrf":      lambda cands: fusion.rrf_select(cands, k=60),
    "fusion_borda":    lambda cands: fusion.borda_select(cands),
    "fusion_normsum":  lambda cands: fusion.norm_sum_select(cands, w=0.5),
    "fusion_tiebreak": lambda cands: fusion.tiebreak_select(cands, margin=5),
}
```

and add them to `arms` in `main()`:

```python
    arms = ["oracle", "geom_only", "color_only", "fusion_f1",
            "fusion_rrf", "fusion_borda", "fusion_normsum", "fusion_tiebreak"]
```

- [ ] **Step 2: Re-run the plumbing test (ensure the new keys resolve)**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest experiments/fgpl_seed/tests/test_fusion_plumbing.py -q`
Expected: PASS.

- [ ] **Step 3: Re-run the comparison (offline, seconds)**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m experiments.fgpl_seed.fusion_run`
Expected: the table now has all arms. **Read it:** does any blend (F2) beat `fusion_f1` — especially on the window/blank panos where color-alone should over-commit? Note that at n=12 small differences are within noise (D26/D29); report which arm most reduces wrong-room + translation, and the confidence-margin behavior.

- [ ] **Step 4: Finalize `FUSION_RESULTS.md` with a written verdict**

Add 3–5 sentences to `FUSION_RESULTS.md`: F1 vs geom_only vs oracle; whether F2 blends help; the n=12 caveat; and whether the pool coverage (Task 4) was global-mode or the 4b per-room fallback.

- [ ] **Step 5: Commit**

```bash
git add experiments/fgpl_seed/fusion_run.py experiments/fgpl_seed/FUSION_RESULTS.md
git commit -m "feat(fusion): F2 blend arms (RRF/Borda/normsum/tiebreak) + verdict"
```

---

## Deferred: Phase F4 — joint color+geometry refinement (separate plan)

F4 (colored-ICP-style joint `E = w_g·E_line + w_c·E_color` refinement to fix the 90° rotation aliasing) is a **distinct subsystem** — a differentiable joint optimizer — with an **open weighting design** (the spec leaves the geom↔color balance to residual-covariance weighting, not yet pinned). Per the writing-plans Scope Check, it gets its own spec-iteration + plan **after F1/F2 land**, because (a) its value is conditional on F1 producing a good selected pose to refine, and (b) F1/F2 results tell us whether rotation is still the dominant residual worth the extra optimizer. Write `docs/specs/2026-07-14-...-joint-refine.md` → `docs/plans/...` at that point. Precedents to reuse: PICCOLO's differentiable sampling loss (color term), FGPL `pose_refine` (line term), Gutiérrez-Gómez covariance weighting.

---

## Self-Review

**Spec coverage:** F1 verify-select → Tasks 2–6. F2 rank/score blend (RRF/Borda/norm-sum/tiebreak, no learned weight) → Tasks 7–8. Confidence via selection margin → `selection_margin` (Task 7), reported in Task 8. Structure-X per-room generation → Task 4b contingency (global pooled is the primary, a plan-time decision documented in the header + §5 of the spec). R1 frame check → Task 1 (blocking). F3 (reliability-weight) and F4 (joint refine) → explicitly deferred (spec §6 marks F3 deferred; F4 gets its own plan). Fairness/D9/determinism → Global Constraints, enforced per task.

**Placeholder scan:** none — every code step has complete code; `fusion_pool.py` Step 3 has one awkward placeholder-avoidance line (the `if False` dummy-seed guard) that is intentional and runnable; simplify to a plain dummy-seed write during execution if preferred.

**Type consistency:** candidate dicts carry `t` (list[3]), `R` (list[3][3]), `n_tight`/`avg_dist`/`cost` (from the estimator dump, Task 3), and `color`/`room`/`geom` (added in Task 5's `fusion_color.py`, where `geom = float(n_tight)`). The `fusion.*` selectors read `geom` (higher=better) + `color` (lower=better), both present on every candidate after Task 5, so `fusion_run` hands the colored candidates straight to `pick_pose` with no further remapping. `selection_margin` takes a key name. Frame conversions are centralised in `_to_cw` (oracle pass-through, color_only transpose, FGPL `_fgpl_rot_to_cw`) so each arm is converted exactly once.
