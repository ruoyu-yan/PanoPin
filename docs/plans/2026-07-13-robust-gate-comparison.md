# Robust-Gate Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare, at 100%-precision coverage, PanoPin's minmax gate on CPO's **mean** loss (a) vs the same gate on a **robust per-point** statistic of the color residuals (c), to decide which lets PanoPin confidently commit more panos to FGPL without ever handing a wrong room.

**Architecture:** Both methods share the existing `calibrate.minmax_scores` gate and differ only in the per-room score. (a) uses `cache[pano]["per_room"]` (mean loss) — no new compute. (c) captures per-point residuals at each room's already-refined cached pose (one cheap sampling forward, no Adam, no score-maps), summarizes each as a 101-point percentile grid, then re-scores rooms by a robust statistic swept offline. Comparison is a precision/coverage analysis; no FGPL re-run.

**Tech Stack:** Python 3.8, PyTorch (CPO primitives via `panopin._cpo_path` → `utils`, `data_utils`, `cpo.sampling_loss`), numpy. Envs: `panopin-gpu` (residual capture, CPO), `panopin` (pure-Python tests + offline analysis).

## Global Constraints

- **Fairness (D5):** `src/panopin/*` reads only losses/residuals/manifest — NEVER GT (room labels/pose). GT is used ONLY in `experiments/` for scoring.
- **Do NOT edit `third_party/cpo/`** (D9) — compose its primitives.
- **`panopin` is not pip-installed** — scripts prepend `src/` and repo root to `sys.path` (mirror `experiments/fgpl_seed/cpo_seeds.py:5-7`).
- **CPO is not bit-reproducible** (~±0.02 loss, D1) — but the residual *sampling* at a FIXED pose IS deterministic (no Adam/search); fidelity asserts to 1e-4.
- **Residual grid = list of 101 floats** = percentiles p0..p100 of the per-point residual vector. Empty-residual guard: store `[999.0]*101`.
- **`work/` is gitignored** — committed numbers live in `experiments/fgpl_seed/ROBUST_RESULTS.md`.
- Commit on branch `feat/fgpl-seed-ablation`.

---

### Task 1: `robust_score.py` — mean→robust re-score of residual grids (pure, no GPU)

**Files:**
- Create: `src/panopin/robust_score.py`
- Test: `tests/test_robust_score.py`

**Interfaces:**
- Produces: `robust_scores(grids: dict, stat: str, **params) -> dict` where `grids = {pano: {room: [101 floats]}}` and the return is `{pano: {room: float}}` (lower = better). `stat ∈ {"median","low_percentile","trimmed_mean"}`; params: `low_percentile` needs `q:int` (0..100), `trimmed_mean` needs `k:int` (0..100, drop top-k%).
- Consumes: nothing (leaf module, numpy only).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_robust_score.py
import numpy as np
from panopin import robust_score as rs

# grid = percentiles p0..p100 of a residual vector
def _grid(vals):
    return np.percentile(np.asarray(vals, float), np.arange(101)).tolist()

def test_stat_values():
    g = _grid(list(range(101)))          # residuals 0..100 -> percentiles ~ 0..100
    grids = {"pA": {"r": g}}
    assert abs(rs.robust_scores(grids, "median")["pA"]["r"] - 50.0) < 1e-6
    assert abs(rs.robust_scores(grids, "low_percentile", q=10)["pA"]["r"] - 10.0) < 1e-6
    # trimmed_mean k=0 == mean of full grid (~50); k=50 drops top half -> mean of p0..p50 (~25)
    assert abs(rs.robust_scores(grids, "trimmed_mean", k=0)["pA"]["r"] - 50.0) < 1.0
    assert abs(rs.robust_scores(grids, "trimmed_mean", k=50)["pA"]["r"] - 25.0) < 1.0

def test_discrimination_true_room_beats_loss_sink():
    # true room: strong low core (most points match ~0.05) + minority window outliers (~0.8)
    true = _grid([0.05]*80 + [0.8]*20)
    # loss-sink: mediocre everywhere (~0.30) -> higher low-core, similar mean
    sink = _grid([0.30]*100)
    grids = {"p": {"true": true, "sink": sink}}
    # under low_percentile the true room's core wins; under plain mean it would NOT (0.20 vs 0.30 is close,
    # and window-heavy panos can flip it) -- this is the whole hypothesis.
    lp = rs.robust_scores(grids, "low_percentile", q=20)["p"]
    assert lp["true"] < lp["sink"]
    tm = rs.robust_scores(grids, "trimmed_mean", k=30)["p"]   # drop the window outliers
    assert tm["true"] < tm["sink"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n panopin python -m pytest tests/test_robust_score.py -q`
Expected: FAIL (`ModuleNotFoundError: panopin.robust_score`).

- [ ] **Step 3: Implement `robust_score.py`**

```python
# src/panopin/robust_score.py
"""Re-score CPO room candidates by a ROBUST statistic of their per-point color
residuals instead of the mean (D26 follow-up). A residual grid is 101 percentiles
(p0..p100) of one (pano, room) residual vector. Fair: reads only residuals, no GT (D5)."""

def _stat(grid, stat, **params):
    if len(grid) != 101:
        raise ValueError(f"grid must be 101 percentiles, got {len(grid)}")
    if stat == "median":
        return float(grid[50])
    if stat == "low_percentile":
        q = int(params["q"])
        return float(grid[q])
    if stat == "trimmed_mean":
        k = int(params["k"])                      # drop the top-k% highest residuals
        kept = grid[:101 - k] if k > 0 else grid
        return float(sum(kept) / len(kept))
    raise ValueError(f"unknown stat {stat!r}")

def robust_scores(grids, stat, **params):
    """grids: {pano: {room: [101 floats]}} -> {pano: {room: score}} (lower = better)."""
    return {p: {r: _stat(g, stat, **params) for r, g in rooms.items()}
            for p, rooms in grids.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n panopin python -m pytest tests/test_robust_score.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/panopin/robust_score.py tests/test_robust_score.py
git commit -m "feat(robust): per-point residual -> robust room score (median/low-pct/trimmed-mean)"
```

---

### Task 2: `residuals_at_pose` — per-point residuals at a fixed pose (GPU, fidelity-gated)

**Files:**
- Modify: `src/panopin/cpo_adapter.py` (append one function; imports already present)
- Create: `experiments/fgpl_seed/check_residual_fidelity.py`

**Interfaces:**
- Produces: `residuals_at_pose(cfg, pano_path, cloud_path, t, R) -> np.ndarray` (1-D per-point residuals). `t` is a length-3 array-like; `R` a 3×3 array-like (the returned CPO camera pose, same convention as `cache[pano]["poses"][room]`).
- Consumes: CPO `utils.{cloud2idx, refine_sampling_coords, sample_from_img}`, `data_utils.read_txt_pcd`, `color_utils.{color_match, color_mod}` (already imported at top of `cpo_adapter.py`). Fidelity check consumes `cpo.sampling_loss.sampling_loss`, `utils.{generate_trans_points, generate_rot_points, histogram_pose_search}`, `dict_utils.get_init_dict_cpo`.

- [ ] **Step 1: Write the failing fidelity check**

`experiments/fgpl_seed/check_residual_fidelity.py` — asserts the mean of `residuals_at_pose` at a candidate pose equals CPO's own `sampling_loss` scalar at that same pose (proves the sampling replication is faithful). Run in `panopin-gpu`.

```python
"""Fidelity gate for residuals_at_pose: mean(residuals) == cpo.sampling_loss scalar at
the SAME pose, to 1e-4. Deterministic (fixed pose, no Adam). Run in panopin-gpu."""
import os, sys, numpy as np, torch, cv2
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin import cpo_adapter
import data_utils
from dict_utils import get_init_dict_cpo
from utils import generate_trans_points, generate_rot_points, histogram_pose_search
from cpo.sampling_loss import sampling_loss

def main():
    pin()
    cfg = load_cfg(sample_rate=30)
    row = subset.build_subset()[0]
    pano, cloud = row["pano_jpg"], row["cloud_txt"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    xyz_np, rgb_np = data_utils.read_txt_pcd(cloud, sample_rate=getattr(cfg, 'sample_rate', 1))
    xyz = torch.from_numpy(xyz_np).float().to(device)
    rgb = torch.from_numpy(rgb_np).float().to(device)
    orig = cv2.resize(cv2.cvtColor(cv2.imread(pano), cv2.COLOR_BGR2RGB), (2048, 1024))
    img = (torch.from_numpy(orig).float() / 255.).to(device)

    init_dict = get_init_dict_cpo(cfg)
    rot = generate_rot_points(init_dict, device=device)
    trans = generate_trans_points(xyz, init_dict, device=device)
    input_trans, input_rot = histogram_pose_search(
        img, xyz, rgb, trans, rot, 1, init_dict['num_split_h'], init_dict['num_split_w'],
        None, init_dict['sin_hist'])

    t_used, R_used, L = sampling_loss(img, xyz, rgb, input_trans, input_rot, 0, cfg, return_list=True)
    t_used = t_used.detach().numpy().reshape(3); R_used = R_used.detach().numpy().reshape(3, 3)
    L = float(L.detach())

    resid = cpo_adapter.residuals_at_pose(cfg, pano, cloud, t_used, R_used)
    m = float(np.mean(resid))
    print(f"sampling_loss L = {L:.6f}   mean(residuals) = {m:.6f}   |diff| = {abs(m - L):.2e}   n_resid = {len(resid)}")
    assert abs(m - L) < 1e-4, f"FIDELITY FAIL: {abs(m - L):.2e} >= 1e-4"
    print("FIDELITY OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run -n panopin-gpu python -m experiments.fgpl_seed.check_residual_fidelity`
Expected: FAIL (`AttributeError: module 'panopin.cpo_adapter' has no attribute 'residuals_at_pose'`).

- [ ] **Step 3: Implement `residuals_at_pose`** — append to `src/panopin/cpo_adapter.py`

```python
def residuals_at_pose(cfg, pano_path, cloud_path, t, R):
    """Per-point color residuals ||sample_rgb - cloud_rgb|| at a FIXED given pose (t,R),
    replicating cpo.sampling_loss's sampling (sampling_loss.py:189-203) WITHOUT the mean.
    Composes CPO primitives only (no third_party edit, D9). Returns a 1-D numpy array."""
    from utils import cloud2idx, refine_sampling_coords, sample_from_img
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sample_rate = getattr(cfg, 'sample_rate', 1)

    xyz_np, rgb_np = data_utils.read_txt_pcd(cloud_path, sample_rate=sample_rate)
    xyz = torch.from_numpy(xyz_np).float().to(device)
    rgb = torch.from_numpy(rgb_np).float().to(device)

    orig_img = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (2048, 1024))
    sharpen_color = getattr(cfg, 'sharpen_color', False)
    match_color = getattr(cfg, 'match_color', False)
    num_bins = getattr(cfg, 'num_bins', 256)
    if sharpen_color or match_color:
        mod_img = (torch.from_numpy(orig_img).float() / 255.).to(device)
        if match_color:
            new_img = color_match(mod_img, rgb); orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)
        if sharpen_color:
            new_img, rgb = color_mod(mod_img, rgb, num_bins); orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)

    mdh = getattr(cfg, 'main_downsample_h', 1); mdw = getattr(cfg, 'main_downsample_w', 1)
    img = cv2.resize(orig_img, (orig_img.shape[1] // mdw, orig_img.shape[0] // mdh))
    img = (torch.from_numpy(img).float() / 255.).to(device)

    t_col = torch.as_tensor(np.asarray(t, dtype=np.float32), device=device).reshape(3, 1)
    R_t = torch.as_tensor(np.asarray(R, dtype=np.float32), device=device).reshape(3, 3)

    new_xyz = torch.transpose(xyz, 0, 1) - t_col
    new_xyz = torch.transpose(torch.matmul(R_t, new_xyz), 0, 1)
    coord_arr = cloud2idx(new_xyz)
    filter_factor = getattr(cfg, 'filter_factor', 1)
    filtered_idx = refine_sampling_coords(
        coord_arr, torch.norm(new_xyz, dim=-1), rgb,
        quantization=(img.shape[0] // filter_factor, img.shape[1] // filter_factor))
    coord_arr = coord_arr[filtered_idx]
    refined_rgb = rgb[filtered_idx]
    sample_rgb = sample_from_img(img, coord_arr)
    mask = torch.sum(sample_rgb == 0, dim=1) != 3
    residuals = torch.norm(sample_rgb[mask] - refined_rgb[mask], dim=-1)
    return residuals.detach().cpu().numpy()
```

- [ ] **Step 4: Run the fidelity check to verify it passes**

Run: `conda run -n panopin-gpu python -m experiments.fgpl_seed.check_residual_fidelity`
Expected: prints `FIDELITY OK` and `|diff|` < 1e-4.
(If it fails: per CLAUDE.md error policy, do NOT patch in place — revert this function and re-derive the sampling replication from `sampling_loss.py:189-203`.)

- [ ] **Step 5: Commit**

```bash
git add src/panopin/cpo_adapter.py experiments/fgpl_seed/check_residual_fidelity.py
git commit -m "feat(robust): residuals_at_pose (fixed-pose per-point residuals) + fidelity gate"
```

---

### Task 3: `robust_capture.py` — residual grids for all 72 (pano, room) pairs (GPU)

**Files:**
- Create: `experiments/fgpl_seed/robust_capture.py`

**Interfaces:**
- Consumes: `cpo_adapter.residuals_at_pose` (Task 2), `cache[pano]["poses"][room] = {"t":[3],"R":[[3x3]]}` from `work/seeds/cpo_cache.json`, `subset.build_subset` (pano→jpg, room→cloud).
- Produces: `work/seeds/residuals.json = {pano: {room: [101 floats]}}` (percentile grids).

- [ ] **Step 1: Write the capture script**

```python
"""Capture per-point residual percentile grids at each room's cached refined pose.
Reuses cpo_cache poses -> one sampling forward per (pano,room) (no Adam/search), ~3 min.
Output: work/seeds/residuals.json = {pano: {room: [101 percentiles]}}. Run in panopin-gpu."""
import os, sys, json, time, numpy as np
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin import cpo_adapter

def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    clouds = {r["room"]: r["cloud_txt"] for r in rows}
    cache = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    grids, t0 = {}, time.time()
    for r in rows:
        pano_name, pano_jpg = r["pano_name"], r["pano_jpg"]
        grids[pano_name] = {}
        for room, cloud in clouds.items():
            pose = cache[pano_name]["poses"][room]
            resid = cpo_adapter.residuals_at_pose(cfg, pano_jpg, cloud, pose["t"], pose["R"])
            if len(resid) == 0:
                grid = [999.0] * 101
            else:
                grid = np.percentile(resid, np.arange(101)).astype(float).tolist()
            grids[pano_name][room] = grid
        print(f"{pano_name} done ({len(clouds)} rooms)", flush=True)
    out = paths.subdir("seeds") / "residuals.json"
    with open(out, "w") as f:
        json.dump(grids, f)
    print(f"\nwrote {out}  ({(time.time()-t0)/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run capture (produces the artifact)**

Run: `conda run -n panopin-gpu python -m experiments.fgpl_seed.robust_capture`
Expected: 12 `... done (6 rooms)` lines, then `wrote .../residuals.json`, ~3 min.

- [ ] **Step 3: Verify the artifact shape**

Run:
```bash
conda run -n panopin python -c "
import json; d=json.load(open('experiments/fgpl_seed/work/seeds/residuals.json'))
assert len(d)==12, len(d)
p=next(iter(d)); assert len(d[p])==6, len(d[p])
r=next(iter(d[p])); assert len(d[p][r])==101, len(d[p][r])
assert all(d[p][r][i] <= d[p][r][i+1] for i in range(100)), 'percentiles not monotonic'
print('residuals.json OK: 12 panos x 6 rooms x 101-grid, monotonic')"
```
Expected: `residuals.json OK: ...`.

- [ ] **Step 4: Commit** (script only; `work/` is gitignored)

```bash
git add experiments/fgpl_seed/robust_capture.py
git commit -m "feat(robust): capture per-point residual percentile grids at cached poses"
```

---

### Task 4: `robust_analysis.py` — precision/coverage comparison + `ROBUST_RESULTS.md` (offline)

**Files:**
- Create: `experiments/fgpl_seed/robust_analysis.py`
- Create: `experiments/fgpl_seed/ROBUST_RESULTS.md` (written by the script's output; commit the numbers)

**Interfaces:**
- Consumes: `robust_score.robust_scores` (Task 1), `calibrate.minmax_scores` (existing), `work/seeds/{cpo_cache.json,residuals.json}`, `subset.build_subset` (GT room per pano — experiments-side, allowed).
- Produces: a printed table + `ROBUST_RESULTS.md`. Metric per method: `prefix_correct` = number of leading panos in confidence order that are correct before the FIRST wrong one = coverage-at-100%-precision × n. Higher is better.

- [ ] **Step 1: Write the analysis script**

```python
"""Precision/coverage compare: minmax gate on MEAN loss (a) vs on ROBUST residual
scores (c). Both use calibrate.minmax_scores; only the score matrix differs. Winner =
most panos confidently+correctly committed before the first wrong room. Offline, no GPU."""
import os, sys, json
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from panopin import calibrate, robust_score

def curve(score_matrix, true_room):
    """Return (prefix_correct, recall_all): add panos in ascending confidence (most
    confident first); prefix_correct = leading all-correct run length; recall_all =
    total correct at full coverage."""
    ss = calibrate.minmax_scores(score_matrix)
    winner = {p: min(ss[p], key=lambda r: ss[p][r]) for p in ss}
    conf = {p: ss[p][winner[p]] for p in ss}
    order = sorted(conf, key=lambda p: conf[p])
    prefix, still = 0, True
    recall = 0
    for i, p in enumerate(order):
        ok = winner[p] == true_room[p]
        recall += ok
        if still and ok:
            prefix += 1
        elif still:
            still = False
    return prefix, recall

def main():
    rows = subset.build_subset()
    true_room = {r["pano_name"]: r["room"] for r in rows}
    n = len(rows)
    cache = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    grids = json.load(open(paths.WORK / "seeds" / "residuals.json"))

    methods = [("a: mean-loss", {p: cache[p]["per_room"] for p in cache})]
    for stat, params, label in [
        ("median", {}, "c: median"),
        ("low_percentile", {"q": 10}, "c: low-pct@10"),
        ("low_percentile", {"q": 20}, "c: low-pct@20"),
        ("low_percentile", {"q": 25}, "c: low-pct@25"),
        ("trimmed_mean", {"k": 10}, "c: trim-top@10"),
        ("trimmed_mean", {"k": 20}, "c: trim-top@20"),
        ("trimmed_mean", {"k": 30}, "c: trim-top@30"),
    ]:
        methods.append((label, robust_score.robust_scores(grids, stat, **params)))

    lines = ["# Robust-gate comparison (precision/coverage, n=%d)\n" % n,
             "Metric: **prefix_correct** = panos confidently+correctly committed before the FIRST wrong",
             "room (= coverage-at-100%%-precision x n). recall@1 = correct at full coverage.\n",
             "| method | prefix_correct (cov@100%%prec) | recall@1 (all) |",
             "|--------|------------------------------|----------------|"]
    results = []
    for label, sm in methods:
        prefix, recall = curve(sm, true_room)
        results.append((label, prefix, recall))
        lines.append(f"| {label} | {prefix}/{n} ({100*prefix/n:.0f}%) | {recall}/{n} ({100*recall/n:.0f}%) |")
    base = results[0][1]
    best = max(results, key=lambda x: x[1])
    lines.append(f"\n**Baseline (a) prefix_correct = {base}/{n}. Best = {best[0]} at {best[1]}/{n}.**")
    lines.append(f"{'Robust BEATS the mean gate.' if best[1] > base and best[0] != results[0][0] else 'Robust does NOT beat the mean gate (n=12 is noisy, D26 caveat).'}")

    out = "\n".join(lines) + "\n"
    print(out)
    with open(os.path.join(_HERE, "ROBUST_RESULTS.md"), "w") as f:
        f.write(out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the analysis (produces the artifact)**

Run: `conda run -n panopin python -m experiments.fgpl_seed.robust_analysis`
Expected: prints the table and writes `ROBUST_RESULTS.md`; the final line states whether robust beats the mean gate.

- [ ] **Step 3: Sanity-check the baseline row against D26**

Confirm the `a: mean-loss` row's `recall@1 (all)` == 8/12 (matches the current cache's calibrated recall, D26). If not, the score matrix wiring is wrong — investigate before trusting the robust rows.

- [ ] **Step 4: Commit**

```bash
git add experiments/fgpl_seed/robust_analysis.py experiments/fgpl_seed/ROBUST_RESULTS.md
git commit -m "feat(robust): precision/coverage comparison of mean gate vs robust gate + results"
```

---

### Task 5: Record the outcome (docs)

**Files:**
- Modify: `docs/DECISIONS.md` (append D27), `docs/PROGRESS.md` (prepend current-state), `experiments/fgpl_seed/RESULTS.md` (link to ROBUST_RESULTS.md)

- [ ] **Step 1: Append D27 to `docs/DECISIONS.md`** — one paragraph: the precision/coverage comparison result (which stat, prefix_correct for (a) vs best (c)), whether robust beats the mean gate, and the honest n=12 caveat. Include the recommended next step (adopt the winner's gate as PanoPin's commit rule, or if no winner, that the gate — not the score — is where PanoPin stands and the path is (b) scaling).
- [ ] **Step 2: Prepend a `## Current state (2026-07-13 — robust-gate comparison)` block to `docs/PROGRESS.md`** — what was built, the headline number, next.
- [ ] **Step 3: Commit**

```bash
git add docs/DECISIONS.md docs/PROGRESS.md experiments/fgpl_seed/RESULTS.md
git commit -m "docs: record robust-gate comparison outcome (D27)"
```

---

## Self-Review

**1. Spec coverage:**
- §3 shared gate → Task 4 (`calibrate.minmax_scores` used for both). ✓
- §4 approach (a) mean-loss → Task 4 first method. ✓
- §5.1 residual capture + fidelity → Tasks 2 (function+fidelity) & 3 (72-pair capture). ✓
- §5.2 robust stats (median/low-pct/trimmed) → Task 1 + swept in Task 4. ✓
- §5.3 gate the robust scores → Task 4. ✓
- §6 precision/coverage deliverable + ROBUST_RESULTS.md → Task 4. ✓
- §7 FGPL deferred → not implemented (correct). ✓
- §9 unit + fidelity tests → Task 1 (unit incl. discrimination) + Task 2 (fidelity). ✓
- Note: spec §5.1 originally framed fidelity vs the *cached weighted* loss; the plan validates against `sampling_loss`'s own scalar (exact, since both are unweighted) — stricter and correct because `refine_pose_sampling_loss` is weighted. Spec to be annotated at handoff.

**2. Placeholder scan:** No TBD/TODO; all code blocks complete; commands have expected output. ✓

**3. Type consistency:** `robust_scores(grids, stat, **params)` signature identical in Tasks 1/4; grid = 101-float list everywhere; `residuals_at_pose(cfg, pano_path, cloud_path, t, R)->np.ndarray` consistent in Tasks 2/3; `cache[pano]["poses"][room]={"t","R"}` matches the D26 cache format; `calibrate.minmax_scores` return `{pano:{room:score}}` consumed correctly. ✓
