# PanoPin → FGPL seed ablation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace FGPL's slow SAM3+IoU jigsaw with PanoPin's color-based coarse seed and measure, via a 5-arm ablation on an S3DIS Area_3 same-shape subset, whether (and with how much guidance) the seed makes FGPL's final pose more accurate.

**Architecture:** A new PanoPin-side orchestration package `experiments/fgpl_seed/` builds the three FGPL per-scene artifacts (3D line map, 2D features, seed+config) for a 6-room subset **entirely in the raw S3DIS frame** (identity `rotation_matrix` ⇒ no floorplan/SAM3/density front-end), runs the config-driven FGPL estimator per arm via subprocess, and scores final poses with PanoPin's existing `eval/metrics.py`. FGPL's cost function is never modified; P2/P3 only *filter the candidate set* (translation grid / 24 rotations) behind default-off config flags.

**Tech Stack:** Python 3 (PanoPin: miniconda `python`; CPO: conda env `panopin-gpu`; FGPL tools+estimator: conda env `scan_env`); Open3D (PLY authoring + FGPL); the vendored CPO `localize_pair`; FGPL's `multiroom_pose_estimation.py` (config-driven) + its C++ `LineFromPointCloud` binary.

## Global Constraints

- **Two conda envs, never mixed:** CPO seed computation runs in **`panopin-gpu`** (RTX 4060; `torch 2.0.1+cu118`). All FGPL tools + the estimator run in **`scan_env`**. PanoPin harness/metrics run in base `python`. (PanoPin's "do not use scan_env" rule applies to *PanoPin's own CPO* code — the FGPL tools are FGPL's and correctly use `scan_env`.)
- **Fairness (D5):** GT may be used ONLY to build reference seeds (Oracle, Wrong-room) and to score. It lives in `experiments/`, never in `src/panopin`. The method-under-test arms (P1/P2/P3) use only CPO output.
- **Do not edit `third_party/cpo/` (D9).** Do not edit `eval/` metrics or S3DIS GT (protocol #5).
- **FGPL edits are minimal + flag-gated + default-off** (preserve current behavior when flags absent) and saved as a patch under `experiments/fgpl_seed/patches/`.
- **Raw S3DIS frame throughout:** the FGPL line map is built from the raw S3DIS clouds; seeds are raw-3D XY; `metadata.json` uses `rotation_matrix = I3` so `aligned_meters_to_raw_3d` is a pass-through (`[x,y] → [x,y,0]`). Frame identity (cloud ↔ pano/pose, ~5 mm) per Point_360.
- **Subset (first cut):** rooms `office_1, office_4, office_5, office_6, office_7, hallway_3`; ≤2 panos/room; in-frame panos only.
- **Determinism:** call `panopin.determinism.pin()` before CPO; treat n≈10 as directional, not significant.
- **FGPL repo root:** `/home/ruoyu/scan2measure-webframework` (constant `FGPL_ROOT`).
- **Design reference:** `docs/specs/2026-07-10-panopin-fgpl-seed-ablation-design.md`.

---

## File structure (all under `experiments/fgpl_seed/` in the PanoPin repo unless noted)

- `paths.py` — all path/env constants (FGPL_ROOT, envs, WORK dir, scene name, FGPL script paths). One responsibility: locations.
- `subset.py` — defines the room set + resolves panos (uuid, jpg, cloud) from `eval/s3dis_gt.py` + `config/datasets.json`.
- `build_ply.py` — concatenate subset room `.txt` clouds → one combined `.ply` for the baker.
- `fgpl_tool.py` — subprocess helper: write a config JSON, run `conda run -n scan_env python <FGPL script> --config <json>`.
- `build_linemap.py` — orchestrate baker → cluster → `3d_line_map.pkl`.
- `build_features.py` — stage each pano jpg + run the 2D feature extractor → `fgpl_features.json`.
- `seed_and_config.py` — identity `metadata.json`; per-arm `demo6_alignment.json` seed writer; per-arm estimator config writer; room centroids/siblings.
- `cpo_seeds.py` — run CPO `localize_pair` (panopin-gpu) per pano × 6 rooms; assign min-loss room; cache `(t,R,loss)`.
- `run_arm.py` — run the estimator for one arm; collect per-pano `camera_pose.json`.
- `score.py` — score an arm's poses vs GT (translation/rotation error + wrong-room rate) via `eval/metrics.py`.
- `run_all.py` — top-level orchestrator: Phase 0 build → gate → Phase 1 arms → report.
- `patches/fgpl_narrowing.patch` — the saved P2/P3 FGPL edit.
- `tests/` — `test_subset.py`, `test_seed_and_config.py`, `test_score.py`, `test_narrowing.py`.
- `work/` (gitignored) — all generated artifacts: `clouds/ linemap/ features/ panos/ seeds/ configs/ poses/ results/`.

FGPL repo edits (Task 10 only): `src/pose_estimation/pose_search.py`, `src/pose_estimation/multiroom_pose_estimation.py` (flag-gated).

---

## Task 1: Scaffold — paths, subset resolution, env preflight

**Files:**
- Create: `experiments/fgpl_seed/__init__.py` (empty), `experiments/fgpl_seed/paths.py`, `experiments/fgpl_seed/subset.py`
- Create: `experiments/fgpl_seed/tests/test_subset.py`
- Modify: `.gitignore` (add `experiments/fgpl_seed/work/`)

**Interfaces:**
- Produces: `paths.FGPL_ROOT, paths.SCAN_ENV, paths.GPU_ENV, paths.WORK, paths.SCENE, paths.BAKER, paths.CLUSTER, paths.FEATURES, paths.ESTIMATOR` (all `pathlib.Path`/`str`); `paths.subdir(name)->Path`.
- Produces: `subset.ROOMS: list[str]`, `subset.MAX_PANOS_PER_ROOM: int`, `subset.build_subset() -> list[dict]` where each dict = `{"pano_name": str, "uuid": str, "room": str, "pano_jpg": str, "cloud_txt": str}` (`pano_name == uuid`).

- [ ] **Step 1: Write `paths.py`**

```python
"""Central path + env constants for the FGPL-seed ablation. No logic."""
from pathlib import Path

FGPL_ROOT = Path("/home/ruoyu/scan2measure-webframework")
SCAN_ENV = "scan_env"        # FGPL tools + estimator (verify in Step 4; single knob if wrong)
GPU_ENV = "panopin-gpu"      # CPO localize_pair
SCENE = "area3_seed_ablation"

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"

BAKER = FGPL_ROOT / "src/geometry_3d/point_cloud_geometry_baker_V4.py"
CLUSTER = FGPL_ROOT / "src/geometry_3d/cluster_3d_lines.py"
FEATURES = FGPL_ROOT / "src/features_2d/image_feature_extractionV2.py"
ESTIMATOR = FGPL_ROOT / "src/pose_estimation/multiroom_pose_estimation.py"
LINE_BINARY = FGPL_ROOT / "3DLineDetection/build/src/LineFromPointCloud"


def subdir(name: str) -> Path:
    d = WORK / name
    d.mkdir(parents=True, exist_ok=True)
    return d
```

- [ ] **Step 2: Write the failing test** `experiments/fgpl_seed/tests/test_subset.py`

```python
import os, pytest
from experiments.fgpl_seed import subset

def test_rooms_are_the_six_same_shape_subset():
    assert subset.ROOMS == ["office_1", "office_4", "office_5", "office_6", "office_7", "hallway_3"]

@pytest.mark.skipif(not os.path.exists("/mnt/d"), reason="needs S3DIS GT on /mnt/d")
def test_build_subset_resolves_clouds_and_panos():
    rows = subset.build_subset()
    rooms = {r["room"] for r in rows}
    assert rooms == set(subset.ROOMS)                       # every room contributes >=1 pano
    for r in rows:
        assert os.path.exists(r["cloud_txt"]), r["cloud_txt"]
        assert os.path.exists(r["pano_jpg"]), r["pano_jpg"]
        assert r["pano_name"] == r["uuid"]
    per_room = {}
    for r in rows:
        per_room[r["room"]] = per_room.get(r["room"], 0) + 1
    assert all(v <= subset.MAX_PANOS_PER_ROOM for v in per_room.values())
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest experiments/fgpl_seed/tests/test_subset.py -q`
Expected: FAIL (`ModuleNotFoundError: experiments.fgpl_seed.subset`).

- [ ] **Step 4: Env preflight (manual, record output)**

Run:
```bash
conda run -n scan_env python -c "import open3d, torch, numpy, cv2; print('scan_env OK', torch.__version__)"
ls -l /home/ruoyu/scan2measure-webframework/3DLineDetection/build/src/LineFromPointCloud
conda run -n panopin-gpu python -c "import torch, torch_scatter; print('gpu env OK', torch.cuda.is_available())"
```
Expected: both envs import cleanly; the `LineFromPointCloud` binary exists and is executable. If `scan_env` is not the correct env name, update `paths.SCAN_ENV` (single constant) and re-run.

- [ ] **Step 5: Write `subset.py`**

```python
"""Resolve the 6-room same-shape subset to (pano_name, uuid, room, jpg, cloud) rows.
Uses the PanoPin harness GT loader; pano_name is the uuid (clean, collision-free)."""
import os, sys
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from eval import s3dis_gt

ROOMS = ["office_1", "office_4", "office_5", "office_6", "office_7", "hallway_3"]
MAX_PANOS_PER_ROOM = 2


def _resolve_pano(uuid, pano_rgb_dir):
    for p in sorted(os.listdir(pano_rgb_dir)):
        parts = p.split("_")
        if len(parts) >= 2 and parts[1] == uuid:
            return os.path.join(pano_rgb_dir, p)
    return None


def build_subset():
    cfg = s3dis_gt.load_config(None)
    a = cfg["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", cfg)
    by_room = {}
    for u, v in gt.items():
        by_room.setdefault(v["room"], []).append(u)
    rows = []
    for room in ROOMS:
        cloud = os.path.join(a["rooms_dir"], room, room + ".txt")
        for u in sorted(by_room.get(room, []))[:MAX_PANOS_PER_ROOM]:
            jpg = _resolve_pano(u, a["pano_rgb_dir"])
            if jpg:
                rows.append({"pano_name": u, "uuid": u, "room": room,
                             "pano_jpg": jpg, "cloud_txt": cloud})
    return rows
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest experiments/fgpl_seed/tests/test_subset.py -q`
Expected: PASS (both tests; the `/mnt/d` one runs since GT is present). Record how many panos resolved per room.

- [ ] **Step 7: Commit**

```bash
git add experiments/fgpl_seed/__init__.py experiments/fgpl_seed/paths.py experiments/fgpl_seed/subset.py experiments/fgpl_seed/tests/test_subset.py .gitignore
git commit -m "feat(fgpl-seed): scaffold + 6-room subset resolution"
```

---

## Task 2: Combined PLY builder

**Files:**
- Create: `experiments/fgpl_seed/build_ply.py`

**Interfaces:**
- Consumes: `subset.build_subset()` rows; `paths.subdir`, `paths.SCENE`.
- Produces: `build_ply.build_combined_ply(rows) -> Path` (writes `work/clouds/{SCENE}.ply`, returns its path). PLY has XYZ + RGB(0-255→0-1) from the unique room clouds.

- [ ] **Step 1: Write `build_ply.py`**

```python
"""Concatenate the subset's unique room .txt clouds (X Y Z R G B) into one .ply
for the FGPL baker (Open3D-readable). Color is kept but the baker ignores it."""
import numpy as np, open3d as o3d
from experiments.fgpl_seed import paths, subset


def build_combined_ply(rows):
    clouds = sorted({r["cloud_txt"] for r in rows})
    xyz_all, rgb_all = [], []
    for c in clouds:
        arr = np.loadtxt(c)                      # (N,6): X Y Z R G B
        xyz_all.append(arr[:, :3])
        rgb_all.append(arr[:, 3:6])
    xyz = np.concatenate(xyz_all, axis=0)
    rgb = np.concatenate(rgb_all, axis=0)
    if rgb.max() > 1.5:                          # S3DIS Original RGB may be 0-255
        rgb = rgb / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1))
    out = paths.subdir("clouds") / f"{paths.SCENE}.ply"
    o3d.io.write_point_cloud(str(out), pcd)
    return out


if __name__ == "__main__":
    rows = subset.build_subset()
    p = build_combined_ply(rows)
    import numpy as np
    print("wrote", p, "points=", np.asarray(o3d.io.read_point_cloud(str(p)).points).shape[0])
```

- [ ] **Step 2: Run it (this is the check — artifact must exist and be non-empty)**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m experiments.fgpl_seed.build_ply`
Expected: prints `wrote .../work/clouds/area3_seed_ablation.ply points= <N>` with N in the millions (6 rooms). If `open3d` is missing in `panopin`, run under `panopin-gpu` (which has it) — either is fine for authoring.

- [ ] **Step 3: Commit**

```bash
git add experiments/fgpl_seed/build_ply.py
git commit -m "feat(fgpl-seed): combine subset room clouds into one PLY"
```

---

## Task 3: 3D line map (baker → cluster)

**Files:**
- Create: `experiments/fgpl_seed/fgpl_tool.py`
- Create: `experiments/fgpl_seed/build_linemap.py`

**Interfaces:**
- Produces: `fgpl_tool.run_tool(script_path: Path, config: dict, tag: str) -> None` (writes `work/configs/{tag}.json`, runs it in `scan_env`, raises on non-zero exit).
- Produces: `build_linemap.build_linemap(ply_path: Path) -> Path` (returns `work/linemap/3d_line_map.pkl`).

- [ ] **Step 1: Write `fgpl_tool.py`**

```python
"""Run an FGPL --config script inside scan_env via subprocess."""
import json, subprocess
from experiments.fgpl_seed import paths


def run_tool(script_path, config, tag):
    cfg_path = paths.subdir("configs") / f"{tag}.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    cmd = ["conda", "run", "--no-capture-output", "-n", paths.SCAN_ENV,
           "python", str(script_path), "--config", str(cfg_path)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(paths.FGPL_ROOT))
```

- [ ] **Step 2: Write `build_linemap.py`**

```python
"""Baker (.ply -> room_geometry.pkl) then cluster (-> 3d_line_map.pkl)."""
from experiments.fgpl_seed import paths, fgpl_tool, build_ply, subset

def build_linemap(ply_path):
    out = paths.subdir("linemap")
    fgpl_tool.run_tool(paths.BAKER, {
        "point_cloud_name": paths.SCENE,
        "point_cloud_path": str(ply_path),
        "output_dir": str(out),
    }, tag="baker")
    room_geom = out / "room_geometry.pkl"
    assert room_geom.exists(), room_geom
    fgpl_tool.run_tool(paths.CLUSTER, {
        "point_cloud_name": paths.SCENE,
        "input_pkl": str(room_geom),
        "output_dir": str(out),
    }, tag="cluster")
    line_map = out / "3d_line_map.pkl"
    assert line_map.exists(), line_map
    return line_map

if __name__ == "__main__":
    rows = subset.build_subset()
    ply = build_ply.build_combined_ply(rows)
    lm = build_linemap(ply)
    import pickle
    with open(lm, "rb") as f:
        d = pickle.load(f)
    print("3d_line_map keys:", sorted(d.keys()))
    print("n dense lines:", len(d["dense_starts"]), " n intersections:", len(d["inter_3d"]))
```

- [ ] **Step 3: Run it (artifact check)**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m experiments.fgpl_seed.build_linemap`
Expected: baker prints its subprocess + C++ line-detection output; cluster runs; final print shows `3d_line_map keys:` containing `dense_starts, dense_ends, principal_3d, inter_3d, inter_3d_idx, inter_3d_mask` and non-zero line/intersection counts.
**If the C++ binary errors** (`LD_LIBRARY_PATH`/build missing): build it per `scan2measure-webframework/scripts/setup-native.sh` and re-run. **If line/intersection counts are ~0** (noisy S3DIS cloud): this is the Phase-0 risk — subsample the combined PLY (voxel_down_sample in `build_ply`) and retry before proceeding.

- [ ] **Step 4: Commit**

```bash
git add experiments/fgpl_seed/fgpl_tool.py experiments/fgpl_seed/build_linemap.py
git commit -m "feat(fgpl-seed): build 3D line map from subset cloud (baker+cluster)"
```

---

## Task 4: 2D features per pano

**Files:**
- Create: `experiments/fgpl_seed/build_features.py`

**Interfaces:**
- Produces: `build_features.stage_and_build(rows) -> dict` mapping `pano_name -> fgpl_features.json path`; also stages each jpg to `work/panos/{pano_name}.jpg`. Feature dir layout is `work/features/{pano_name}_v2/fgpl_features.json` (the estimator appends `/{pano}_v2/fgpl_features.json` to `features_2d_dir`).

- [ ] **Step 1: Write `build_features.py`**

```python
"""Stage each pano jpg to work/panos/{pano_name}.jpg and run the FGPL 2D feature
extractor into work/features/{pano_name}_v2/fgpl_features.json."""
import shutil, json
from experiments.fgpl_seed import paths, fgpl_tool, subset

def stage_and_build(rows):
    pano_dir = paths.subdir("panos")
    feat_base = paths.subdir("features")
    out = {}
    for r in rows:
        name = r["pano_name"]
        staged = pano_dir / f"{name}.jpg"
        if not staged.exists():
            shutil.copy(r["pano_jpg"], staged)
        outdir = feat_base / f"{name}_v2"
        fgpl_tool.run_tool(paths.FEATURES, {
            "room_name": name,
            "pano_path": str(staged),
            "output_dir": str(outdir),
        }, tag=f"feat_{name}")
        fj = outdir / "fgpl_features.json"
        assert fj.exists(), fj
        with open(fj) as f:
            d = json.load(f)
        assert d.get("n_lines", 0) > 0, f"no lines for {name}"
        out[name] = str(fj)
    return out

if __name__ == "__main__":
    rows = subset.build_subset()
    m = stage_and_build(rows)
    print(f"built features for {len(m)} panos")
```

- [ ] **Step 2: Run it (artifact check)**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m experiments.fgpl_seed.build_features`
Expected: one feature run per pano; prints `built features for <N> panos`; each `work/features/{pano}_v2/fgpl_features.json` exists with `n_lines > 0`.

- [ ] **Step 3: Commit**

```bash
git add experiments/fgpl_seed/build_features.py
git commit -m "feat(fgpl-seed): extract FGPL 2D features per subset pano"
```

---

## Task 5: Identity metadata + seed/config writer

**Files:**
- Create: `experiments/fgpl_seed/seed_and_config.py`
- Create: `experiments/fgpl_seed/tests/test_seed_and_config.py`

**Interfaces:**
- Consumes: `subset` rows; `eval/s3dis_gt` GT; a CPO cache dict (Task 9) `{pano_name: {"t":[x,y,z], "R":[[..]*3], "loss":float, "room":str}}`.
- Produces:
  - `seed_and_config.write_identity_metadata() -> Path` (`work/configs/metadata.json`, `rotation_matrix=I3`).
  - `seed_and_config.room_centroids(rows) -> dict[str, [x,y,z]]` (mean XYZ per room cloud).
  - `seed_and_config.SIBLING: dict[str,str]` (wrong-room map).
  - `seed_and_config.write_seed(arm, rows, gt, cpo_cache) -> Path` (`work/seeds/{arm}.json`, demo6 format, raw-3D `camera_position=[x,y]`).
  - `seed_and_config.write_config(arm, rows, seed_path, line_map, metadata_path, feat_dir, pano_dir, narrowing) -> Path` (`work/configs/pose_{arm}.json`).

- [ ] **Step 1: Write the failing test** `tests/test_seed_and_config.py`

```python
import json
from experiments.fgpl_seed import seed_and_config as sc

ROWS = [
    {"pano_name": "u1", "uuid": "u1", "room": "office_4", "cloud_txt": "/x", "pano_jpg": "/x.jpg"},
    {"pano_name": "u2", "uuid": "u2", "room": "office_6", "cloud_txt": "/y", "pano_jpg": "/y.jpg"},
]
GT = {"u1": {"location": [1.0, 2.0, 1.4], "R_cw": [[1,0,0],[0,1,0],[0,0,1]], "room": "office_4"},
      "u2": {"location": [8.0, 9.0, 1.4], "R_cw": [[1,0,0],[0,1,0],[0,0,1]], "room": "office_6"}}
CPO = {"u1": {"t": [1.1, 2.1, 1.3], "R": [[1,0,0],[0,1,0],[0,0,1]], "loss": 0.1, "room": "office_4"},
       "u2": {"t": [7.9, 9.2, 1.5], "R": [[1,0,0],[0,1,0],[0,0,1]], "loss": 0.1, "room": "office_6"}}

def test_oracle_seed_uses_gt_xy(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.paths, "WORK", tmp_path)
    p = sc.write_seed("oracle", ROWS, GT, CPO)
    m = {x["pano_name"]: x["camera_position"] for x in json.load(open(p))["matches"]}
    assert m["u1"] == [1.0, 2.0]                      # GT xy, z dropped
    assert m["u2"] == [8.0, 9.0]

def test_p1_seed_uses_cpo_xy(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.paths, "WORK", tmp_path)
    p = sc.write_seed("p1", ROWS, GT, CPO)
    m = {x["pano_name"]: x["camera_position"] for x in json.load(open(p))["matches"]}
    assert m["u1"] == [1.1, 2.1]                      # CPO t xy

def test_wrong_room_seed_uses_sibling_centroid(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.paths, "WORK", tmp_path)
    # sibling of office_4 is office_6 -> u1 should be seeded at u2's room area, not its own
    cents = {"office_4": [1.0, 2.0, 1.4], "office_6": [8.0, 9.0, 1.4]}
    p = sc.write_seed("wrong_room", ROWS, GT, CPO, centroids=cents)
    m = {x["pano_name"]: x["camera_position"] for x in json.load(open(p))["matches"]}
    assert m["u1"] == [8.0, 9.0]                      # office_6 centroid (deliberately wrong)

def test_identity_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.paths, "WORK", tmp_path)
    p = sc.write_identity_metadata()
    md = json.load(open(p))
    assert md["rotation_matrix"] == [[1,0,0],[0,1,0],[0,0,1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest experiments/fgpl_seed/tests/test_seed_and_config.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `seed_and_config.py`**

```python
"""Identity metadata + per-arm demo6 seed + per-arm estimator config.
All positions are RAW S3DIS 3D; only XY is emitted (estimator maps [x,y]->[x,y,0])."""
import json, numpy as np
from experiments.fgpl_seed import paths

I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Deliberately-wrong same-shape target for the Wrong-room floor arm.
SIBLING = {"office_4": "office_6", "office_6": "office_4",
           "office_5": "office_7", "office_7": "office_5",
           "office_1": "office_5", "hallway_3": "office_1"}


def _seeds_dir():
    d = paths.WORK / "seeds"; d.mkdir(parents=True, exist_ok=True); return d

def _configs_dir():
    d = paths.WORK / "configs"; d.mkdir(parents=True, exist_ok=True); return d


def write_identity_metadata():
    md = {"min_coords": [0.0, 0.0, 0.0], "max_dim": 1.0, "offset": [0.0, 0.0],
          "image_width": 256, "image_height": 256, "translation": [0.0, 0.0, 0.0],
          "rotation_matrix": I3}
    p = _configs_dir() / "metadata.json"
    with open(p, "w") as f:
        json.dump(md, f, indent=2)
    return p


def room_centroids(rows):
    cents = {}
    for c in sorted({r["cloud_txt"] for r in rows}):
        arr = np.loadtxt(c)
        room = [r["room"] for r in rows if r["cloud_txt"] == c][0]
        cents[room] = arr[:, :3].mean(axis=0).tolist()
    return cents


def _pos_for(arm, r, gt, cpo, centroids):
    name, room = r["pano_name"], r["room"]
    if arm == "oracle":
        return gt[name]["location"][:2]
    if arm == "wrong_room":
        sib = SIBLING[room]
        return centroids[sib][:2]
    return cpo[name]["t"][:2]                       # p1/p2/p3 all use CPO position


def write_seed(arm, rows, gt, cpo, centroids=None):
    if arm == "wrong_room" and centroids is None:
        centroids = room_centroids(rows)
    matches = []
    for r in rows:
        pos = _pos_for(arm, r, gt, cpo, centroids)
        matches.append({"pano_name": r["pano_name"], "room_label": r["room"],
                        "room_idx": 0, "score": 1.0, "rotation_deg": 0.0,
                        "camera_position": [float(pos[0]), float(pos[1])]})
    doc = {"metadata": {"pipeline": f"panopin-seed:{arm}"}, "matches": matches}
    p = _seeds_dir() / f"{arm}.json"
    with open(p, "w") as f:
        json.dump(doc, f, indent=2)
    return p


def write_config(arm, rows, seed_path, line_map, metadata_path, feat_dir, pano_dir, narrowing=None):
    cfg = {
        "point_cloud_name": paths.SCENE,
        "pano_names": [r["pano_name"] for r in rows],
        "use_local_filtering": True,
        "pkl_3d_path": str(line_map),
        "alignment_path": str(seed_path),
        "metadata_path": str(metadata_path),
        "point_cloud_path": str(paths.WORK / "clouds" / f"{paths.SCENE}.ply"),
        "features_2d_dir": str(feat_dir),
        "pano_dir": str(pano_dir),
        "output_dir": str(paths.WORK / "poses" / arm),
    }
    if narrowing:
        cfg.update(narrowing)                        # seed_trans_radius / seed_yaw_deg / seed_yaw_tol
    p = _configs_dir() / f"pose_{arm}.json"
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    return p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest experiments/fgpl_seed/tests/test_seed_and_config.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/fgpl_seed/seed_and_config.py experiments/fgpl_seed/tests/test_seed_and_config.py
git commit -m "feat(fgpl-seed): identity metadata + per-arm seed/config writer"
```

---

## Task 6: Arm runner (estimator wrapper)

**Files:**
- Create: `experiments/fgpl_seed/run_arm.py`

**Interfaces:**
- Consumes: a pose config path (Task 5); `paths`, `subset`.
- Produces: `run_arm.run_arm(config_path, rows) -> dict` = `{pano_name: {"translation":[x,y,z], "rotation":[[..]*3]}}` read from `work/poses/{arm}/{pano}/camera_pose.json`. Missing/failed panos map to `None`.

- [ ] **Step 1: Write `run_arm.py`**

```python
"""Run the FGPL estimator for one arm config; collect per-pano camera_pose.json.
camera_pose.json is written (line 524) BEFORE the composite viz (line 581), so a
viz-stage crash still leaves per-pano poses readable — we tolerate a non-zero exit
and read whatever poses were produced."""
import json, subprocess, os
from experiments.fgpl_seed import paths

def run_arm(config_path, rows):
    with open(config_path) as f:
        cfg = json.load(f)
    out_base = cfg["output_dir"]
    env = dict(os.environ)
    # Ada/cu-mismatch guard (D20: scan_env torch segfaulted on the 4060). If the
    # estimator crashes on GPU, uncomment to force CPU:  env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = ["conda", "run", "--no-capture-output", "-n", paths.SCAN_ENV,
           "python", str(paths.ESTIMATOR), "--config", str(config_path)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(paths.FGPL_ROOT), env=env, check=False)  # tolerate viz crash
    poses = {}
    for r in rows:
        cp = os.path.join(out_base, r["pano_name"], "camera_pose.json")
        if os.path.exists(cp):
            with open(cp) as f:
                d = json.load(f)
            poses[r["pano_name"]] = {"translation": d["translation"], "rotation": d["rotation"]}
        else:
            poses[r["pano_name"]] = None
    return poses
```

- [ ] **Step 2: Smoke-run on the oracle arm for ONE pano (check a pose file appears)**

Run (assumes Tasks 2-5 artifacts exist; builds a 1-pano oracle config inline):
```bash
cd /home/ruoyu/PanoPin && conda run -n panopin python -c "
from experiments.fgpl_seed import subset, seed_and_config as sc, run_arm, paths
from eval import s3dis_gt
rows = subset.build_subset()[:1]
gt = s3dis_gt.load_gt('Area_3', s3dis_gt.load_config(None))
md = sc.write_identity_metadata()
seed = sc.write_seed('oracle', rows, gt, {})
cfg = sc.write_config('oracle', rows, seed, paths.WORK/'linemap'/'3d_line_map.pkl', md,
                      paths.WORK/'features', paths.WORK/'panos')
poses = run_arm.run_arm(cfg, rows)
print('pose for', rows[0]['pano_name'], '=', poses[rows[0]['pano_name']])
"
```
Expected: prints a non-`None` pose dict with `translation` (3 numbers) + `rotation` (3×3). If `None`, inspect the estimator stderr above (missing artifact / frame KeyError / env).

- [ ] **Step 3: Commit**

```bash
git add experiments/fgpl_seed/run_arm.py
git commit -m "feat(fgpl-seed): estimator arm runner (viz-crash tolerant)"
```

---

## Task 7: Scorer

**Files:**
- Create: `experiments/fgpl_seed/score.py`
- Create: `experiments/fgpl_seed/tests/test_score.py`

**Interfaces:**
- Consumes: arm `poses` (Task 6), `gt` (`eval/s3dis_gt`), `rows`, `room_centroids`.
- Produces: `score.score_arm(poses, gt, rows, centroids) -> dict` with keys `translation` (from `metrics.translation_errors`), `rotation` (from `metrics.rotation_errors`), `wrong_room_rate`, `n_localized`. Wrong-room = final translation's nearest room centroid ≠ true room.

- [ ] **Step 1: Write the failing test** `tests/test_score.py`

```python
from experiments.fgpl_seed import score

ROWS = [{"pano_name": "u1", "room": "office_4"}, {"pano_name": "u2", "room": "office_6"}]
GT = {"u1": {"location": [1, 2, 1.4], "R_cw": [[1,0,0],[0,1,0],[0,0,1]]},
      "u2": {"location": [8, 9, 1.4], "R_cw": [[1,0,0],[0,1,0],[0,0,1]]}}
CENTS = {"office_4": [1, 2, 1.4], "office_6": [8, 9, 1.4]}

def test_perfect_poses_zero_error_and_no_wrong_room():
    poses = {"u1": {"translation": [1, 2, 1.4], "rotation": [[1,0,0],[0,1,0],[0,0,1]]},
             "u2": {"translation": [8, 9, 1.4], "rotation": [[1,0,0],[0,1,0],[0,0,1]]}}
    s = score.score_arm(poses, GT, ROWS, CENTS)
    assert s["translation"]["max"] < 1e-6
    assert s["wrong_room_rate"] == 0.0
    assert s["n_localized"] == 2

def test_swapped_pose_counts_as_wrong_room():
    poses = {"u1": {"translation": [8, 9, 1.4], "rotation": [[1,0,0],[0,1,0],[0,0,1]]},  # in office_6
             "u2": {"translation": [8, 9, 1.4], "rotation": [[1,0,0],[0,1,0],[0,0,1]]}}
    s = score.score_arm(poses, GT, ROWS, CENTS)
    assert s["wrong_room_rate"] == 0.5           # u1 landed in the wrong room

def test_none_pose_excluded_from_localized():
    poses = {"u1": None, "u2": {"translation": [8, 9, 1.4], "rotation": [[1,0,0],[0,1,0],[0,0,1]]}}
    s = score.score_arm(poses, GT, ROWS, CENTS)
    assert s["n_localized"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest experiments/fgpl_seed/tests/test_score.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `score.py`**

```python
"""Score one arm's FGPL poses vs S3DIS GT using the PanoPin harness metrics."""
import math
from eval import metrics

def _nearest_room(xyz, centroids):
    best, bd = None, float("inf")
    for room, c in centroids.items():
        d = sum((xyz[i] - c[i]) ** 2 for i in range(3))
        if d < bd:
            bd, best = d, room
    return best

def score_arm(poses, gt, rows, centroids):
    true_room = {r["pano_name"]: r["room"] for r in rows}
    pred_t = {u: p["translation"] for u, p in poses.items() if p is not None}
    pred_R = {u: p["rotation"] for u, p in poses.items() if p is not None}
    gt_loc = {u: gt[u]["location"] for u in gt}
    gt_R = {u: gt[u]["R_cw"] for u in gt}
    wrong = 0
    for u, p in poses.items():
        if p is None:
            continue
        if _nearest_room(p["translation"], centroids) != true_room[u]:
            wrong += 1
    n_loc = len(pred_t)
    return {
        "n_localized": n_loc,
        "translation": metrics.translation_errors(pred_t, gt_loc),
        "rotation": metrics.rotation_errors(pred_R, gt_R),
        "wrong_room_rate": (wrong / n_loc) if n_loc else None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest experiments/fgpl_seed/tests/test_score.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/fgpl_seed/score.py experiments/fgpl_seed/tests/test_score.py
git commit -m "feat(fgpl-seed): arm scorer (trans/rot error + wrong-room rate)"
```

---

## Task 8: **PHASE-0 GATE** — Oracle arm end-to-end

**Files:**
- Create: `experiments/fgpl_seed/gate_oracle.py`

**Interfaces:**
- Consumes: everything from Tasks 2-7. Produces: `work/results/gate_oracle.json` + a printed verdict.

**This task is the go/no-go milestone. Do NOT proceed to Task 9+ unless it passes.**

- [ ] **Step 1: Write `gate_oracle.py`**

```python
"""Phase-0 gate: build artifacts (if absent), run the ORACLE arm (GT-position seed),
score it. Requires median translation error < 0.5 m — else FGPL isn't localizing on
S3DIS clouds and we stop to diagnose (frame/convention/line-map quality) BEFORE the
ablation. Uses GT only for the reference seed (allowed; experiments-only)."""
import json
from experiments.fgpl_seed import (subset, build_ply, build_linemap, build_features,
                                    seed_and_config as sc, run_arm, score, paths)
from eval import s3dis_gt

GATE_MEDIAN_M = 0.5

def main():
    rows = subset.build_subset()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    ply = build_ply.build_combined_ply(rows)
    line_map = build_linemap.build_linemap(ply)
    build_features.stage_and_build(rows)
    md = sc.write_identity_metadata()
    seed = sc.write_seed("oracle", rows, gt, {})
    cfg = sc.write_config("oracle", rows, seed, line_map, md,
                          paths.WORK / "features", paths.WORK / "panos")
    poses = run_arm.run_arm(cfg, rows)
    cents = sc.room_centroids(rows)
    s = score.score_arm(poses, gt, rows, cents)
    out = paths.subdir("results") / "gate_oracle.json"
    with open(out, "w") as f:
        json.dump(s, f, indent=2)
    med = s["translation"]["median"]
    print(json.dumps(s, indent=2))
    print(f"\nGATE: median translation = {med} m (threshold {GATE_MEDIAN_M} m), "
          f"localized {s['n_localized']}/{len(rows)}, wrong-room {s['wrong_room_rate']}")
    ok = med is not None and med < GATE_MEDIAN_M
    print("GATE PASSED" if ok else "GATE FAILED — STOP and diagnose before Phase 1")
    return ok

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
```

- [ ] **Step 2: Run the gate**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m experiments.fgpl_seed.gate_oracle`
Expected: prints the score JSON and `GATE PASSED` with median translation < 0.5 m. **Record the full output in `docs/PROGRESS.md`.**

- [ ] **Step 3: If the gate FAILS — diagnose (do not patch blindly)**

Check, in order: (a) frame/convention — is FGPL `translation` camera-world position in the S3DIS frame? Compare one pose to its GT `location` component-wise; a consistent axis flip/offset ⇒ convention mismatch, fix the comparison (or a sign) in `score`/seed, not the metric. (b) line-map quality — were dense-line/intersection counts non-trivial (Task 3)? (c) seed reached the estimator — is `camera_position` non-degenerate in `work/seeds/oracle.json`? Only continue once oracle localizes.

- [ ] **Step 4: Commit**

```bash
git add experiments/fgpl_seed/gate_oracle.py
git commit -m "feat(fgpl-seed): Phase-0 oracle gate (FGPL-on-S3DIS go/no-go)"
```

---

## Task 9: CPO seeds (compute once, reuse for P1/P2/P3)

**Files:**
- Create: `experiments/fgpl_seed/cpo_seeds.py`

**Interfaces:**
- Consumes: `subset` rows; CPO `localize_pair` (panopin-gpu), `panopin.cpo_config.load_cfg`, `panopin.determinism.pin`.
- Produces: `work/seeds/cpo_cache.json` = `{pano_name: {"t":[x,y,z], "R":[[..]*3], "loss":float, "room":str, "per_room":{room:loss}}}`. `room` = min-loss room (the assignment); `t,R` are that room's CPO pose. **Runs in `panopin-gpu`.**

- [ ] **Step 1: Write `cpo_seeds.py`**

```python
"""Per pano: CPO localize_pair vs each of the 6 candidate rooms -> assign min-loss
room; keep that room's (t,R). This is the seed for P1/P2/P3. Run in panopin-gpu."""
import os, sys, json
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from experiments.fgpl_seed import subset, paths
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import localize_pair

def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    clouds = {r["room"]: r["cloud_txt"] for r in rows}   # room -> its cloud .txt
    cache = {}
    for r in rows:
        per_room, best = {}, None
        for room, cloud in clouds.items():
            t, R, loss = localize_pair(cfg, r["pano_jpg"], cloud)
            per_room[room] = loss
            if best is None or loss < best[2]:
                best = (t, R, loss, room)
        t, R, loss, room = best
        cache[r["pano_name"]] = {"t": [float(x) for x in t],
                                 "R": [[float(x) for x in row] for row in R],
                                 "loss": float(loss), "room": room, "per_room": per_room}
        print(f"{r['pano_name']} true={r['room']} -> assigned={room} loss={loss:.3f}")
    out = paths.subdir("seeds") / "cpo_cache.json"
    with open(out, "w") as f:
        json.dump(cache, f, indent=2)
    print("wrote", out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (expensive — ~6 rooms × ~10 panos × ~25 s ≈ 25 min on GPU)**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin-gpu python -m experiments.fgpl_seed.cpo_seeds`
Expected: one line per pano showing true vs assigned room + loss; writes `cpo_cache.json`. Sanity: assigned==true for a majority (this is PanoPin's ~58% room accuracy regime; same-shape confusions expected).

- [ ] **Step 3: Verify cache structure**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -c "import json,glob; d=json.load(open('experiments/fgpl_seed/work/seeds/cpo_cache.json')); k=next(iter(d)); assert set(d[k])>={'t','R','loss','room','per_room'}; print('cache ok', len(d), 'panos')"`
Expected: `cache ok <N> panos`.

- [ ] **Step 4: Commit**

```bash
git add experiments/fgpl_seed/cpo_seeds.py
git commit -m "feat(fgpl-seed): CPO seed computation + room assignment cache"
```

---

## Task 10: P2/P3 narrowing (flag-gated FGPL edit)

**Files:**
- Modify: `scan2measure-webframework/src/pose_estimation/pose_search.py` (`generate_translation_grid` ~L117; `build_rotation_candidates` ~L72)
- Modify: `scan2measure-webframework/src/pose_estimation/multiroom_pose_estimation.py` (local-mode grid call ~L329; rotation call ~L386; read new cfg keys ~L163)
- Create: `experiments/fgpl_seed/patches/fgpl_narrowing.patch` (saved `git diff` of the two FGPL files)
- Create: `experiments/fgpl_seed/tests/test_narrowing.py`

**Interfaces:**
- New config keys (all optional, default absent ⇒ current behavior): `seed_trans_radius` (float, meters), `seed_yaw_deg` (float), `seed_yaw_tol` (float, default 30).
- `generate_translation_grid(..., center=None, radius=None)` — when both set, keep only grid points within `radius` of `center` (XY).
- `build_rotation_candidates(..., seed_yaw_deg=None, seed_yaw_tol=30)` — when set, keep only candidates whose implied yaw is within tol of `seed_yaw_deg`.

- [ ] **Step 1: Edit `pose_search.py::generate_translation_grid`** — add params + a post-filter. Insert after the existing quantile/chamfer grid `trans` is built (just before it is returned):

```python
def generate_translation_grid(starts, ends, num_trans=1700, chamfer_min_dist=0.3, spacing=None,
                              center=None, radius=None):
    # ... existing body unchanged, producing `trans` (M,3) ...
    if center is not None and radius is not None:
        import numpy as _np
        c = _np.asarray(center)[:2]
        keep = _np.linalg.norm(_np.asarray(trans)[:, :2] - c, axis=1) <= float(radius)
        if keep.any():
            trans = trans[keep]
    return trans
```

- [ ] **Step 2: Edit `pose_search.py::build_rotation_candidates`** — add yaw filter before returning `rotations, perms_expanded`:

```python
def build_rotation_candidates(principal_2d, principal_3d, seed_yaw_deg=None, seed_yaw_tol=30):
    # ... existing body unchanged, producing `rotations` (24,3,3), `perms_expanded` (24,3) ...
    if seed_yaw_deg is not None:
        import numpy as _np
        yaws = _np.degrees(_np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0]))  # yaw about world Z
        diff = _np.abs((yaws - float(seed_yaw_deg) + 180) % 360 - 180)
        keep = diff <= float(seed_yaw_tol)
        if keep.any():
            rotations, perms_expanded = rotations[keep], perms_expanded[keep]
    return rotations, perms_expanded
```

- [ ] **Step 3: Edit `multiroom_pose_estimation.py`** — read the 3 keys near the other cfg reads (~L163) and thread them into the two call sites:

```python
# near the other cfg.get(...) reads:
seed_trans_radius = cfg.get("seed_trans_radius", None)
seed_yaw_deg = cfg.get("seed_yaw_deg", None)
seed_yaw_tol = cfg.get("seed_yaw_tol", 30)
```
Local-mode translation grid call (~L329), pass the seed center (already available as the pano's raw-3D position `pano_positions[pano_name]`) + radius:
```python
local_trans = generate_translation_grid(local_starts, local_ends, num_trans=NUM_TRANS,
        chamfer_min_dist=CHAMFER_MIN_DIST,
        center=pano_positions[pano_name] if seed_trans_radius else None,
        radius=seed_trans_radius)
```
Rotation call (~L386):
```python
rotations, perms_expanded = build_rotation_candidates(principal_2d, principal_3d,
        seed_yaw_deg=seed_yaw_deg, seed_yaw_tol=seed_yaw_tol)
```

- [ ] **Step 4: Write the failing test** `experiments/fgpl_seed/tests/test_narrowing.py` (imports the FGPL functions directly)

```python
import sys, numpy as np, pytest
sys.path.insert(0, "/home/ruoyu/scan2measure-webframework/src/pose_estimation")
import pose_search  # noqa

def test_translation_grid_radius_filters_points():
    starts = np.array([[0, 0, 0], [5, 5, 0], [10, 0, 0]], float)
    ends = np.array([[1, 0, 0], [6, 5, 0], [11, 0, 0]], float)
    full = pose_search.generate_translation_grid(starts, ends, num_trans=200)
    near = pose_search.generate_translation_grid(starts, ends, num_trans=200,
                                                 center=[0, 0, 0], radius=2.0)
    assert len(near) <= len(full)
    assert np.all(np.linalg.norm(np.asarray(near)[:, :2], axis=1) <= 2.0 + 1e-6)

def test_rotation_yaw_filter_reduces_count():
    p2 = np.eye(3); p3 = np.eye(3)
    full, _ = pose_search.build_rotation_candidates(p2, p3)
    sub, _ = pose_search.build_rotation_candidates(p2, p3, seed_yaw_deg=0.0, seed_yaw_tol=20)
    assert len(sub) <= len(full) and len(sub) >= 1
```

- [ ] **Step 5: Run test to verify it passes** (default-off behavior preserved; filters work)

Run: `cd /home/ruoyu/PanoPin && conda run -n scan_env python -m pytest experiments/fgpl_seed/tests/test_narrowing.py -q`
Expected: PASS (2 tests). (Run in `scan_env` because it imports FGPL's numpy/torch stack.)

- [ ] **Step 6: Save the patch + commit**

```bash
cd /home/ruoyu/scan2measure-webframework && git diff src/pose_estimation/pose_search.py src/pose_estimation/multiroom_pose_estimation.py > /home/ruoyu/PanoPin/experiments/fgpl_seed/patches/fgpl_narrowing.patch
cd /home/ruoyu/PanoPin && git add experiments/fgpl_seed/patches/fgpl_narrowing.patch experiments/fgpl_seed/tests/test_narrowing.py
git commit -m "feat(fgpl-seed): flag-gated translation/rotation narrowing for P2/P3 + patch"
```
(The FGPL working-tree edits stay in the scan2measure repo; the patch under PanoPin makes them reproducible. Optionally commit them in scan2measure too.)

---

## Task 11: Full ablation + report

**Files:**
- Create: `experiments/fgpl_seed/run_all.py`

**Interfaces:**
- Consumes: all prior tasks + `cpo_cache.json`. Produces: `work/results/ablation.json` + a printed markdown table.

- [ ] **Step 1: Write `run_all.py`**

```python
"""Run all 5 arms on the subset, score each vs GT, emit a comparison table.
Arms: oracle / p1 / p2 / p3 / wrong_room. P1/P2/P3 read the CPO cache; P2 adds a
translation radius; P3 adds P2 + a per-pano yaw prior from CPO's R."""
import json, time, math, numpy as np
from experiments.fgpl_seed import (subset, seed_and_config as sc, run_arm, score, paths)
from eval import s3dis_gt

P2_RADIUS = 2.0            # meters (design §7; adjustable)
P3_YAW_TOL = 30.0          # degrees

ARMS = ["oracle", "p1", "p2", "p3", "wrong_room"]

def _yaw_of(R):
    return math.degrees(math.atan2(R[1][0], R[0][0]))

def main():
    rows = subset.build_subset()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    cpo = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    cents = sc.room_centroids(rows)
    md = paths.WORK / "configs" / "metadata.json"
    line_map = paths.WORK / "linemap" / "3d_line_map.pkl"
    feat, panos = paths.WORK / "features", paths.WORK / "panos"
    results = {}
    for arm in ARMS:
        narrowing = None
        if arm == "p2":
            narrowing = {"seed_trans_radius": P2_RADIUS}
        elif arm == "p3":
            # per-pano yaw prior: write it into the config is not per-pano; use the
            # median CPO yaw across panos as a single prior (first cut). Refine later.
            yaws = [_yaw_of(cpo[r["pano_name"]]["R"]) for r in rows]
            narrowing = {"seed_trans_radius": P2_RADIUS,
                         "seed_yaw_deg": float(np.median(yaws)), "seed_yaw_tol": P3_YAW_TOL}
        seed = sc.write_seed(arm, rows, gt, cpo, centroids=cents)
        cfg = sc.write_config(arm, rows, seed, line_map, md, feat, panos, narrowing=narrowing)
        t0 = time.time()
        poses = run_arm.run_arm(cfg, rows)
        dt = time.time() - t0
        s = score.score_arm(poses, gt, rows, cents)
        s["runtime_s"] = dt
        results[arm] = s
    out = paths.subdir("results") / "ablation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    # markdown table
    print("\n| arm | n_loc | trans median (m) | trans mean | rot median (deg) | wrong-room | runtime s |")
    print("|-----|-------|------------------|------------|------------------|-----------|-----------|")
    for arm in ARMS:
        s = results[arm]
        tm, tmn = s["translation"]["median"], s["translation"]["mean"]
        rm = s["rotation"]["median"]
        print(f"| {arm} | {s['n_localized']}/{len(rows)} | {tm} | {tmn} | {rm} | "
              f"{s['wrong_room_rate']} | {s['runtime_s']:.0f} |")
    print("\nwrote", out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full ablation** (needs Task 8 gate passed + Task 9 cache present + Task 10 edits applied)

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m experiments.fgpl_seed.run_all`
Expected: runs 5 arms (~45 min), prints the comparison table, writes `work/results/ablation.json`. **Interpretation:** Oracle = ceiling; Wrong-room = floor (high error / high wrong-room rate); P1/P2/P3 should sit between and ideally near Oracle. The sweet-spot arm (does position alone match Oracle, or is grid/rotation needed?) is the experiment's answer.

- [ ] **Step 3: Record results in `docs/PROGRESS.md` + a new decision in `docs/DECISIONS.md`**

Append a PROGRESS "Current state" block with the table and the verdict (which arm wins; did PanoPin seeding match/beat Oracle; wrong-room floor confirmed). Add a DECISIONS entry (Dxx) recording the ablation outcome + the P2 radius / P3 yaw choices. Flip `docs/tasks.json` T3/T4-related items only if their `verify` ran green.

- [ ] **Step 4: Commit**

```bash
git add experiments/fgpl_seed/run_all.py docs/PROGRESS.md docs/DECISIONS.md docs/tasks.json
git commit -m "feat(fgpl-seed): full 5-arm ablation runner + results"
```

---

## Self-Review

**1. Spec coverage** (against `docs/specs/2026-07-10-panopin-fgpl-seed-ablation-design.md`):
- §2 objective / FGPL loss untouched → P2/P3 only *filter candidate sets* (Task 10), never change scoring. ✓
- §4 raw-frame → identity `metadata.json` (Task 5) + raw-3D seeds. ✓
- §5 subset → Task 1 `ROOMS`, ≤2 panos/room. ✓
- §6 Phase-0 artifacts (line map / features / seed adapter) → Tasks 2-5; **oracle gate** → Task 8. ✓
- §7 arms Oracle/P1/P2/P3/Wrong-room → Tasks 5,9,10,11. ✓
- §8 metric (trans/rot error + wrong-room rate + runtime, via `eval/metrics.py`) → Task 7 + Task 11. ✓
- §9 components/isolation → file structure matches; fairness (GT only in experiments/) honored. ✓
- §10 risks (gate first; CPO rotation unvalidated = P3; frame identity check; nondeterminism `pin()`) → Task 8 diagnose step, Task 9 `pin()`, Task 11 P3. ✓

**2. Placeholder scan:** No "TBD"/"handle errors"/"similar to". The P3 per-pano-yaw simplification (single median yaw, first cut) is called out explicitly with a "refine later" note, not left vague. `paths.SCAN_ENV` is a single verified constant, not a placeholder. ✓

**3. Type consistency:** `build_subset()` row keys (`pano_name/uuid/room/pano_jpg/cloud_txt`) are consumed identically in build_ply/build_features/seed_and_config/cpo_seeds/score. `cpo_cache[pano] = {t,R,loss,room,per_room}` written in Task 9 matches the reads in Task 5's test and Task 11. `score_arm(poses, gt, rows, centroids)` signature matches Task 8/11 calls. `write_seed(arm, rows, gt, cpo, centroids=None)` matches all call sites. ✓

---

## Execution Handoff

Two known runtime unknowns are gated deliberately: the **Task 8 oracle gate** (does FGPL localize on S3DIS at all) and the **Ada/scan_env GPU** guard in `run_arm` (D20). Both are contingencies with explicit diagnose steps, not placeholders.
