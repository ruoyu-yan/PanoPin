# PanoPin v0 (CPO coarse room selector) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic coarse room-selector that assigns each S3DIS pano to a room by CPO color-consistency and beats the 5.9% random baseline (task T3).

**Architecture:** Vendor CPO (`82magnolia/panoramic-localization`, Apache-2.0) and compose its primitives in a returning orchestrator `localize_pair(cfg, pano_path, cloud_path) -> (t, R, loss)`. Rank candidate rooms by CPO's color-consistency `loss` via a two-tier funnel (Tier-1 `num_iter=1,top_k=1` over all rooms → Tier-2 `num_iter=100` over top-k), pick the min-loss room, emit room + coarse pose. See `docs/specs/2026-07-08-render-compare-cpo-design.md`.

**Tech Stack:** Python 3.8, PyTorch 1.10 (CPU), numpy/opencv/pandas/scipy/scikit-learn/Pillow; vendored CPO; harness is stdlib-only.

**Scope:** This plan covers **M0** (vendor + env + real-data smoke) and **M1/T3** (room selector beats baseline). M2 (coarse pose + FGPL hand-off, T4) and M3 (robustness, T5) are follow-on plans.

## Global Constraints
- **Deterministic / training-free only** — no neural-network training, no learned weights (D1, D7).
- **Fairness (D5):** the solver reads **only** the anonymized manifest (panos by uuid + candidate room clouds). Never read room/pose from filenames, pose JSONs, or `camera_to_room.json`. GT may be used **only** in `smoke/` dev scripts, never in `src/panopin/`.
- **Harness stays stdlib-only (D6):** never add deps to `eval/`. Solver deps live in the `panopin` conda env only.
- **Solver env (D10):** python 3.8, torch 1.10 CPU, numpy~=1.23, opencv-python, pandas, scipy, scikit-learn, Pillow, tqdm, pyyaml, matplotlib.
- **Point-cloud format:** whitespace `X Y Z R G B`, RGB in **0–255** (CPO's `read_txt_pcd` divides by 255 → [0,1]). Pano: equirectangular RGB, CPO resizes to **2048×1024**, z-up frame.
- **CPO cfg:** `dataset` must be `'stanford'`. Build cfg from `third_party/cpo/config/stanford_cpo.ini`; override immutably via `cfg._replace(...)`.
- **Licensing:** keep `third_party/cpo/LICENSE` (Apache-2.0) + `third_party/cpo/ATTRIBUTION.md` (upstream commit SHA, authors, changes). Cite CPO (ECCV 2022) in any paper.
- **Run all Python in the `panopin` env** (`conda run -n panopin ...` or after `conda activate panopin`).

## File Structure
- Create `third_party/cpo/` — vendored CPO (repo minus `.git`, `ldl/superglue_models/`); keep `LICENSE`; add `ATTRIBUTION.md`.
- Create `src/panopin/_cpo_path.py` — puts `third_party/cpo/` on `sys.path` (import for its side effect before importing CPO modules).
- Create `src/panopin/cpo_config.py` — `load_cfg(**overrides)`, `TIER1`, `TIER2`.
- Create `src/panopin/cpo_adapter.py` — `localize_pair(cfg, pano_path, cloud_path) -> (t, R, loss)`.
- Create `src/panopin/select_room.py` — `select_room(pano_path, candidate_rooms, top_k=5) -> RoomResult`.
- Create `src/panopin/solve.py` — CLI: manifest → predictions JSON.
- Create `tests/conftest.py` — put `src/` and `third_party/cpo/` on `sys.path`; synthetic fixtures.
- Create `tests/synthetic.py` — `box_room(...)`, `write_cloud_txt(...)`, `render_pano_png(...)`.
- Create `tests/test_env_cpo.py`, `tests/test_cpo_config.py`, `tests/test_cpo_adapter.py`, `tests/test_select_room.py`, `tests/test_solve.py`.
- Create `smoke/reproduce_cpo_one_room.py` — M0 real-data evidence (GT dev-only).
- Modify `docs/PROGRESS.md`, `docs/tasks.json` (final task).

---

### Task 1: Vendor CPO + build the `panopin` env

**Files:**
- Create: `third_party/cpo/` (vendored), `third_party/cpo/ATTRIBUTION.md`
- Create: `src/panopin/_cpo_path.py`
- Create: `tests/conftest.py`, `tests/test_env_cpo.py`

**Interfaces:**
- Produces: importable CPO modules `color_utils`, `utils`, `data_utils`, `dict_utils`, `cpo.sampling_loss`; a working `panopin` env.

- [ ] **Step 1: Write the failing test**

`tests/test_env_cpo.py`:
```python
def test_cpo_primitives_import():
    import _cpo_path  # noqa: F401  (adds third_party/cpo to sys.path)
    import color_utils, utils, data_utils, dict_utils
    from cpo.sampling_loss import refine_pose_sampling_loss
    for name in ("read_txt_pcd",):
        assert hasattr(data_utils, name)
    for name in ("histogram_pose_search", "make_pano", "make_score_map_2d", "make_score_map_3d"):
        assert hasattr(utils, name)
    assert callable(refine_pose_sampling_loss)

def test_torch_cpu_op():
    import torch
    x = torch.ones(3) + torch.ones(3)
    assert x.sum().item() == 6.0
```

`tests/conftest.py`:
```python
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "panopin"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "cpo"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n base python -m pytest tests/test_env_cpo.py -v` (env/vendor absent yet)
Expected: FAIL (ModuleNotFoundError: color_utils / no `panopin` env).

- [ ] **Step 3: Vendor CPO**

```bash
cd /home/ruoyu/PanoPin
git clone https://github.com/82magnolia/panoramic-localization third_party/cpo
( cd third_party/cpo && git rev-parse HEAD ) > /tmp/cpo_sha.txt
rm -rf third_party/cpo/.git third_party/cpo/ldl/superglue_models
test -f third_party/cpo/LICENSE && echo "LICENSE present"
```

`third_party/cpo/ATTRIBUTION.md`:
```markdown
# Vendored: 82magnolia/panoramic-localization (CPO/PICCOLO/LDL/FGPL)
- Upstream: https://github.com/82magnolia/panoramic-localization
- Commit: <paste contents of /tmp/cpo_sha.txt>
- License: Apache-2.0 (see ./LICENSE). Authors: Junho Kim, Hojun Jang, Changwoon Choi, Young Min Kim.
- Changes from upstream: removed `.git/` and `ldl/superglue_models/` (LDL-only weights, unused).
  No source edits. PanoPin composes CPO primitives from `src/panopin/`; cite CPO (ECCV 2022).
```

- [ ] **Step 4: Create the sys.path shim**

`src/panopin/_cpo_path.py`:
```python
"""Import for side effect: puts vendored CPO on sys.path so its bare imports resolve."""
import os, sys
_CPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "cpo"))
if _CPO not in sys.path:
    sys.path.insert(0, _CPO)
```

- [ ] **Step 5: Build the env**

```bash
conda create -n panopin python=3.8 -y
conda run -n panopin pip install "torch==1.10.1+cpu" -f https://download.pytorch.org/whl/torch_stable.html
conda run -n panopin pip install "numpy==1.23.5" opencv-python pandas scipy scikit-learn Pillow tqdm pyyaml matplotlib
```
Fallback if the 1.10.1+cpu wheel won't install on this platform: `conda run -n panopin pip install torch --index-url https://download.pytorch.org/whl/cpu` (newer CPU torch; the CPO color path uses only basic ops and should run — note the substitution in ATTRIBUTION/PROGRESS if used).

- [ ] **Step 6: Run test to verify it passes**

Run: `conda run -n panopin python -m pytest tests/test_env_cpo.py -v`
Expected: PASS (both tests).

- [ ] **Step 7: Commit**

```bash
git add third_party/cpo src/panopin/_cpo_path.py tests/conftest.py tests/test_env_cpo.py
git commit -m "feat(cpo): vendor CPO (Apache-2.0) + panopin env + import smoke test"
```

---

### Task 2: CPO config builder

**Files:**
- Create: `src/panopin/cpo_config.py`
- Test: `tests/test_cpo_config.py`

**Interfaces:**
- Consumes: `third_party/cpo/config/stanford_cpo.ini`, CPO `parse_utils.parse_ini`.
- Produces: `load_cfg(**overrides) -> cfg` (namedtuple; `cfg.dataset == "stanford"`); dict constants `TIER1 = {"num_iter": 1, "top_k_candidate": 1}`, `TIER2 = {"num_iter": 100}`.

- [ ] **Step 1: Write the failing test**

`tests/test_cpo_config.py`:
```python
def test_load_cfg_defaults():
    from panopin.cpo_config import load_cfg
    cfg = load_cfg()
    assert cfg.dataset == "stanford"
    assert hasattr(cfg, "num_split_h") and hasattr(cfg, "num_iter")

def test_load_cfg_override_is_immutable_replace():
    from panopin.cpo_config import load_cfg, TIER1
    cfg = load_cfg(**TIER1)
    assert cfg.num_iter == 1
    assert cfg.top_k_candidate == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_cpo_config.py -v`
Expected: FAIL (No module named 'panopin.cpo_config').

- [ ] **Step 3: Write minimal implementation**

`src/panopin/cpo_config.py`:
```python
"""Build a CPO cfg namedtuple from stanford_cpo.ini, with immutable overrides."""
import os
from panopin import _cpo_path  # noqa: F401  (sys.path side effect)
from parse_utils import parse_ini

_INI = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "third_party", "cpo", "config", "stanford_cpo.ini"))

TIER1 = {"num_iter": 1, "top_k_candidate": 1}   # cheap cost at the coarse pose (all rooms)
TIER2 = {"num_iter": 100}                        # full Adam refine (top-k rooms)

def load_cfg(**overrides):
    cfg = parse_ini(_INI)                        # namedtuple
    if overrides:
        cfg = cfg._replace(**overrides)
    return cfg
```
Note: if `parse_ini` takes a different argument (open file vs path) or returns a non-namedtuple, adapt this wrapper — inspect `third_party/cpo/parse_utils.py` and keep the `load_cfg(**overrides) -> cfg` contract. If the parsed object lacks a key in `overrides` (so `_replace` raises), add the key to a local dict and rebuild via the same namedtuple type.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n panopin python -m pytest tests/test_cpo_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/panopin/cpo_config.py tests/test_cpo_config.py
git commit -m "feat(cpo): config builder with tier-1/tier-2 overrides"
```

---

### Task 3: `localize_pair` — returning CPO orchestrator + synthetic fixtures

**Files:**
- Create: `src/panopin/cpo_adapter.py`
- Create: `tests/synthetic.py`
- Test: `tests/test_cpo_adapter.py`

**Interfaces:**
- Consumes: `load_cfg` (Task 2); CPO `data_utils.read_txt_pcd`, `dict_utils.get_init_dict_cpo`, `color_utils.color_match/color_mod`, `utils.{generate_trans_points,generate_rot_points,make_score_map_2d,process_score_map_2d,make_score_map_3d,histogram_pose_search,make_pano}`, `cpo.sampling_loss.refine_pose_sampling_loss`.
- Produces: `localize_pair(cfg, pano_path, cloud_path) -> (t: np.ndarray shape (3,), R: np.ndarray shape (3,3), loss: float)`. And test helpers `synthetic.box_room(w,d,h,patch=None) -> (xyz, rgb)`, `synthetic.write_cloud_txt(path, xyz, rgb)`, `synthetic.render_pano_png(path, xyz, rgb, trans, R)`.

- [ ] **Step 1: Write the synthetic fixtures**

`tests/synthetic.py`:
```python
import numpy as np

def box_room(w=4.0, d=4.0, h=2.8, n=40000, patch=None, seed=0):
    """Colored point cloud of a box room. Walls beige; optional colored `patch`
    = (wall, rgb) where wall in {'x0','x1','y0','y1'} paints that wall's center."""
    rng = np.random.RandomState(seed)
    pts, cols = [], []
    beige = np.array([200, 190, 170], float)
    def face(fixed_axis, fixed_val, a_rng, b_rng, axes):
        a = rng.uniform(*a_rng, n // 6); b = rng.uniform(*b_rng, n // 6)
        p = np.zeros((n // 6, 3)); p[:, axes[0]] = a; p[:, axes[1]] = b; p[:, fixed_axis] = fixed_val
        return p
    faces = {
        'x0': face(0, 0, (0, d), (0, h), (1, 2)), 'x1': face(0, w, (0, d), (0, h), (1, 2)),
        'y0': face(1, 0, (0, w), (0, h), (0, 2)), 'y1': face(1, d, (0, w), (0, h), (0, 2)),
        'z0': face(2, 0, (0, w), (0, d), (0, 1)), 'z1': face(2, h, (0, w), (0, d), (0, 1)),
    }
    for name, p in faces.items():
        c = np.tile(beige, (len(p), 1))
        if patch is not None and patch[0] == name:
            c[:] = np.array(patch[1], float)
        pts.append(p); cols.append(c)
    return np.concatenate(pts), np.concatenate(cols)

def write_cloud_txt(path, xyz, rgb):
    np.savetxt(path, np.hstack([xyz, rgb]), fmt="%.4f")  # X Y Z R G B, RGB 0-255

def render_pano_png(path, xyz, rgb, trans, R):
    import cv2, _cpo_path  # noqa: F401
    from utils import make_pano
    import torch
    xt = torch.from_numpy(xyz).float(); rt = torch.from_numpy(rgb / 255.).float()
    centered = (xt - torch.tensor(trans).float()) @ torch.tensor(R).float().T
    pano = make_pano(centered, rt, resolution=(1024, 2048))   # uint8 HxWx3
    cv2.imwrite(path, cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
```

- [ ] **Step 2: Write the failing test**

`tests/test_cpo_adapter.py`:
```python
import os, numpy as np, tests.synthetic as S

def _prep(tmp_path, patch, pose_trans):
    xyz, rgb = S.box_room(patch=patch)
    cloud = str(tmp_path / "room.txt"); S.write_cloud_txt(cloud, xyz, rgb)
    R = np.eye(3)
    pano = str(tmp_path / "q.png"); S.render_pano_png(pano, xyz, rgb, pose_trans, R)
    return pano, cloud

def test_localize_pair_returns_shapes_and_finite_loss(tmp_path):
    from panopin.cpo_config import load_cfg, TIER2
    from panopin.cpo_adapter import localize_pair
    pano, cloud = _prep(tmp_path, patch=('x1', [220, 40, 40]), pose_trans=[2.0, 2.0, 1.5])
    cfg = load_cfg(**TIER2, sample_rate=1)
    t, R, loss = localize_pair(cfg, pano, cloud)
    assert t.shape == (3,) and R.shape == (3, 3)
    assert np.isfinite(loss)

def test_matching_cloud_scores_lower_than_mismatched(tmp_path):
    from panopin.cpo_config import load_cfg, TIER1
    from panopin.cpo_adapter import localize_pair
    # pano is of a room with a RED patch on wall x1
    pano, match_cloud = _prep(tmp_path, patch=('x1', [220, 40, 40]), pose_trans=[2.0, 2.0, 1.5])
    # a mismatched candidate: patch is BLUE and on a different wall
    xyz2, rgb2 = S.box_room(patch=('y0', [40, 40, 220]))
    mis_cloud = str(tmp_path / "mis.txt"); S.write_cloud_txt(mis_cloud, xyz2, rgb2)
    cfg = load_cfg(**TIER1, sample_rate=1)
    _, _, loss_match = localize_pair(cfg, pano, match_cloud)
    _, _, loss_mis = localize_pair(cfg, pano, mis_cloud)
    assert loss_match < loss_mis   # color consistency discriminates
```

- [ ] **Step 3: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_cpo_adapter.py -v`
Expected: FAIL (No module named 'panopin.cpo_adapter').

- [ ] **Step 4: Write `localize_pair`** (transcribed from `third_party/cpo/cpo/localize_single.py:38-150`, viz removed, returns added)

`src/panopin/cpo_adapter.py`:
```python
"""Return-valued CPO single-pair localizer (composes CPO primitives).
Body mirrors third_party/cpo/cpo/localize_single.py (lines 38-150), minus the
result.png visualization, plus a return of (t, R, loss)."""
import numpy as np, torch, cv2
from panopin import _cpo_path  # noqa: F401
import data_utils
from dict_utils import get_init_dict_cpo
from color_utils import color_match, color_mod
from utils import (out_of_room, generate_trans_points, generate_rot_points,
                   make_score_map_2d, process_score_map_2d, make_score_map_3d,
                   histogram_pose_search)
from cpo.sampling_loss import refine_pose_sampling_loss


def localize_pair(cfg, pano_path, cloud_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sample_rate = getattr(cfg, 'sample_rate', 1)
    top_k_candidate = getattr(cfg, 'top_k_candidate', 5)
    init_downsample_h = getattr(cfg, 'init_downsample_h', 1)
    init_downsample_w = getattr(cfg, 'init_downsample_w', 1)
    main_downsample_h = getattr(cfg, 'main_downsample_h', 1)
    main_downsample_w = getattr(cfg, 'main_downsample_w', 1)

    xyz_np, rgb_np = data_utils.read_txt_pcd(cloud_path, sample_rate=sample_rate)
    xyz = torch.from_numpy(xyz_np).float().to(device)
    rgb = torch.from_numpy(rgb_np).float().to(device)

    orig_img = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (2048, 1024))

    sharpen_color = getattr(cfg, 'sharpen_color', False)
    match_color = getattr(cfg, 'match_color', False)
    num_bins = getattr(cfg, 'num_bins', 256)
    mod_img = (torch.from_numpy(orig_img).float() / 255.).to(device)
    if sharpen_color or match_color:
        if match_color:
            new_img = color_match(mod_img, rgb); orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)
        if sharpen_color:
            new_img, rgb = color_mod(mod_img, rgb, num_bins); orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)

    img = cv2.resize(orig_img, (orig_img.shape[1] // init_downsample_w, orig_img.shape[0] // init_downsample_h))
    img = (torch.from_numpy(img).float() / 255.).to(device)
    input_xyz = xyz
    init_dict = get_init_dict_cpo(cfg)

    inlier_init_dict = dict(init_dict); inlier_init_dict['is_inlier_dict'] = True
    inlier_init_dict['num_trans'] = getattr(cfg, 'inlier_num_trans', init_dict['num_trans'])
    inlier_init_dict['num_yaw'] = getattr(cfg, 'inlier_num_yaw', 4)
    inlier_init_dict['num_pitch'] = getattr(cfg, 'inlier_num_pitch', 4)
    inlier_init_dict['num_roll'] = getattr(cfg, 'inlier_num_roll', 4)
    inlier_init_dict['trans_init_mode'] = getattr(cfg, 'inlier_trans_init_mode', 'quantile')
    inlier_test_trans = generate_trans_points(input_xyz, inlier_init_dict, device=input_xyz.device)
    inlier_test_rot = generate_rot_points(inlier_init_dict, device=input_xyz.device)
    inlier_num_split_h = getattr(cfg, 'inlier_num_split_h', 8)
    inlier_num_split_w = getattr(cfg, 'inlier_num_split_w', 16)
    margin = inlier_num_split_h // 8

    score_map_2d = make_score_map_2d(img, input_xyz, rgb, inlier_test_trans, inlier_test_rot,
                                     inlier_num_split_h, inlier_num_split_w, margin)
    score_map_2d_search = process_score_map_2d(torch.zeros(cfg.num_split_h, cfg.num_split_w, device=xyz.device),
                                               score_map_2d, 'preserve', 0.0)
    score_map_2d_refine = process_score_map_2d(torch.from_numpy(orig_img).to(xyz.device),
                                               score_map_2d, 'preserve', 0.0).unsqueeze(-1)
    score_map_3d = make_score_map_3d(img, xyz, rgb, inlier_test_trans, inlier_test_rot,
                                     inlier_num_split_h, inlier_num_split_w, margin, match_rgb=match_color)
    pcd_weight = score_map_3d

    rot = generate_rot_points(init_dict, device=img.device)
    trans = generate_trans_points(xyz, init_dict, device=img.device)
    input_trans, input_rot = histogram_pose_search(img, input_xyz, rgb, trans, rot, top_k_candidate,
                                                    init_dict['num_split_h'], init_dict['num_split_w'],
                                                    score_map_2d_search, init_dict['sin_hist'])

    img = cv2.resize(orig_img, (orig_img.shape[1] // main_downsample_w, orig_img.shape[0] // main_downsample_h))
    img = (torch.from_numpy(img).float() / 255.).to(device)

    result = []
    for i in range(top_k_candidate):
        result.append(refine_pose_sampling_loss(img, input_xyz, rgb, input_trans, input_rot, i, cfg,
                                                 img_weight=score_map_2d_refine, pcd_weight=pcd_weight))
    result = np.asarray(result, dtype=object)
    min_ind = int(result[:, 2].argmin())
    t = np.asarray(result[min_ind, 0]).reshape(3)
    R = np.asarray(result[min_ind, 1]).reshape(3, 3)
    loss = float(result[min_ind, 2])
    return t, R, loss
```
Note: `histogram_pose_search` needs `top_k_candidate` candidate poses; with TIER1 (`top_k_candidate=1`) the refine loop runs once. If any CPO call errors on a signature mismatch, open the corresponding `third_party/cpo` module and align the call — do NOT invent behavior; the reference wiring is `localize_single.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n panopin python -m pytest tests/test_cpo_adapter.py -v`
Expected: PASS (shapes/finite-loss + matching-cloud-scores-lower). If `test_matching_cloud_scores_lower` is flaky on sparse synthetic clouds, raise `n` in `box_room` and keep `match_color=True` (default in `stanford_cpo.ini`); the discrimination must hold.

- [ ] **Step 6: Commit**

```bash
git add src/panopin/cpo_adapter.py tests/synthetic.py tests/test_cpo_adapter.py
git commit -m "feat(cpo): returning localize_pair orchestrator + synthetic fixtures"
```

---

### Task 4: M0 smoke — reproduce CPO on one real Area_3 room

**Files:**
- Create: `smoke/reproduce_cpo_one_room.py`

**Interfaces:**
- Consumes: `load_cfg`, `localize_pair`; `eval/s3dis_gt.py` (GT — **dev-only**, allowed in `smoke/`).

- [ ] **Step 1: Write the smoke script**

`smoke/reproduce_cpo_one_room.py`:
```python
"""M0 evidence: run localize_pair on ONE real Area_3 room + its GT pano; print
translation error vs GT. GT is used here for DEV VALIDATION ONLY (never in src/panopin)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from panopin.cpo_config import load_cfg, TIER2
from panopin.cpo_adapter import localize_pair
import s3dis_gt

def main():
    cfg = load_cfg(**TIER2, sample_rate=10)          # subsample big clouds for CPU speed
    gtcfg = s3dis_gt.load_config(None)
    gt = s3dis_gt.load_gt("Area_3", gtcfg)
    uuid = next(u for u, g in gt.items() if g.get("pano_rgb") and g.get("room"))  # a pano with GT
    room = gt[uuid]["room"]
    cloud = os.path.join(gtcfg["s3dis"]["Area_3"]["rooms_dir"], room, room + ".txt")
    pano = gt[uuid]["pano_rgb"]
    t, R, loss = localize_pair(cfg, pano, cloud)
    t_gt = np.asarray(gt[uuid]["pose"]["t"], float)
    print(f"room={room} loss={loss:.4f} t_est={t} t_gt={t_gt} err={np.linalg.norm(t - t_gt):.3f} m")
    assert np.linalg.norm(t - t_gt) < 1.5, "M0: localization far from GT — investigate frame/scale"

if __name__ == "__main__":
    main()
```
Note: adapt the GT field names (`pano_rgb`, `pose`/`t`, `room`) to `eval/s3dis_gt.py`'s actual return schema — read it first. The 1.5 m bound is a loose sanity gate, not precision.

- [ ] **Step 2: Run the smoke test**

Run: `conda run -n panopin python smoke/reproduce_cpo_one_room.py`
Expected: prints `room=… loss=… err=<…> m` with err < 1.5 m. If it fails the frame/scale assert, verify the S3DIS pano/cloud share CPO's z-up `atan2(y,x)` convention (spec §9 risk 5) before proceeding.

- [ ] **Step 3: Commit**

```bash
git add smoke/reproduce_cpo_one_room.py
git commit -m "test(smoke): M0 reproduce CPO on one real Area_3 room"
```

---

### Task 5: `select_room` — two-tier funnel

**Files:**
- Create: `src/panopin/select_room.py`
- Test: `tests/test_select_room.py`

**Interfaces:**
- Consumes: `load_cfg, TIER1, TIER2` (Task 2), `localize_pair` (Task 3).
- Produces: `RoomResult = namedtuple("RoomResult", "room t R loss")`; `select_room(pano_path, candidate_rooms: dict[str,str], top_k=5, sample_rate=10) -> RoomResult`.

- [ ] **Step 1: Write the failing test (the same-shape disambiguation test)**

`tests/test_select_room.py`:
```python
import numpy as np, tests.synthetic as S

def test_picks_the_same_shape_room_with_matching_content(tmp_path):
    from panopin.select_room import select_room
    # Two IDENTICAL-SHAPE rooms; only the colored patch differs (wall + color).
    A_xyz, A_rgb = S.box_room(patch=('x1', [220, 40, 40]))    # red on x1
    B_xyz, B_rgb = S.box_room(patch=('y0', [40, 40, 220]))    # blue on y0
    a = str(tmp_path / "A.txt"); S.write_cloud_txt(a, A_xyz, A_rgb)
    b = str(tmp_path / "B.txt"); S.write_cloud_txt(b, B_xyz, B_rgb)
    # Query pano is taken inside room A.
    pano = str(tmp_path / "q.png"); S.render_pano_png(pano, A_xyz, A_rgb, [2.0, 2.0, 1.5], np.eye(3))
    res = select_room(pano, {"A": a, "B": b}, top_k=2, sample_rate=1)
    assert res.room == "A"          # shape can't tell them apart; content must

def test_returns_room_in_candidate_set(tmp_path):
    from panopin.select_room import select_room
    xyz, rgb = S.box_room(patch=('x1', [220, 40, 40]))
    a = str(tmp_path / "A.txt"); S.write_cloud_txt(a, xyz, rgb)
    pano = str(tmp_path / "q.png"); S.render_pano_png(pano, xyz, rgb, [2.0, 2.0, 1.5], np.eye(3))
    res = select_room(pano, {"A": a}, top_k=1, sample_rate=1)
    assert res.room == "A" and res.t.shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_select_room.py -v`
Expected: FAIL (No module named 'panopin.select_room').

- [ ] **Step 3: Write minimal implementation**

`src/panopin/select_room.py`:
```python
"""Two-tier room selector: cheap Tier-1 cost over all rooms -> refine top-k."""
from collections import namedtuple
from panopin.cpo_config import load_cfg, TIER1, TIER2
from panopin.cpo_adapter import localize_pair

RoomResult = namedtuple("RoomResult", "room t R loss")

def select_room(pano_path, candidate_rooms, top_k=5, sample_rate=10):
    cfg1 = load_cfg(**TIER1, sample_rate=sample_rate)
    cfg2 = load_cfg(**TIER2, sample_rate=sample_rate)
    # Tier 1: cheap cost for every candidate room.
    tier1 = []
    for room, cloud in candidate_rooms.items():
        _, _, loss = localize_pair(cfg1, pano_path, cloud)
        tier1.append((loss, room, cloud))
    tier1.sort(key=lambda x: x[0])
    survivors = tier1[:max(1, min(top_k, len(tier1)))]
    # Tier 2: full refine on survivors; pick global min loss.
    best = None
    for _, room, cloud in survivors:
        t, R, loss = localize_pair(cfg2, pano_path, cloud)
        if best is None or loss < best.loss:
            best = RoomResult(room, t, R, loss)
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n panopin python -m pytest tests/test_select_room.py -v`
Expected: PASS — the same-shape room is disambiguated by content. (If flaky, increase `box_room` `n` and keep `top_k>=2`.)

- [ ] **Step 5: Commit**

```bash
git add src/panopin/select_room.py tests/test_select_room.py
git commit -m "feat: two-tier CPO room selector (same-shape disambiguation test passes)"
```

---

### Task 6: `solve.py` — manifest → predictions CLI

**Files:**
- Create: `src/panopin/solve.py`
- Test: `tests/test_solve.py`

**Interfaces:**
- Consumes: `select_room` (Task 5).
- Produces: CLI `python -m panopin.solve --manifest PATH --out PATH [--top-k 5] [--sample-rate 10]`; writes predictions JSON per `eval/PREDICTIONS_SCHEMA.md` (`{"area","method","predictions":{uuid:{"room":..., "coarse_pose":{"t":[x,y,z]}}}}`). Reads ONLY the manifest.

- [ ] **Step 1: Write the failing test**

`tests/test_solve.py`:
```python
import json, numpy as np, tests.synthetic as S

def test_solve_writes_valid_predictions(tmp_path):
    from panopin.solve import run
    A_xyz, A_rgb = S.box_room(patch=('x1', [220, 40, 40]))
    B_xyz, B_rgb = S.box_room(patch=('y0', [40, 40, 220]))
    a = str(tmp_path / "A.txt"); S.write_cloud_txt(a, A_xyz, A_rgb)
    b = str(tmp_path / "B.txt"); S.write_cloud_txt(b, B_xyz, B_rgb)
    pano = str(tmp_path / "u1.png"); S.render_pano_png(pano, A_xyz, A_rgb, [2.0, 2.0, 1.5], np.eye(3))
    manifest = str(tmp_path / "manifest.json")
    json.dump({"area": "Synth", "panos": {"u1": {"image": pano}},
               "candidate_rooms": {"A": a, "B": b}}, open(manifest, "w"))
    out = str(tmp_path / "pred.json")
    run(manifest, out, top_k=2, sample_rate=1)
    pred = json.load(open(out))
    assert pred["predictions"]["u1"]["room"] in {"A", "B"}
    assert len(pred["predictions"]["u1"]["coarse_pose"]["t"]) == 3
    assert pred["predictions"]["u1"]["room"] == "A"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_solve.py -v`
Expected: FAIL (No module named 'panopin.solve').

- [ ] **Step 3: Write minimal implementation**

`src/panopin/solve.py`:
```python
"""PanoPin solver: anonymized manifest -> predictions JSON. Reads ONLY the manifest (fairness)."""
import argparse, json
from panopin.select_room import select_room

def run(manifest_path, out_path, top_k=5, sample_rate=10):
    man = json.load(open(manifest_path))
    rooms = man["candidate_rooms"]
    preds = {}
    for uuid, p in man["panos"].items():
        img = p.get("image")
        if not img:
            continue
        res = select_room(img, rooms, top_k=top_k, sample_rate=sample_rate)
        preds[uuid] = {"room": res.room, "coarse_pose": {"t": [float(x) for x in res.t]}}
    out = {"area": man.get("area", ""), "method": "cpo_render_compare_v0", "predictions": preds}
    json.dump(out, open(out_path, "w"), indent=2)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=5); ap.add_argument("--sample-rate", type=int, default=10)
    a = ap.parse_args(); run(a.manifest, a.out, a.top_k, a.sample_rate)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n panopin python -m pytest tests/test_solve.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/panopin/solve.py tests/test_solve.py
git commit -m "feat: solver CLI (manifest -> predictions), manifest-only"
```

---

### Task 7: T3 acceptance — beat the baseline on Area_3

**Files:**
- Modify: `docs/PROGRESS.md`, `docs/tasks.json`

**Interfaces:**
- Consumes: `eval/make_manifest.py`, `src/panopin/solve.py`, `eval/score.py`.

- [ ] **Step 1: Build the anonymized manifest**

Run: `conda run -n panopin python eval/make_manifest.py --area Area_3`
Expected: `wrote runs/manifest_Area_3/manifest.json (85 panos, … linked, 23 candidate rooms)`.

- [ ] **Step 2: Sanity run on a SUBSET first (CPU cost control)**

Create `runs/manifest_Area_3/manifest_head.json` = the manifest with only the first ~8 panos (copy JSON, trim `panos`). Run:
`PYTHONPATH=src conda run -n panopin python -m panopin.solve --manifest runs/manifest_Area_3/manifest_head.json --out runs/pred_head.json --top-k 5 --sample-rate 20`
Expected: completes without error; `runs/pred_head.json` has room + coarse_pose per pano. Note wall-clock per pano; if too slow, raise `--sample-rate` (more subsampling) and/or lower `--top-k`.

- [ ] **Step 3: Full run (may be long — run in background/overnight)**

Run: `PYTHONPATH=src conda run -n panopin python -m panopin.solve --manifest runs/manifest_Area_3/manifest.json --out runs/pred_cpo_v0.json --top-k 5 --sample-rate 20`
Expected: `runs/pred_cpo_v0.json` covering all linked panos.

- [ ] **Step 4: Score it**

Run: `conda run -n panopin python eval/score.py --predictions runs/pred_cpo_v0.json --area Area_3`
Expected: **room accuracy ≫ 0.06**. Record the exact number.

- [ ] **Step 5: Record results + flip T3**

Update `docs/PROGRESS.md` (new session entry: the accuracy number, per-room notes on the ~7 identical-shape pairs, wall-clock, chosen `sample_rate`/`top_k`). In `docs/tasks.json` set T3 `status:"done", passes:true, notes:"…accuracy=<N> via runs/pred_cpo_v0.json"` **only if** the scored accuracy cleared the baseline.

- [ ] **Step 6: Commit**

```bash
git add docs/PROGRESS.md docs/tasks.json
git commit -m "feat: T3 — CPO room selector beats random baseline on Area_3 (acc=<N>)"
```

---

## Self-Review

**1. Spec coverage:** build-upon-CPO (Tasks 1,3) ✓; compose-primitives returning orchestrator D9 (Task 3) ✓; two-tier funnel D11 (Tasks 2,5) ✓; dedicated env D10 (Task 1) ✓; fairness manifest-only D5 (Task 6, GT only in Task 4 smoke) ✓; render-and-compare decider D8 (Task 5 same-shape test) ✓; coarse pose emit (Task 6 `coarse_pose.t`) ✓; accuracy-bar/T3 (Task 7) ✓; Apache-2.0 attribution (Task 1) ✓. Deferred by scope: M2 FGPL hand-off adapter, M3 all-85 robustness/diagnostic — follow-on plans.

**2. Placeholder scan:** no "TBD/TODO"; each code step carries complete code. The "adapt if signature differs" notes point at the exact reference file (`localize_single.py`) — grounding, not placeholders.

**3. Type consistency:** `load_cfg(**overrides)->cfg`, `TIER1/TIER2` dicts, `localize_pair(cfg,pano_path,cloud_path)->(t(3,),R(3,3),loss:float)`, `select_room(pano_path,candidate_rooms,top_k,sample_rate)->RoomResult(room,t,R,loss)`, `run(manifest,out,top_k,sample_rate)` — names/shapes consistent across Tasks 2→3→5→6.
