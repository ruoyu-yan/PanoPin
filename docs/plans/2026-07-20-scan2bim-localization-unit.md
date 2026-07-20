# Scan2BIM Localization Unit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Stage 0 to the Point_360 (Scan2BIM) pipeline that produces panorama camera poses from PanoPin-seeded, upright-prior-enabled FGPL, with FGPL and PanoPin as pinned submodules.

**Architecture:** Point_360 gains `run_localization.py` (thin orchestrator) plus a `localization/` package holding a pure pose adapter and a preflight checker. FGPL and PanoPin become pinned git submodules. The three conda envs (`point360`, `panopin-gpu`, `scan_env`) cannot share a process, so cross-env work happens via `conda run`, following the existing `run_scan2measure_3d_lines.py:186` pattern.

**Tech Stack:** Python 3.12 (`point360`), Python 3.8 + torch 2.0.1+cu118 (`panopin-gpu`), Python 3.8 + torch 1.12+cu116 + open3d 0.19 (`scan_env`), git submodules, numpy.

**Spec:** `docs/specs/2026-07-20-scan2bim-localization-unit-design.md`

## Global Constraints

- **Never touch `/home/ruoyu/Point_360` directly.** Another session owns that working tree. All Point_360 work happens in the worktree created in Task 1.
- **Point_360 test convention:** plain-assert scripts run as `python test_x.py`, tests beside their subjects, ending in an `if __name__ == "__main__":` block that calls each test and prints `ALL PASS`. **No pytest in Point_360.**
- **PanoPin test convention:** pytest, via `conda run -n panopin python -m pytest tests/... -q`.
- **The pose contract is exactly** `{pano_name: {"R": [[3x3]], "t": [x,y,z]}}` where `R` is world→camera and `t` is the camera centre in world coordinates. Extra top-level keys break nothing, but per-entry extras must not be added — quality data goes in a sidecar.
- **FGPL submodule pin: `528061b`** (HEAD of `feat/upright-rotation-prior` in `github.com/ruoyu-yan/scan2measure-webframework`). The branch stays unmerged. Do not delete or force-push it.
- **Upright prior config keys:** `"upright_prior": true`, `"up_world": [0.0, 0.0, 1.0]`, `"max_tilt_deg": 10.0`. Default is `upright_prior: False` (`multiroom_pose_estimation.py:63`).
- **Manhattan pool, in order:** `["office_5", "hallway_1", "lounge_1", "conferenceRoom_1", "WC_1"]` — 22 panos total.
- **Raw room clouds:** `/home/ruoyu/Point_360/data/s3dis/Stanford3dDataset_v1.2/Area_3/<room>/<room>.txt`
- **Never read room or pose identity from filenames, `camera_to_room.json`, or pose files when scoring** (PanoPin fairness rule D5).
- **Per global CLAUDE.md:** after writing or editing any Python file, run it to verify no errors. If it errors, undo the change completely and rewrite that section from scratch rather than patching in place.

---

### Task 1: Worktree and branch setup

**Files:**
- Create: worktree at `/home/ruoyu/Point_360-localization`

**Interfaces:**
- Consumes: nothing
- Produces: a working directory on branch `feat/localization-unit`, based on `s3dis-eval` (548728b), where all subsequent Point_360 tasks run.

- [ ] **Step 1: Confirm the other session's branch is untouched**

```bash
cd /home/ruoyu/Point_360 && git status --short --branch | head -3
```

Expected: `## feat/segbackend-instances` and possibly modified files. **Do not modify anything here.** If the branch is no longer `feat/segbackend-instances`, stop and ask — someone changed the assumption this plan is built on.

- [ ] **Step 2: Create the worktree**

```bash
cd /home/ruoyu/Point_360
git worktree add ../Point_360-localization -b feat/localization-unit s3dis-eval
```

Expected: `Preparing worktree (new branch 'feat/localization-unit')` and `HEAD is now at 548728b`.

- [ ] **Step 3: Verify isolation**

```bash
cd /home/ruoyu/Point_360-localization && git status --short --branch
git worktree list
```

Expected: branch `feat/localization-unit`, clean tree. `git worktree list` shows two entries. The original `/home/ruoyu/Point_360` must still show `feat/segbackend-instances`.

- [ ] **Step 4: Commit nothing yet**

No commit — the worktree itself is not a repository change. Proceed to Task 2.

---

### Task 2: Pin the FGPL submodule to the fork

**Files:**
- Modify: `/home/ruoyu/Point_360-localization/.gitmodules`

**Interfaces:**
- Consumes: worktree from Task 1
- Produces: an initialized `DavidThesis/scan2measure-webframework` containing `src/pose_estimation/multiroom_pose_estimation.py` with the upright prior, pinned at `528061b`.

- [ ] **Step 1: Verify the pin target is reachable on the remote**

```bash
cd /home/ruoyu/scan2measure-webframework
git branch -r --contains 528061b
```

Expected: `origin/feat/upright-rotation-prior`. If empty, stop — the commit is not pushed and cannot be pinned.

- [ ] **Step 2: Re-point the submodule URL**

Edit `/home/ruoyu/Point_360-localization/.gitmodules` so it reads:

```
[submodule "DavidThesis/scan2measure-webframework"]
	path = DavidThesis/scan2measure-webframework
	url = https://github.com/ruoyu-yan/scan2measure-webframework.git
```

- [ ] **Step 3: Sync and initialize at the pinned commit**

```bash
cd /home/ruoyu/Point_360-localization
git submodule sync DavidThesis/scan2measure-webframework
git submodule update --init DavidThesis/scan2measure-webframework 2>&1 | tail -3 || true
cd DavidThesis/scan2measure-webframework
git fetch origin feat/upright-rotation-prior
git checkout 528061b
```

Expected: detached HEAD at `528061b`. The `|| true` on the update is deliberate — the recorded pin `acdc812` does not exist in the fork, so that step is expected to fail before the explicit fetch+checkout fixes it.

- [ ] **Step 4: Verify the prior is actually present**

```bash
cd /home/ruoyu/Point_360-localization/DavidThesis/scan2measure-webframework
grep -c "upright_prior" src/pose_estimation/multiroom_pose_estimation.py
grep -c "up_can = principal_3d" src/pose_estimation/pose_search.py
```

Expected: first ≥ 4, second = 1. The second is the canonical-frame fix from `b12ec86` — without it the prior selects the wrong candidates and locked poses go to 0/22.

- [ ] **Step 5: Record the new pin**

```bash
cd /home/ruoyu/Point_360-localization
git add .gitmodules DavidThesis/scan2measure-webframework
git commit -m "chore(submodule): pin FGPL to ruoyu-yan fork @528061b (upright prior)

The submodule declared ugurfeyzullah/scan2measure-webframework pinned at
acdc812 and was never initialized, so the directory was empty. That
upstream commit does not exist in the fork and contains neither the
upright rotation prior nor the PanoPin narrowing work.

Pinned to 528061b (HEAD of feat/upright-rotation-prior). The branch stays
unmerged by decision; do not delete or force-push it while this pin
references it."
```

---

### Task 3: PanoPin CLI entry point

**Files:**
- Create: `/home/ruoyu/PanoPin/src/panopin/cli.py`
- Test: `/home/ruoyu/PanoPin/tests/test_cli.py`

**Interfaces:**
- Consumes: `panopin.seed.seed_rooms`, `panopin.fgpl_export.export_alignment` (existing).
- Produces: a command `python -m panopin.cli seed --panos <json> --clouds <json> --metadata <path> --out <path> [--tau FLOAT] [--room-order R1,R2,...]` that writes a `demo6_alignment.json` and prints the admitted pano names, one per line, to stdout. Point_360's Stage 0 (Task 8) parses that stdout.

PanoPin currently has **no CLI anywhere under `src/`** — verified, zero `argparse`/`__main__` matches. Stage 0 must not call `experiments/fgpl_seed/*`, which is experiment code with hardcoded paths, so this wrapper is required.

- [ ] **Step 1: Write the failing test**

Create `/home/ruoyu/PanoPin/tests/test_cli.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return p


def test_seed_from_scores_writes_alignment(tmp_path):
    """cli.seed_from_cached builds an alignment json from a score matrix + poses."""
    from panopin import cli

    scores = {"panoA": {"roomX": 0.05, "roomY": 0.40},
              "panoB": {"roomX": 0.42, "roomY": 0.06}}
    poses = {"panoA": {"roomX": ([1.0, 2.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                       "roomY": ([9.0, 9.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])},
             "panoB": {"roomX": ([8.0, 8.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                       "roomY": ([3.0, 4.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])}}
    meta = _write(tmp_path / "metadata.json",
                  {"rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    out = tmp_path / "demo6_alignment.json"

    admitted = cli.seed_from_cached(scores, poses, ["roomX", "roomY"], meta, out, tau=0.10)

    assert sorted(admitted) == ["panoA", "panoB"], admitted
    data = json.loads(out.read_text())
    assert set(data) == {"metadata", "matches"}, sorted(data)
    assert data["metadata"]["pano_names"] == admitted
    by_name = {m["pano_name"]: m for m in data["matches"]}
    assert by_name["panoA"]["room_label"] == "roomX", by_name["panoA"]
    assert by_name["panoA"]["camera_position"] == [1.0, 2.0], by_name["panoA"]


def test_cli_module_is_runnable():
    """python -m panopin.cli --help exits 0 (the entry point Stage 0 depends on)."""
    r = subprocess.run([sys.executable, "-m", "panopin.cli", "--help"],
                       cwd=str(REPO / "src"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "seed" in r.stdout, r.stdout
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_cli.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'panopin.cli'`.

- [ ] **Step 3: Write the implementation**

Create `/home/ruoyu/PanoPin/src/panopin/cli.py`:

```python
"""Command-line entry point for PanoPin's deployable hand-off.

Stage 0 of the Scan2BIM pipeline calls this. It deliberately wraps only the
shipped, tested API (seed.seed_rooms / fgpl_export.export_alignment) and never
experiments/fgpl_seed/*, which carries hardcoded paths and cached-grid
assumptions.
"""
import argparse
import json
import sys
from pathlib import Path

from panopin import fgpl_export, seed


def seed_from_cached(score_matrix, poses, room_order, metadata_path, out_path, tau=0.10):
    """Build a demo6_alignment.json from an existing score matrix + poses.

    poses maps pano -> room -> (t, R). Returns the admitted pano names, which
    the caller MUST use as FGPL's cfg["pano_names"].
    """
    return fgpl_export.export_alignment(
        score_matrix, poses, list(room_order), str(metadata_path), str(out_path), tau=tau)


def seed_from_clouds(panos, candidate_clouds, metadata_path, out_path, tau=0.10):
    """GPU path: localize every pano against every candidate room, then export.

    panos maps pano_id -> pano image path; candidate_clouds maps room -> cloud path.
    Room order is taken from candidate_clouds' insertion order.

    Uses localize_and_score rather than seed_rooms: the FGPL hand-off is PER-PANO
    (D34 -- FGPL localizes each pano from its own seed), while seed_rooms returns
    one seed per ROOM. The full per-(pano, room) score matrix is what build_matches
    needs.
    """
    score_matrix, poses = seed.localize_and_score(
        panos, candidate_clouds, seed.load_cfg(sample_rate=30))
    return seed_from_cached(score_matrix, poses, list(candidate_clouds),
                            metadata_path, out_path, tau=tau)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="panopin.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("seed", help="build a demo6_alignment.json for FGPL")
    s.add_argument("--panos", required=True, type=Path,
                   help="JSON {pano_id: pano_image_path}")
    s.add_argument("--clouds", required=True, type=Path,
                   help="JSON {room: cloud_path}; key order sets room_idx")
    s.add_argument("--metadata", required=True, type=Path,
                   help="metadata.json holding rotation_matrix")
    s.add_argument("--out", required=True, type=Path,
                   help="output demo6_alignment.json path")
    s.add_argument("--tau", type=float, default=0.10,
                   help="per-pano admission threshold on the low-pct winner score")

    args = parser.parse_args(argv)

    panos = json.loads(args.panos.read_text())
    clouds = json.loads(args.clouds.read_text())
    admitted = seed_from_clouds(panos, clouds, args.metadata, args.out, tau=args.tau)
    for name in admitted:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

```bash
cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_cli.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Verify no regression in the existing suite**

```bash
cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/ -q
```

Expected: all previously-passing tests still pass (baseline was 14 in `test_fgpl_export*` plus 3 in `test_roundtrip`).

- [ ] **Step 6: Commit**

```bash
cd /home/ruoyu/PanoPin
git add src/panopin/cli.py tests/test_cli.py
git commit -m "feat(cli): add panopin.cli seed entry point for the Scan2BIM hand-off

Point_360's Stage 0 needs a stable public interface. It must not call
experiments/fgpl_seed/*, which has hardcoded paths and assumes cached
grids. This wraps only the shipped API: seed_rooms + export_alignment."
```

---

### Task 4: Add PanoPin as a submodule of Point_360

**Files:**
- Modify: `/home/ruoyu/Point_360-localization/.gitmodules`
- Create: `/home/ruoyu/Point_360-localization/external/PanoPin` (submodule)

**Interfaces:**
- Consumes: PanoPin's `cli.py` from Task 3 (must be pushed first).
- Produces: `external/PanoPin` at a pinned commit, importable by `conda run -n panopin-gpu`.

- [ ] **Step 1: Push PanoPin so a commit exists to pin**

```bash
cd /home/ruoyu/PanoPin
git push -u origin HEAD
git rev-parse HEAD
```

Record the SHA. If the current branch is `docs/scan2bim-localization-spec`, that is fine — pin to whatever commit carries `cli.py`.

- [ ] **Step 2: Add the submodule**

```bash
cd /home/ruoyu/Point_360-localization
git submodule add https://github.com/ruoyu-yan/PanoPin.git external/PanoPin
cd external/PanoPin && git checkout <SHA from Step 1> && cd ../..
```

- [ ] **Step 3: Verify the CLI is reachable through the submodule**

```bash
cd /home/ruoyu/Point_360-localization
conda run -n panopin-gpu python -c "import sys; sys.path.insert(0, 'external/PanoPin/src'); import panopin.cli; print('cli import OK')"
```

Expected: `cli import OK`.

- [ ] **Step 4: Commit**

```bash
cd /home/ruoyu/Point_360-localization
git add .gitmodules external/PanoPin
git commit -m "chore(submodule): add PanoPin at external/PanoPin

PanoPin is the coarse pano->room seeder that completes FGPL on repetitive
multi-room scenes. Pinning it here makes the FGPL + prior + PanoPin unit
reproducible from a fresh clone instead of depending on sibling
directories that happen to exist."
```

---

### Task 5: The pure pose adapter

**Files:**
- Create: `/home/ruoyu/Point_360-localization/localization/__init__.py`
- Create: `/home/ruoyu/Point_360-localization/localization/fgpl_pose_adapter.py`
- Test: `/home/ruoyu/Point_360-localization/test_fgpl_pose_adapter.py`

**Interfaces:**
- Consumes: FGPL's `local_filter_results.json`, whose entries always carry `n_tight` (int), `avg_dist` (float), `n_matched` (int), `t` (3-list), `R` (3x3), plus `n_translations`/`n_dense_lines`/`n_sparse_lines`/`n_intersections` when `use_local` is true.
- Produces:
  - `convert(results, axis_fix=None) -> dict` — `{pano: {"R": 3x3 list, "t": 3 list}}`
  - `quality(results) -> dict` — `{pano: {"n_tight":…, "avg_dist":…, "n_matched":…}}`
  - `convert_file(in_path, out_path, axis_fix=None, sidecar_path=None) -> dict`

`axis_fix` left-multiplies `R` exactly as `s3dis/pose_adapter.py:47` does (`R2 = axis_fix @ R`), so both pose producers share one convention knob. Task 6 determines its correct value; the default is identity.

- [ ] **Step 1: Write the failing test**

Create `/home/ruoyu/Point_360-localization/test_fgpl_pose_adapter.py`:

```python
import json
import tempfile
from pathlib import Path

import numpy as np

from localization import fgpl_pose_adapter as A

# A real, non-identity rotation. Using identity here would collapse the frames
# and prove nothing -- the upright-prior bug survived 5/5 passing tests for
# exactly that reason (principal_3d=eye(3) hid a canonical-vs-world mixup).
R_REAL = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
C = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]

SAMPLE = {
    "panoA": {"n_tight": 42, "avg_dist": 0.031, "n_matched": 88,
              "t": [1.5, -2.25, 1.6], "R": R_REAL,
              "n_translations": 900, "n_dense_lines": 120,
              "n_sparse_lines": 30, "n_intersections": 45},
    "panoB": {"n_tight": 3, "avg_dist": 0.29, "n_matched": 12,
              "t": [7.0, 0.5, 1.6], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
}


def test_convert_passes_r_and_t_through_by_default():
    got = A.convert(SAMPLE)
    assert set(got) == {"panoA", "panoB"}, sorted(got)
    assert got["panoA"]["R"] == R_REAL, got["panoA"]["R"]
    assert got["panoA"]["t"] == [1.5, -2.25, 1.6], got["panoA"]["t"]
    print("ok: default convert is a pass-through")


def test_axis_fix_left_multiplies_like_pose_adapter():
    got = A.convert(SAMPLE, axis_fix=C)
    expect = (np.asarray(C, float) @ np.asarray(R_REAL, float)).tolist()
    assert got["panoA"]["R"] == expect, got["panoA"]["R"]
    # t is a world-frame camera centre and must NOT be rotated by axis_fix.
    assert got["panoA"]["t"] == [1.5, -2.25, 1.6], got["panoA"]["t"]
    print("ok: axis_fix left-multiplies R, leaves t alone")


def test_convert_emits_only_contract_keys():
    got = A.convert(SAMPLE)
    for name, entry in got.items():
        assert sorted(entry) == ["R", "t"], (name, sorted(entry))
    print("ok: no extra per-entry keys leak into the pose contract")


def test_quality_preserves_fgpl_confidence_fields():
    q = A.quality(SAMPLE)
    assert q["panoA"]["n_tight"] == 42, q["panoA"]
    assert q["panoB"]["avg_dist"] == 0.29, q["panoB"]
    assert q["panoA"]["n_matched"] == 88, q["panoA"]
    print("ok: quality sidecar keeps n_tight/avg_dist/n_matched")


def test_malformed_entry_fails_loud():
    bad = {"panoA": {"t": [0, 0, 0]}}  # no R
    raised = False
    try:
        A.convert(bad)
    except (KeyError, ValueError):
        raised = True
    assert raised, "expected a loud failure for an entry missing R"
    print("ok: missing R raises")


def test_convert_file_roundtrip_writes_both_files():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "local_filter_results.json"
        src.write_text(json.dumps(SAMPLE))
        out = td / "poses.json"
        side = td / "poses.quality.json"
        A.convert_file(src, out, axis_fix=C, sidecar_path=side)
        poses = json.loads(out.read_text())
        assert sorted(poses) == ["panoA", "panoB"], sorted(poses)
        assert sorted(poses["panoA"]) == ["R", "t"], sorted(poses["panoA"])
        qual = json.loads(side.read_text())
        assert qual["panoA"]["n_tight"] == 42, qual["panoA"]
    print("ok: convert_file writes pose json + quality sidecar")


if __name__ == "__main__":
    test_convert_passes_r_and_t_through_by_default()
    test_axis_fix_left_multiplies_like_pose_adapter()
    test_convert_emits_only_contract_keys()
    test_quality_preserves_fgpl_confidence_fields()
    test_malformed_entry_fails_loud()
    test_convert_file_roundtrip_writes_both_files()
    print("ALL PASS")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /home/ruoyu/Point_360-localization && conda run -n point360 python test_fgpl_pose_adapter.py
```

Expected: FAIL — `ModuleNotFoundError: No module named 'localization'`.

- [ ] **Step 3: Write the implementation**

Create `/home/ruoyu/Point_360-localization/localization/__init__.py`:

```python
"""Stage 0: panorama camera pose estimation for the Scan2BIM pipeline."""
```

Create `/home/ruoyu/Point_360-localization/localization/fgpl_pose_adapter.py`:

```python
"""FGPL local_filter_results.json -> the Point_360 pose contract.

The contract consumed by Stages 2 and 4 is
    {pano_name: {"R": 3x3 world->camera, "t": camera centre in world}}
read by project_newdata_semantic_masks.load_pose (:74-77) and used as
    p_cam = (points - t) @ R.T

FGPL writes R and t under those same names, so the conversion is a pass-through
plus an optional axis rectification. `axis_fix` mirrors s3dis/pose_adapter.py:47
(R2 = axis_fix @ R) so both pose producers share one convention knob. Its
correct value is determined empirically, not assumed -- see
docs/superpowers/specs/2026-07-20-scan2bim-localization-unit-design.md §5.2.
"""
import json
from pathlib import Path

import numpy as np

QUALITY_KEYS = ("n_tight", "avg_dist", "n_matched")


def convert(results, axis_fix=None):
    """{pano: fgpl_entry} -> {pano: {"R", "t"}}.

    axis_fix (3x3, default identity) left-multiplies R. t is a world-frame
    camera centre and is never rotated by it.
    """
    fix = np.eye(3) if axis_fix is None else np.asarray(axis_fix, dtype=np.float64)
    out = {}
    for name, entry in results.items():
        if "R" not in entry or "t" not in entry:
            raise KeyError(f"{name}: FGPL entry missing 'R' or 't' (got {sorted(entry)})")
        R = np.asarray(entry["R"], dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"{name}: R must be 3x3, got {R.shape}")
        t = np.asarray(entry["t"], dtype=np.float64)
        if t.shape != (3,):
            raise ValueError(f"{name}: t must have 3 elements, got {t.shape}")
        out[name] = {"R": (fix @ R).tolist(), "t": t.tolist()}
    return out


def quality(results):
    """{pano: fgpl_entry} -> {pano: {n_tight, avg_dist, n_matched}}.

    Point_360 ignores FGPL's own confidence fields entirely. Preserving them in
    a sidecar keeps weak poses visible without touching the pose contract.
    """
    return {name: {k: entry[k] for k in QUALITY_KEYS if k in entry}
            for name, entry in results.items()}


def convert_file(in_path, out_path, axis_fix=None, sidecar_path=None):
    """Read FGPL results, write the pose contract (+ optional quality sidecar)."""
    results = json.loads(Path(in_path).read_text(encoding="utf-8"))
    poses = convert(results, axis_fix=axis_fix)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(poses, indent=2), encoding="utf-8")
    if sidecar_path is not None:
        sidecar_path = Path(sidecar_path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(quality(results), indent=2), encoding="utf-8")
    return poses
```

- [ ] **Step 4: Run the tests**

```bash
cd /home/ruoyu/Point_360-localization && conda run -n point360 python test_fgpl_pose_adapter.py
```

Expected: six `ok:` lines then `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
cd /home/ruoyu/Point_360-localization
git add localization/__init__.py localization/fgpl_pose_adapter.py test_fgpl_pose_adapter.py
git commit -m "feat(localization): pure FGPL -> pose-contract adapter

Converts FGPL's local_filter_results.json into the {pano: {R,t}} contract
Stages 2 and 4 consume, with an axis_fix knob mirroring
s3dis/pose_adapter.py so both pose producers share one convention.

Preserves FGPL's n_tight/avg_dist/n_matched into a sidecar; the pipeline
currently discards them and has no confidence gating anywhere.

Tests use a real non-identity rotation deliberately: the upright-prior bug
passed 5/5 tests built on principal_3d=eye(3), which collapsed the
canonical and world frames and proved nothing."
```

---

### Task 6: Determine `axis_fix`, and build the Gate 2 reprojection check

**Files:**
- Create: `/home/ruoyu/Point_360-localization/localization/determine_axis_fix.py`
- Create: `/home/ruoyu/Point_360-localization/localization/gate_reprojection.py`
- Create: `/home/ruoyu/Point_360-localization/localization/AXIS_FIX_RESULT.md`

**Interfaces:**
- Consumes: `fgpl_pose_adapter.convert` (Task 5); cached D36 poses at `/home/ruoyu/PanoPin/experiments/fgpl_seed/work/poses/manhattan_upright/local_filter_results.json` (22 panos, verified present); S3DIS GT poses via `s3dis/pose_adapter.load_s3dis_pose`; `s3dis/validate_projection.py`'s `sample_gt_points`, `roundtrip_err`, `_exr_for`.
- Produces: the correct `axis_fix` value in `AXIS_FIX_RESULT.md` (used as the default in Task 8's orchestrator), and `gate_reprojection.check(pose_json, pano_root) -> dict` implementing Gate 2.

**Why Gate 2 needs its own script.** `s3dis/validate_projection.py` cannot be reused directly: its CLI is `--pano-root / --room / --n-samples / --out-dir` and it reads GT poses from `pano_root/pose` itself (`validate_projection.py:132-138`). **There is no `--pose-json` flag**, so it cannot score externally-supplied poses. Do not modify it — it is a proven gate with a recorded result. Instead reuse its internals.

**Why this task exists.** Two facts are in tension. Point_360 consumes FGPL's `R`/`t` **raw** for TMB (`smoke_test/run_smoke.py:10`, no `axis_fix`, and it works). But S3DIS GT needs `axis_fix = C`. Meanwhile PanoPin's D25 notes record FGPL's convention as `Rp.T @ C = camera→world`, which implies world→camera `= C.T @ Rp`, i.e. **not** a pass-through. These cannot all be right. Guessing here produces masks silently landing on wrong 3D points (`roadmap.md:464-467`, risk hotspot 1.3). This task settles it offline, from cached data, before any GPU run — exactly as `s3dis/validate_projection.py:138-157` originally brute-forced the S3DIS rectification.

- [ ] **Step 1: Write the determination script**

Create `/home/ruoyu/Point_360-localization/localization/determine_axis_fix.py`:

```python
"""Determine the axis_fix that maps FGPL's R into the Point_360 pose contract.

Runs offline against cached D36 poses -- no GPU, no FGPL run. Compares each
candidate rectification against S3DIS ground truth and reports the rotation
error for each. The winner should be near 0 deg for panos FGPL localized
correctly; every other candidate should be far off.

Usage:
    conda run -n point360 python -m localization.determine_axis_fix \
        --fgpl <local_filter_results.json> --pose-dir <s3dis pose dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localization import fgpl_pose_adapter  # noqa: E402

C = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

CANDIDATES = {
    "identity": np.eye(3),
    "C": C,
    "C.T": C.T,
}


def rotation_error_deg(R_a, R_b):
    """Geodesic angle between two rotations, in degrees."""
    rel = np.asarray(R_a, float) @ np.asarray(R_b, float).T
    cos = (np.trace(rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def load_gt(pose_dir):
    """{pano_id: R_world_to_camera} from S3DIS pose JSONs, with axis_fix=C applied.

    C is the rectification validate_projection.py already proved to 0.001 px,
    so GT here is expressed in the pipeline's own camera convention.
    """
    gt = {}
    for p in sorted(Path(pose_dir).glob("*_pose.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        rt = np.asarray(data["camera_rt_matrix"], dtype=np.float64)
        stem = p.stem
        suffix = "_frame_equirectangular_domain_pose"
        if not stem.endswith(suffix):
            continue
        gt[stem[: -len(suffix)]] = C @ rt[:, :3]
    return gt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fgpl", required=True, type=Path)
    ap.add_argument("--pose-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    results = json.loads(args.fgpl.read_text(encoding="utf-8"))
    gt = load_gt(args.pose_dir)

    shared = [n for n in results if n in gt]
    if not shared:
        raise SystemExit(f"no overlap between {len(results)} FGPL panos and {len(gt)} GT poses")
    print(f"comparing {len(shared)} panos\n")

    for label, fix in CANDIDATES.items():
        converted = fgpl_pose_adapter.convert(results, axis_fix=fix)
        errs = sorted(rotation_error_deg(converted[n]["R"], gt[n]) for n in shared)
        median = errs[len(errs) // 2]
        n_locked = sum(1 for e in errs if e < 5.0)
        print(f"{label:>10}: median {median:7.2f} deg   locked(<5deg) {n_locked}/{len(errs)}")
    print("\nThe correct axis_fix maximises locked and minimises median.")
    print("If NO candidate locks any pano, the mismatch is not a fixed "
          "rectification -- stop and investigate before wiring Stage 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it against the cached D36 poses**

```bash
cd /home/ruoyu/Point_360-localization
conda run -n point360 python -m localization.determine_axis_fix \
  --fgpl /home/ruoyu/PanoPin/experiments/fgpl_seed/work/poses/manhattan_upright/local_filter_results.json \
  --pose-dir "/mnt/d/Python Workplace/Master_Thesis/scan2measure-webframework/data/area_3/pano/pose"
```

That pose root is `validate_projection.py:176`'s default with `/pose` appended. **Verify it exists first** (`ls` it); if not, take the recorded root from `python -c "import json;print(json.load(open('/home/ruoyu/PanoPin/config/datasets.json'))['s3dis']['Area_3'])"`.

Expected: three lines, one per candidate. Exactly one should show a low median and a high locked count.

**Decision rule.** D36 reported 15/22 rotation-locked with the prior on, so expect the winning candidate to lock roughly that many, and the losers to lock ~0. If two candidates tie, or none locks anything, **stop and report** — that means the relationship is not a fixed rectification and Stage 0 must not be wired until it is understood.

- [ ] **Step 3: Record the result**

Create `/home/ruoyu/Point_360-localization/localization/AXIS_FIX_RESULT.md` with the verbatim command, its full output, and a one-line conclusion naming the winner. Example shape (fill in the real numbers — do not copy these):

```markdown
# axis_fix determination

Command:

    conda run -n point360 python -m localization.determine_axis_fix --fgpl ... --pose-dir ...

Output:

    comparing 22 panos

      identity: median   X.XX deg   locked(<5deg) N/22
             C: median   X.XX deg   locked(<5deg) N/22
           C.T: median   X.XX deg   locked(<5deg) N/22

Conclusion: axis_fix = <WINNER>. Used as the default in run_localization.py.
Source data: cached D36 manhattan_upright arm (22 panos), S3DIS Area_3 GT.
```

- [ ] **Step 4: Write the Gate 2 reprojection check**

Create `/home/ruoyu/Point_360-localization/localization/gate_reprojection.py`:

```python
"""Gate 2: reproject with ESTIMATED poses and compare against S3DIS ground truth.

Reuses s3dis/validate_projection.py's proven machinery (it is not modified --
its CLI reads GT poses itself and has no --pose-json flag).

WHAT THIS GATE IS FOR. It detects convention and frame BLUNDERS -- a wrong
axis_fix, a transposed rotation, a swapped axis -- which throw reprojection off
by hundreds of pixels. It is NOT an accuracy gate. A pose that is a perfectly
good 8 cm and 1 deg off still reprojects tens of pixels away, so the GT
thresholds in validate_projection.py (median < 2 px, p95 < 5 px) are
unreachable here by construction and must not be reused.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "s3dis"))

import validate_projection as VP  # noqa: E402

# A wrong convention lands the projection essentially anywhere (~90-180 deg of
# rotation error = hundreds of px on a 2k-wide equirect). A correct-but-coarse
# pose stays well inside this. Blunder detector, not an accuracy metric.
BLUNDER_MEDIAN_PX = 100.0


def check(pose_json, pano_root, n=5000, seed=0):
    """Score every pano in `pose_json` and flag likely convention blunders."""
    poses = json.loads(Path(pose_json).read_text(encoding="utf-8"))
    pano_root = Path(pano_root)
    per_pano, medians = {}, []
    for pid, entry in poses.items():
        R = np.asarray(entry["R"], dtype=np.float64)
        t = np.asarray(entry["t"], dtype=np.float64)
        try:
            P, u0, v0, W, H = VP.sample_gt_points(VP._exr_for(pano_root, pid), n=n, seed=seed)
        except FileNotFoundError:
            per_pano[pid] = {"skipped": "no GT exr"}
            continue
        err = VP.roundtrip_err(P, u0, v0, R, t, W, H)
        med = float(np.median(err))
        per_pano[pid] = {"median_px": med, "p95_px": float(np.percentile(err, 95))}
        medians.append(med)
    if not medians:
        raise SystemExit("[Gate 2] no pano had ground truth -- cannot run this gate.")
    agg = float(np.median(medians))
    passed = agg < BLUNDER_MEDIAN_PX
    return {"panos": per_pano, "agg_median_px": agg,
            "threshold_px": BLUNDER_MEDIAN_PX, "passed": passed}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-json", required=True, type=Path)
    ap.add_argument("--pano-root", required=True, type=Path,
                    help="S3DIS area pano root (contains pose/ and the GT exr files)")
    ap.add_argument("--n-samples", type=int, default=5000)
    args = ap.parse_args(argv)
    res = check(args.pose_json, args.pano_root, n=args.n_samples)
    print(json.dumps(res, indent=2))
    print("GATE 2:", "PASS" if res["passed"] else "FAIL")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Verify Gate 2 fires on a deliberately wrong convention**

```bash
cd /home/ruoyu/Point_360-localization
conda run -n point360 python -c "
import json, sys, numpy as np
sys.path.insert(0, '.')
from localization import fgpl_pose_adapter as A
src = '/home/ruoyu/PanoPin/experiments/fgpl_seed/work/poses/manhattan_upright/local_filter_results.json'
res = json.loads(open(src).read())
C = [[0,0,1],[-1,0,0],[0,-1,0]]
A.convert_file(src, '/tmp/poses_right.json', axis_fix=<WINNER from Step 3>)
A.convert_file(src, '/tmp/poses_wrong.json', axis_fix=np.array(C) @ np.array(C))
print('wrote both')
"
conda run -n point360 python -m localization.gate_reprojection \
  --pose-json /tmp/poses_right.json --pano-root <PANO_ROOT> | tail -2
conda run -n point360 python -m localization.gate_reprojection \
  --pose-json /tmp/poses_wrong.json --pano-root <PANO_ROOT> | tail -2
```

Expected: the first prints `GATE 2: PASS`, the second `GATE 2: FAIL`. A gate that passes both is not a gate — if that happens, raise the sensitivity before trusting it.

`<PANO_ROOT>` is the S3DIS area pano root. `validate_projection.py:176` defaults it to
`/mnt/d/Python Workplace/Master_Thesis/scan2measure-webframework/data/area_3/pano` — verify that path exists on this machine before running, and use whatever `config/datasets.json` records if it does not.

- [ ] **Step 6: Commit**

```bash
cd /home/ruoyu/Point_360-localization
git add localization/determine_axis_fix.py localization/gate_reprojection.py localization/AXIS_FIX_RESULT.md
git commit -m "feat(localization): determine axis_fix empirically from cached D36 poses

Point_360 consumes FGPL's R raw for TMB, but S3DIS GT needs axis_fix=C,
and PanoPin's D25 notes imply world->camera = C.T @ Rp. Those cannot all
be right, and guessing makes masks land on wrong 3D points with no error
(roadmap.md:464-467).

Settles it offline against 22 cached D36 poses before any GPU run, the
same way validate_projection.py originally brute-forced the S3DIS
rectification."
```

---

### Task 7: Preflight and Gate 1

**Files:**
- Create: `/home/ruoyu/Point_360-localization/localization/preflight.py`
- Test: `/home/ruoyu/Point_360-localization/test_localization_preflight.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure filesystem/JSON inspection).
- Produces:
  - `check_submodule(path) -> None` — raises `SystemExit` if the directory is missing or empty.
  - `check_prior_present(fgpl_root) -> None` — raises `SystemExit` unless both the prior and its canonical-frame fix are in the source.
  - `check_prior_enabled(cfg) -> None` — raises `SystemExit` unless `cfg["upright_prior"]` is truthy.
  - `run_all(fgpl_root, panopin_root, cfg) -> None`

Gate 1 checks presence **and** enablement because either alone fails silently: the prior ships default-off (`multiroom_pose_estimation.py:63`), so a correct pin with a default config yields 24 rotation candidates instead of ~4, poses at ~0.96 m instead of 0.084 m, and a perfectly valid-looking BIM.

- [ ] **Step 1: Write the failing test**

Create `/home/ruoyu/Point_360-localization/test_localization_preflight.py`:

```python
import tempfile
from pathlib import Path

from localization import preflight


def _fake_fgpl(root, with_prior=True, with_fix=True):
    """Build a minimal fake FGPL tree with the two files preflight inspects."""
    pe = root / "src" / "pose_estimation"
    pe.mkdir(parents=True)
    mr = "upright_prior = cfg.get('upright_prior', UPRIGHT_PRIOR)\n" if with_prior else "pass\n"
    (pe / "multiroom_pose_estimation.py").write_text(mr)
    ps = "up_can = principal_3d @ (uw / uw.norm())\n" if with_fix else "pass\n"
    (pe / "pose_search.py").write_text(ps)
    return root


def test_missing_submodule_raises():
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "scan2measure-webframework"
        empty.mkdir()
        raised = False
        try:
            preflight.check_submodule(empty)
        except SystemExit:
            raised = True
        assert raised, "expected SystemExit for an empty (uninitialized) submodule"
    print("ok: uninitialized submodule -> SystemExit")


def test_prior_absent_raises():
    with tempfile.TemporaryDirectory() as td:
        root = _fake_fgpl(Path(td), with_prior=False)
        raised = False
        try:
            preflight.check_prior_present(root)
        except SystemExit:
            raised = True
        assert raised, "expected SystemExit when the prior is not in the source"
    print("ok: FGPL pin without the prior -> SystemExit")


def test_canonical_frame_fix_absent_raises():
    with tempfile.TemporaryDirectory() as td:
        root = _fake_fgpl(Path(td), with_prior=True, with_fix=False)
        raised = False
        try:
            preflight.check_prior_present(root)
        except SystemExit:
            raised = True
        assert raised, "expected SystemExit when b12ec86's canonical-frame fix is missing"
    print("ok: prior without the canonical-frame fix -> SystemExit")


def test_prior_present_but_disabled_in_config_raises():
    raised = False
    try:
        preflight.check_prior_enabled({"pano_names": ["a"]})
    except SystemExit:
        raised = True
    assert raised, "expected SystemExit when the config omits upright_prior"
    print("ok: prior present but config-disabled -> SystemExit")


def test_clean_state_passes():
    with tempfile.TemporaryDirectory() as td:
        root = _fake_fgpl(Path(td))
        preflight.check_submodule(root)
        preflight.check_prior_present(root)
        preflight.check_prior_enabled({"upright_prior": True})
    print("ok: clean state passes all Gate 1 checks")


if __name__ == "__main__":
    test_missing_submodule_raises()
    test_prior_absent_raises()
    test_canonical_frame_fix_absent_raises()
    test_prior_present_but_disabled_in_config_raises()
    test_clean_state_passes()
    print("ALL PASS")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /home/ruoyu/Point_360-localization && conda run -n point360 python test_localization_preflight.py
```

Expected: FAIL — `ImportError: cannot import name 'preflight'`.

- [ ] **Step 3: Write the implementation**

Create `/home/ruoyu/Point_360-localization/localization/preflight.py`:

```python
"""Stage 0 preflight: fail closed and loud before spending GPU hours.

Gate 1 (the upright prior) checks presence AND enablement. Either alone fails
silently: the prior ships opt-in and default-off, so a correct submodule pin
combined with a default config yields FGPL enumerating all 24 rotation
candidates instead of ~4 -- poses land at ~0.96 m or flipped instead of
0.084 m, and the pipeline still emits a valid-looking BIM.
"""
import sys
from pathlib import Path

# The prior's config key (multiroom_pose_estimation.py:165) and the marker for
# b12ec86's canonical-frame fix, without which the prior selects exactly the
# wrong candidates (locked poses went 10/22 -> 0/22 when this was wrong).
PRIOR_KEY = "upright_prior"
CANONICAL_FIX_MARKER = "up_can = principal_3d"


def _die(message):
    raise SystemExit(f"[Stage 0 preflight] {message}")


def check_submodule(path):
    """Fail unless `path` exists and is non-empty."""
    path = Path(path)
    if not path.is_dir():
        _die(f"{path} does not exist. Run: git submodule update --init")
    if not any(path.iterdir()):
        _die(f"{path} is empty (submodule not initialized). "
             f"Run: git submodule update --init {path}")


def check_prior_present(fgpl_root):
    """Fail unless the pinned FGPL source carries the prior AND its frame fix."""
    fgpl_root = Path(fgpl_root)
    orchestrator = fgpl_root / "src" / "pose_estimation" / "multiroom_pose_estimation.py"
    search = fgpl_root / "src" / "pose_estimation" / "pose_search.py"
    for f in (orchestrator, search):
        if not f.is_file():
            _die(f"{f} not found -- is the FGPL submodule pinned correctly?")
    if PRIOR_KEY not in orchestrator.read_text(encoding="utf-8"):
        _die(f"the pinned FGPL commit has no '{PRIOR_KEY}' support. "
             f"Expected the pin at 528061b (feat/upright-rotation-prior).")
    if CANONICAL_FIX_MARKER not in search.read_text(encoding="utf-8"):
        _die("the pinned FGPL commit has the prior but NOT the canonical-frame "
             "fix (b12ec86). Without it the prior keeps the wrong candidates. "
             "Expected the pin at 528061b.")


def check_prior_enabled(cfg):
    """Fail unless the run config actually turns the prior on."""
    if not cfg.get(PRIOR_KEY):
        _die(f"config does not set '{PRIOR_KEY}': true. The prior is opt-in and "
             f"defaults to OFF, so this run would silently use 24 rotation "
             f"candidates and degrade poses ~11x without erroring.")


def run_all(fgpl_root, panopin_root, cfg):
    """Every Gate 1 check, in order."""
    check_submodule(fgpl_root)
    check_submodule(panopin_root)
    check_prior_present(fgpl_root)
    check_prior_enabled(cfg)
    print("[Stage 0 preflight] OK: submodules initialized, prior present and enabled.")
```

- [ ] **Step 4: Run the tests**

```bash
cd /home/ruoyu/Point_360-localization && conda run -n point360 python test_localization_preflight.py
```

Expected: five `ok:` lines then `ALL PASS`.

- [ ] **Step 5: Verify Gate 1 fires against the real pinned submodule**

```bash
cd /home/ruoyu/Point_360-localization
conda run -n point360 python -c "
from localization import preflight
preflight.check_submodule('DavidThesis/scan2measure-webframework')
preflight.check_prior_present('DavidThesis/scan2measure-webframework')
print('real pin passes Gate 1')
try:
    preflight.check_prior_enabled({})
except SystemExit as e:
    print('and a default config is correctly rejected:'); print(e)
"
```

Expected: `real pin passes Gate 1`, then the rejection message. This is acceptance criterion 2.

- [ ] **Step 6: Commit**

```bash
cd /home/ruoyu/Point_360-localization
git add localization/preflight.py test_localization_preflight.py
git commit -m "feat(localization): Gate 1 preflight -- prior present AND enabled

Checks both, because either alone fails silently. The prior is opt-in and
default-off, so a correct pin with a default config gives 24 rotation
candidates instead of ~4: poses at ~0.96m instead of 0.084m, and a
valid-looking BIM. Also rejects an uninitialized submodule, which is the
state the repo was actually found in."
```

---

### Task 8: The Stage 0 orchestrator

**Files:**
- Create: `/home/ruoyu/Point_360-localization/run_localization.py`

**Interfaces:**
- Consumes: `localization.preflight.run_all`, `localization.fgpl_pose_adapter.convert_file`, `s3dis.pose_adapter.convert_room`, PanoPin's `python -m panopin.cli seed`, FGPL's `multiroom_pose_estimation.py --config`.
- Produces: a CLI matching the other stage scripts, writing the pose contract to `--output`.

- [ ] **Step 1: Write the orchestrator**

Create `/home/ruoyu/Point_360-localization/run_localization.py`:

```python
"""Stage 0: panorama camera pose estimation.

Produces the {pano: {R, t}} contract consumed by Stages 2 and 4. Two sources:

  --pose-source gt       S3DIS ground truth via s3dis/pose_adapter (the default
                         for paper evaluation; pose accuracy is not a metric of
                         this paper).
  --pose-source panopin  PanoPin-seeded, upright-prior-enabled FGPL.

The three conda envs cannot share a process (point360 py3.12 / panopin-gpu
py3.8+cu118 / scan_env py3.8+cu116), so cross-env work goes through conda run,
following run_scan2measure_3d_lines.py:186.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from localization import fgpl_pose_adapter, preflight  # noqa: E402

DEFAULT_FGPL_ROOT = ROOT / "DavidThesis" / "scan2measure-webframework"
DEFAULT_PANOPIN_ROOT = ROOT / "external" / "PanoPin"
PANOPIN_ENV = "panopin-gpu"
ESTIMATOR_ENV = "panopin-gpu"  # cu118 is Ada-native; scan_env's cu116 falls back to CPU

# Determined empirically in Task 6; see localization/AXIS_FIX_RESULT.md.
# Replace this with the winner recorded there before the acceptance runs.
AXIS_FIX = None  # None = identity


def _run(cmd, cwd, check=True):
    print("RUN:", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=check)


def build_seed(panopin_root, panos_json, clouds_json, metadata, out_alignment, tau):
    """Run PanoPin's CLI; return the admitted pano names it printed."""
    cmd = ["conda", "run", "--no-capture-output", "-n", PANOPIN_ENV,
           "python", "-m", "panopin.cli", "seed",
           "--panos", panos_json, "--clouds", clouds_json,
           "--metadata", metadata, "--out", out_alignment, "--tau", tau]
    print("RUN:", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(Path(panopin_root) / "src"),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"PanoPin seeding failed:\n{r.stderr}")
    admitted = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    if not admitted:
        raise SystemExit("PanoPin admitted zero panos -- nothing to localize.")
    print(f"[Stage 0] PanoPin admitted {len(admitted)} panos")
    return admitted


def write_fgpl_config(cfg_path, scene, admitted, line_map, alignment, metadata,
                      density_png, cloud_ply, features_dir, pano_dir, output_dir):
    """Write the estimator config, with the upright prior ON."""
    cfg = {
        "point_cloud_name": scene,
        "pano_names": admitted,
        "use_local_filtering": True,
        "pkl_3d_path": str(line_map),
        "alignment_path": str(alignment),
        "metadata_path": str(metadata),
        "density_image_path": str(density_png),
        "point_cloud_path": str(cloud_ply),
        "features_2d_dir": str(features_dir),
        "pano_dir": str(pano_dir),
        "output_dir": str(output_dir),
        # S3DIS raw frame is Z-up, so gravity is [0,0,1] in the cloud frame.
        "upright_prior": True,
        "up_world": [0.0, 0.0, 1.0],
        "max_tilt_deg": 10.0,
    }
    cfg_path = Path(cfg_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def run_estimator(fgpl_root, cfg_path):
    estimator = Path(fgpl_root) / "src" / "pose_estimation" / "multiroom_pose_estimation.py"
    # check=False mirrors experiments/fgpl_seed/run_arm.py:20 -- the visualization
    # stage can crash after poses are already written.
    _run(["conda", "run", "--no-capture-output", "-n", ESTIMATOR_ENV,
          "python", str(estimator), "--config", str(cfg_path)],
         cwd=fgpl_root, check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-source", choices=["gt", "panopin"], default="panopin")
    ap.add_argument("--output", required=True, type=Path,
                    help="pose contract JSON consumed by Stages 2 and 4")
    ap.add_argument("--sidecar", type=Path, default=None,
                    help="optional quality sidecar (n_tight/avg_dist/n_matched)")
    # gt arm
    ap.add_argument("--s3dis-pose-dir", type=Path, default=None)
    ap.add_argument("--room", default=None)
    # panopin arm
    ap.add_argument("--scene", default=None)
    ap.add_argument("--panos-json", type=Path, default=None)
    ap.add_argument("--clouds-json", type=Path, default=None)
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--line-map", type=Path, default=None)
    ap.add_argument("--cloud-ply", type=Path, default=None)
    ap.add_argument("--density-png", type=Path, default=None)
    ap.add_argument("--features-dir", type=Path, default=None)
    ap.add_argument("--pano-dir", type=Path, default=None)
    ap.add_argument("--work-dir", type=Path, default=ROOT / "data" / "localization")
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--fgpl-root", type=Path, default=DEFAULT_FGPL_ROOT)
    ap.add_argument("--panopin-root", type=Path, default=DEFAULT_PANOPIN_ROOT)
    args = ap.parse_args(argv)

    if args.pose_source == "gt":
        sys.path.insert(0, str(ROOT / "s3dis"))
        from pose_adapter import convert_room
        if not args.s3dis_pose_dir or not args.room:
            raise SystemExit("--pose-source gt requires --s3dis-pose-dir and --room")
        C = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        convert_room(args.s3dis_pose_dir, args.room, args.output, axis_fix=np.array(C))
        print(f"[Stage 0] wrote GT poses to {args.output}")
        return 0

    required = ["scene", "panos_json", "clouds_json", "metadata", "line_map",
                "cloud_ply", "density_png", "features_dir", "pano_dir"]
    missing = [f"--{r.replace('_', '-')}" for r in required if getattr(args, r) is None]
    if missing:
        raise SystemExit(f"--pose-source panopin requires: {', '.join(missing)}")

    work = Path(args.work_dir)
    alignment = work / args.scene / "demo6_alignment.json"
    cfg_path = work / args.scene / "pose_config.json"
    est_out = work / args.scene / "poses"

    cfg = write_fgpl_config(cfg_path, args.scene, [], args.line_map, alignment,
                            args.metadata, args.density_png, args.cloud_ply,
                            args.features_dir, args.pano_dir, est_out)
    preflight.run_all(args.fgpl_root, args.panopin_root, cfg)

    admitted = build_seed(args.panopin_root, args.panos_json, args.clouds_json,
                          args.metadata, alignment, args.tau)
    write_fgpl_config(cfg_path, args.scene, admitted, args.line_map, alignment,
                      args.metadata, args.density_png, args.cloud_ply,
                      args.features_dir, args.pano_dir, est_out)

    run_estimator(args.fgpl_root, cfg_path)

    results = est_out / "local_filter_results.json"
    if not results.is_file():
        raise SystemExit(f"FGPL produced no {results} -- the estimator failed.")
    fgpl_pose_adapter.convert_file(results, args.output, axis_fix=AXIS_FIX,
                                   sidecar_path=args.sidecar)
    print(f"[Stage 0] wrote {args.output} ({len(admitted)} panos)")
    print("[Stage 0] NEXT: run the Gate 2 reprojection check before Stages 2/4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify it runs and the CLI is well-formed**

```bash
cd /home/ruoyu/Point_360-localization && conda run -n point360 python run_localization.py --help
```

Expected: exit 0, help text listing `--pose-source`, `--output`, and the panopin arm flags.

- [ ] **Step 3: Verify Gate 1 fires end-to-end**

```bash
cd /home/ruoyu/Point_360-localization
conda run -n point360 python run_localization.py --pose-source panopin --output /tmp/x.json 2>&1 | tail -3
```

Expected: a `SystemExit` naming the missing required flags. This confirms the CLI fails closed rather than proceeding with partial input.

- [ ] **Step 4: Set `AXIS_FIX` from Task 6**

Edit `run_localization.py` and replace `AXIS_FIX = None` with the winner recorded in `localization/AXIS_FIX_RESULT.md`. If the winner was `identity`, leave it as `None` and update the comment to say so explicitly.

- [ ] **Step 5: Commit**

```bash
cd /home/ruoyu/Point_360-localization
git add run_localization.py
git commit -m "feat(stage0): run_localization.py -- PanoPin + prior-enabled FGPL

Adds Stage 0 as a peer of the four existing stage scripts. --pose-source
gt|panopin is the swap seam: both arms emit the identical pose contract,
so Stages 2 and 4 cannot tell them apart.

Preflight runs before the GPU work, and the estimator config sets
upright_prior explicitly rather than relying on a default that is off."
```

---

### Task 9: Acceptance runs

**Files:**
- Create: `/home/ruoyu/Point_360-localization/localization/ACCEPTANCE.md`

**Interfaces:**
- Consumes: everything above.
- Produces: recorded evidence for the five acceptance criteria in spec §7.

This task is long-running — the estimator takes roughly 330-390 s per pano, so 22 panos is several hours. Run it deliberately, not in a loop.

- [ ] **Step 1: Build the fresh artifacts from the raw room clouds**

Per spec §7.1 the acceptance run rebuilds rather than reusing cached artifacts.

**Do NOT run the build scripts as modules.** Their `__main__` blocks call `subset.build_subset()`, which uses the module-level `subset.ROOMS` — the 6-room *ablation* subset, not the Manhattan pool. Running `python -m experiments.fgpl_seed.build_ply` would silently build the wrong scene. The Manhattan runs call the build *functions* with Manhattan rows instead (see `manhattan_roundtrip.py`).

Write and run a driver, `/home/ruoyu/PanoPin/experiments/fgpl_seed/manhattan_build.py`:

```python
"""Rebuild the Manhattan scene's FGPL inputs from the raw room clouds.

Deliberately NOT the build_* __main__ blocks: those call subset.build_subset(),
which resolves to the 6-room ablation subset, not the 5-room Manhattan pool.
"""
from experiments.fgpl_seed import build_features, build_linemap, build_ply, manhattan

if __name__ == "__main__":
    rows = manhattan.build_pool()
    rooms = sorted({r["room"] for r in rows})
    print("rooms:", rooms)
    assert rooms == sorted(["office_5", "hallway_1", "lounge_1",
                            "conferenceRoom_1", "WC_1"]), rooms
    print(f"panos: {len(rows)}")
    assert len(rows) == 22, len(rows)

    ply = build_ply.build_combined_ply(rows, scene=manhattan.SCENE)
    print("ply:", ply)
    lm = build_linemap.build_linemap(ply, scene=manhattan.SCENE,
                                     out_subdir=manhattan.LINEMAP_SUBDIR)
    print("line map:", lm)
    feats = build_features.stage_and_build(rows)
    print(f"features for {len(feats)} panos")
```

```bash
cd /home/ruoyu/PanoPin && conda run -n panopin python -m experiments.fgpl_seed.manhattan_build
```

Expected: `rooms:` listing exactly the five, `panos: 22`, then the three artifact paths. The two asserts are the point — they fail loud if the pool ever drifts.

`manhattan.build_pool()` (`manhattan.py:51`) returns Manhattan-only rows through largeval's validated in-frame pano filter (D17). The matching `manhattan.pool_clouds()` (`manhattan.py:56`) gives `{room: per-room S3DIS cloud .txt}` — that is the `--clouds-json` content for Step 3.

- [ ] **Step 2: Acceptance 4 first — TMB smoke test (cheap, catches regressions early)**

```bash
cd /home/ruoyu/Point_360-localization && conda run -n point360 python smoke_test/run_smoke.py
```

Expected: same result as on `s3dis-eval`. Stage 0 is purely additive, so any change here is a real regression — stop and investigate.

- [ ] **Step 3: Acceptance 1 — run Stage 0 and compare poses to GT**

Run `run_localization.py --pose-source panopin` with the artifacts from Step 1, then re-use Task 6's comparison script against the output:

```bash
cd /home/ruoyu/Point_360-localization
conda run -n point360 python -m localization.determine_axis_fix \
  --fgpl data/localization/<scene>/poses/local_filter_results.json \
  --pose-dir <s3dis pose dir>
```

Expected: the chosen `axis_fix` locks roughly 15/22 panos, matching D36. **If it disagrees**, do not guess — re-run Stage 0 against the cached D33/D36 artifacts instead of the fresh build. Agreement there isolates the fault to the rebuild; disagreement isolates it to the wiring.

- [ ] **Step 4: Acceptance 3 — Gate 2 reprojection**

```bash
cd /home/ruoyu/Point_360-localization
conda run -n point360 python -m localization.gate_reprojection \
  --pose-json <Stage 0 output> --pano-root <PANO_ROOT> | tail -3
```

Expected: `GATE 2: PASS`. Note this uses `localization/gate_reprojection.py` from Task 6, **not** `s3dis/validate_projection.py` — that script reads GT poses itself and has no `--pose-json` flag.

Remember what this gate does and does not prove: it catches convention blunders (hundreds of px), not accuracy. Passing it does not mean the poses are good; criterion 1 is what measures that.

- [ ] **Step 5: Acceptance 5 — Stages 2 through 4 on conferenceRoom_1**

Follow the six-command recipe in `docs/point360_env.md:41-77`, substituting Stage 0's output for the GT pose JSON in the `--pose-json` argument of Stages 2 and 4. Compare the resulting `clean_object_boundaries_for_dynamo.json` element counts against the GT-pose baseline: wall 13, door 16, floor 15, ceiling 15, beam 24, window 8.

Hard requirements: schema intact, no crash. Counts and geometry deltas are **reported, not thresholded**.

- [ ] **Step 6: Record everything**

Create `localization/ACCEPTANCE.md` with, for each of the five criteria: the verbatim command, its output, and a one-line verdict. Where a criterion is partially met, say so plainly rather than rounding up.

- [ ] **Step 7: Commit**

```bash
cd /home/ruoyu/Point_360-localization
git add localization/ACCEPTANCE.md
git commit -m "docs(localization): record Stage 0 acceptance results"
```

- [ ] **Step 8: Merge back**

Only after the other session's branch has landed or is quiescent:

```bash
cd /home/ruoyu/Point_360-localization && git status --short   # must be clean
cd /home/ruoyu/Point_360
git checkout s3dis-eval
git merge feat/localization-unit
```

Expected: a clean merge. File surfaces are disjoint — this branch adds `run_localization.py`, `localization/`, and `.gitmodules` entries; the other touches `seg_backends.py`, `seg_types.py`, `index.html`, `vendor/`, `smoke_test/`. **Check with the user before checking out `s3dis-eval` in the shared worktree**, since that changes what the other session sees.

- [ ] **Step 9: Remove the worktree**

```bash
cd /home/ruoyu/Point_360 && git worktree remove ../Point_360-localization
```

---

## Notes for the implementer

**Things that will bite you, all verified:**

1. **`multiroom_pose_estimation.py` has no argparse.** It calls `load_config()` from `src/utils/config_loader.py:17-43`, which uses `parse_known_args()` with `add_help=False`. Unknown flags are silently ignored and `--help` does nothing. A typo in a config key is silently ignored too — every key is read via `cfg.get(key, DEFAULT)`.

2. **`roomformer_path` ignores the config** (`multiroom_pose_estimation.py:268`) and is always recomputed from `ROOT`. If the estimator fails looking for `predictions.json`, that is why.

3. **The Electron-generated configs in `data/projects/*/` are not a template.** Several keys there (`pose_estimates_dir`, `pose_json_path`, `panorama_dir`, `density_image_dir`) are never read. The estimator reads `pano_dir`, not `panorama_dir`, and `density_image_path`, not `density_image_dir`. Model new configs on the keys in Task 8, not on those files.

4. **The estimator writes `local_filter_results.json` to `output_dir`**, alongside per-pano `camera_pose.json` files. Some existing configs point `output_dir` at a mesh directory, so poses land somewhere surprising.

5. **`check=False` on the estimator is deliberate** — the visualization stage can crash after poses are written. Always verify `local_filter_results.json` exists rather than trusting the return code.

6. **PanoPin's build scripts take no arguments.** They read `subset.build_subset()` with hardcoded defaults. If the acceptance run needs a different room set, that function is what to change.

7. **`fgpl_export.export_alignment` reads `metadata.json` itself** and expects a `rotation_matrix` key. It returns a plain `list[str]` of admitted pano names, and the caller **must** use that list as FGPL's `cfg["pano_names"]` — panos present in `pano_names` but absent from the alignment file will `KeyError` in `compute_voronoi_assignment`.

8. **Stale line references.** The comment at `pose_search.py:137` cites `:334` and `:501`; the current anchors are `:343` and `:510`. The `(-n_tight, avg_dist)` sort moved from `:471` to `:492`.
