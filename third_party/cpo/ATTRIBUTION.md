# Vendored: 82magnolia/panoramic-localization (CPO/PICCOLO/LDL/FGPL)
- Upstream: https://github.com/82magnolia/panoramic-localization
- Commit: 7c070676937423c0a198e58abbfc6ed446144d8b
- License: Apache-2.0 (see ./LICENSE). Authors: Junho Kim, Hojun Jang, Changwoon Choi, Young Min Kim.
- Changes from upstream: removed `.git/` and `ldl/superglue_models/` (LDL-only weights, unused).
  No source edits. PanoPin composes CPO primitives from `src/panopin/`; cite CPO (ECCV 2022).
- Runtime env deps beyond the plan's D10 list: `open3d==0.19.0` was **required** (the vendored
  `data_utils.py:12` imports open3d at module load, so it cannot be dropped as D10 assumed — see
  `docs/DECISIONS.md` D12); `opencv-python` installed unpinned (resolved 5.0.0.93); `pytest` added
  as the test runner.
