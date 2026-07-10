"""Central path + env constants for the FGPL-seed ablation. No logic."""
from pathlib import Path

FGPL_ROOT = Path("/home/ruoyu/scan2measure-webframework")
SCAN_ENV = "scan_env"        # FGPL build tools (baker/cluster/features): cu116, fine for those
GPU_ENV = "panopin-gpu"      # CPO localize_pair
# Estimator env: panopin-gpu (torch cu118) is Ada-native and ~1.2x faster than scan_env's
# cu116 (which falls back to CPU on the RTX 4060, D20). The XDF search is CPU-bound numpy
# either way (~330-390s/pano), so this is a modest but real win. Has all FGPL deps.
ESTIMATOR_ENV = "panopin-gpu"
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
