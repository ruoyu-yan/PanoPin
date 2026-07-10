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
