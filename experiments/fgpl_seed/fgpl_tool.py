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
