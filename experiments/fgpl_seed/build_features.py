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
