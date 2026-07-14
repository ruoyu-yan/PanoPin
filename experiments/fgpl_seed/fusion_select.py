"""Apply the fusion selection rules to the colored candidate pool and report per-pano→room
recall for each (geometry / F1 verify-select / F2 blends). Offline, no GPU. Fair: reads only
the pool scores + true room for scoring (GT used only to score, never to select). Produces the
FUSION_RESULTS.md table. RESULT (2026-07-14): fixed-pose color-verify is REFUTED — see
FUSION_RESULTS.md (color at unrefined FGPL poses is degenerate; office_5 loss-sink; D18/D20)."""
import os, sys, json
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from panopin import fusion


def _ranks(vals, asc):
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=not asc)
    r = [0] * len(vals)
    for k, i in enumerate(order):
        r[i] = k
    return r


def _geom(cs):    return max(range(len(cs)), key=lambda i: cs[i]["n_tight"])
def _rrf(cs, k=60):
    rg = _ranks([c["n_tight"] for c in cs], False); rc = _ranks([c["color"] for c in cs], True)
    return max(range(len(cs)), key=lambda i: 1/(k+rg[i]) + 1/(k+rc[i]))
def _borda(cs):
    rg = _ranks([c["n_tight"] for c in cs], False); rc = _ranks([c["color"] for c in cs], True)
    return min(range(len(cs)), key=lambda i: rg[i] + rc[i])
def _tiebreak(cs, margin=20):
    bg = max(c["n_tight"] for c in cs); pool = [i for i, c in enumerate(cs) if c["n_tight"] >= bg - margin]
    return min(pool, key=lambda i: cs[i]["color"])


SELECTORS = [
    ("geometry (n_tight)", _geom),
    ("F1 verify-select", lambda cs: fusion.verify_select(cs)),
    ("F2 RRF", _rrf),
    ("F2 Borda", _borda),
    ("F2 tie-break@20", _tiebreak),
]


def main():
    rows = subset.build_subset()
    true = {r["pano_name"]: r["room"] for r in rows}
    pool = json.load(open(paths.WORK / "seeds" / "fusion_pool_colored.json"))
    n = len(pool)
    gt_in = sum(1 for name, v in pool.items()
                if true[name] in {c["room"] for c in v["candidates"]})
    print(f"pool: {n} panos; GT-room-in-pool = {gt_in}/{n}\n")
    print(f"{'selector':22s} room_recall")
    for label, fn in SELECTORS:
        hit = sum(1 for name, v in pool.items() if v["candidates"][fn(v["candidates"])]["room"] == true[name])
        print(f"{label:22s} {hit}/{n}")
    print(f"\nref: color-only via CPO localize_pair (refined) = 8/12 (cpo_cache, D25/D28)")


if __name__ == "__main__":
    main()
