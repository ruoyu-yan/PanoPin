"""D21 fix test: does per-room loss CALIBRATION demote the fixed 'loss-sink' rooms and
lift recall? Computes the full loss matrix (query panos x all candidate rooms) via full
localize_pair on GPU, SAVES it (so calibration formulas can be re-analyzed without the
~2 h recompute), then reports raw vs leave-one-out-calibrated recall@1/3/5. GT dev-only.

Baseline_r = median loss(q, r) over query panos q whose true room != r (cross-matches).
Calibrated loss(p, r) = loss(p, r) - baseline_r. A true room matched genuinely sits well
below its cross-baseline (-> ranks first); a loss-sink sits ~at its baseline (-> demoted).

Usage:
  conda run -n panopin-gpu python smoke/exp_calibration.py            # compute + analyze
  conda run -n panopin-gpu python smoke/exp_calibration.py --load OUT # re-analyze saved matrix
"""
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import localize_pair
import s3dis_gt

QUERY_ROOMS = ["office_1", "office_3", "office_4", "office_5", "office_7", "office_8",
               "WC_1", "WC_2", "conferenceRoom_1", "lounge_1", "hallway_2", "storage_1"]

def resolve(uuid, d):
    for p in glob.glob(os.path.join(d, "*.png")):
        pt = os.path.basename(p).split("_")
        if len(pt) >= 2 and pt[1] == uuid: return p
    return None

def compute(out_path, sample_rate):
    pin(0)
    cfg = load_cfg(sample_rate=sample_rate)
    g = s3dis_gt.load_config(None); a = g["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", g)
    all_rooms = s3dis_gt.candidate_rooms("Area_3", g)
    clouds = {r: os.path.join(a["rooms_dir"], r, r + ".txt") for r in all_rooms}
    by_room = {}
    for u, v in gt.items():
        by_room.setdefault(v["room"], []).append(u)
    rows = []
    for qi, qr in enumerate(QUERY_ROOMS):
        uuid = by_room.get(qr, [None])[0]
        pano = resolve(uuid, a["pano_rgb_dir"]) if uuid else None
        if not pano:
            print(f"{qr}: no pano, skip", flush=True); continue
        t0 = time.time()
        losses = {cand: float(localize_pair(cfg, pano, c)[2]) for cand, c in clouds.items()}
        rows.append({"query_room": qr, "uuid": uuid, "losses": losses})
        print(f"[{qi+1}/{len(QUERY_ROOMS)}] {qr}: done in {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump({"candidates": all_rooms, "rows": rows}, open(out_path, "w"), indent=2)  # incremental save
    return all_rooms, rows

def analyze(all_rooms, rows):
    def ranks(score_of):  # score_of(row) -> {cand: score}; lower=better
        rk = []
        for row in rows:
            s = score_of(row); order = sorted(s, key=lambda c: s[c])
            rk.append(order.index(row["query_room"]) + 1)
        return rk
    raw_score = lambda row: row["losses"]
    # leave-one-out baseline per candidate: median loss over rows whose true room != candidate
    def baseline(cand):
        vals = [r["losses"][cand] for r in rows if r["query_room"] != cand]
        return float(np.median(vals)) if vals else 0.0
    base = {c: baseline(c) for c in all_rooms}
    calib_score = lambda row: {c: row["losses"][c] - base[c] for c in all_rooms}
    raw_rk, cal_rk = ranks(raw_score), ranks(calib_score)
    n = len(rows)
    print(f"\n{'pano(query room)':20s} {'raw_rank':>9s} {'calib_rank':>11s}")
    for row, rr, cr in zip(rows, raw_rk, cal_rk):
        flag = "  <-- fixed" if rr > 1 and cr == 1 else ("  <-- broke" if rr == 1 and cr > 1 else "")
        print(f"{row['query_room']:20s} {rr:9d} {cr:11d}{flag}")
    for k in (1, 3, 5):
        rr = sum(1 for x in raw_rk if x <= k); cr = sum(1 for x in cal_rk if x <= k)
        print(f"recall@{k}: raw {rr}/{n} ({100*rr/n:.0f}%)   calibrated {cr}/{n} ({100*cr/n:.0f}%)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "runs", "calib_matrix.json"))
    ap.add_argument("--load", default=None)
    ap.add_argument("--sample-rate", type=int, default=30)
    args = ap.parse_args()
    if args.load:
        d = json.load(open(args.load)); analyze(d["candidates"], d["rows"])
    else:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        all_rooms, rows = compute(args.out, args.sample_rate)
        analyze(all_rooms, rows)
        print(f"\nsaved loss matrix -> {args.out}")

if __name__ == "__main__":
    main()
