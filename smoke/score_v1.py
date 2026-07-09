"""Score PanoPin v1 (percentile calibration + confidence gate) on a cached loss matrix.
Reports OVERALL accuracy, and — the v1 headline — accuracy on the CONFIDENT subset +
coverage. GT (true room) used for scoring only. Loads runs/calib_matrix.json by default."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from panopin.calibrate import assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=os.path.join(os.path.dirname(__file__), "..", "runs", "calib_matrix.json"))
    ap.add_argument("--conf-threshold", type=float, default=-0.3)
    args = ap.parse_args()
    d = json.load(open(args.matrix))
    lm = {r["query_room"]: r["losses"] for r in d["rows"]}   # key by true room (for scoring)
    res = assign(lm, conf_threshold=args.conf_threshold)

    print(f"{'pano(true room)':18s} {'assigned':16s} {'conf(mm)  ':>10s} {'confident':9s} {'correct':7s}")
    conf_correct = conf_total = all_correct = 0
    for true_room, a in res.items():
        correct = a.room == true_room
        all_correct += correct
        if a.is_confident:
            conf_total += 1; conf_correct += correct
        print(f"{true_room:18s} {a.room:16s} {a.confidence:10.3f} {str(a.is_confident):9s} {str(correct):7s}")
    n = len(res)
    print(f"\nconf_threshold = {args.conf_threshold}")
    print(f"OVERALL accuracy : {all_correct}/{n} = {all_correct/n:.0%}")
    print(f"CONFIDENT subset : {conf_correct}/{conf_total} = {conf_correct/max(1,conf_total):.0%} accuracy, "
          f"coverage {conf_total}/{n} = {conf_total/n:.0%}")
    print(f"FLAGGED (abstain): {n-conf_total}/{n}")


if __name__ == "__main__":
    main()
