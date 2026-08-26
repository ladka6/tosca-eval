"""Aggregate tosca-eval's per-(dataset, seed) logs into mean +/- std tables
over the 5 seeds (1993-1997), matching the paper's Table 1 (accuracy) and
Table 2 (efficiency) columns.

Log files are plain text (logging.FileHandler, default append mode), so a
rerun's output is appended after any earlier run in the same file rather
than overwriting it -- this script always takes the LAST occurrence of each
marker line per file, which is robust to that. Numeric values inside
printed dicts/lists may be wrapped as ``np.float64(...)`` (repr of a numpy
scalar via .format() on a dict/list) while scalar log lines are plain
numbers -- a single numeric regex handles both since it only pulls the
digits out, ignoring any wrapper.

Usage:
    python aggregate_results.py --root logs
"""
import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict

NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

TOP1_CURVE_RE = re.compile(r"CNN top1 curve: \[(.*?)\]")
AVG_ACC_RE = re.compile(r"Average Accuracy \(CNN\): (" + NUM + r")")
COST_SUMMARY_RE = re.compile(r"Cost summary: \{(.*?)\}")

COST_KEYS = [
    "Train Time (min)",
    "Eval (ms/smp)",
    "Inference (GFLOPs)",
    "Train (GFLOPs)",
    "Peak Train Mem. (MB)",
    "Params (M)",
    "Trainable Params (M)",
]


def _numbers(s):
    # Strip any np.float64(...) wrapper first -- otherwise the literal "64"
    # in "float64" itself gets matched as a spurious extra number.
    s = re.sub(r"np\.float64\(([^)]+)\)", r"\1", s)
    return [float(x) for x in re.findall(NUM, s)]


def parse_log(path):
    with open(path) as f:
        text = f.read()

    curve_matches = TOP1_CURVE_RE.findall(text)
    avg_acc_matches = AVG_ACC_RE.findall(text)
    cost_matches = COST_SUMMARY_RE.findall(text)

    if not curve_matches or not avg_acc_matches or not cost_matches:
        return None

    curve = _numbers(curve_matches[-1])
    avg_acc = float(avg_acc_matches[-1])
    cost_blob = cost_matches[-1]

    cost = {}
    for key in COST_KEYS:
        m = re.search(re.escape(key) + r"'?:\s*(?:np\.float64\()?(" + NUM + r")", cost_blob)
        if m:
            cost[key] = float(m.group(1))

    return {
        "final_top1": curve[-1] if curve else None,
        "avg_inc_acc": avg_acc,
        "n_tasks": len(curve),
        **cost,
    }


def find_runs(root):
    """-> {(dataset, prefix): [(seed, parsed_dict), ...]}

    Grouping must include prefix, not just dataset: different prefixes under
    the same model_name/dataset folder (e.g. the batch-mode TOSCA baseline,
    prefix=" ", vs. the per-sample-entropy ablation, prefix="persample")
    are different experiments that happen to share a log directory -- if
    keyed by dataset alone their seeds get silently averaged together."""
    runs = defaultdict(list)
    for path in glob.glob(os.path.join(root, "tosca", "*", "*", "*", "*.log")):
        # logs/tosca/<dataset>/<init_cls>/<increment>/<prefix>_<seed>_<backbone>.log
        parts = path.split(os.sep)
        dataset = parts[-4]
        fname = parts[-1]
        m = re.search(r"^(.*?)_(\d{4})_", fname)
        prefix = m.group(1).strip() if m else "?"
        seed = m.group(2) if m else "?"
        parsed = parse_log(path)
        if parsed is None:
            print(f"skipping (incomplete/unparseable): {path}")
            continue
        runs[(dataset, prefix)].append((seed, parsed))
    return runs


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} +/- {statistics.stdev(values):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs")
    args = ap.parse_args()

    runs = find_runs(args.root)

    def label(prefix):
        return prefix if prefix else "(default)"

    print("\n=== Accuracy (mean +/- std over seeds) ===")
    acc_rows = []
    for dataset, prefix in sorted(runs):
        entries = runs[(dataset, prefix)]
        seeds = [s for s, _ in entries]
        avg_acc = fmt([e["avg_inc_acc"] for _, e in entries])
        final_top1 = fmt([e["final_top1"] for _, e in entries])
        n = len(entries)
        tag = f"{dataset}[{label(prefix)}]"
        print(f"{tag:28s}  n_seeds={n}  seeds={seeds}  Abar={avg_acc:>16s}  A_T={final_top1:>16s}")
        acc_rows.append([dataset, prefix, n, ",".join(seeds), avg_acc, final_top1])

    print("\n=== Efficiency (mean +/- std over seeds) ===")
    eff_rows = []
    header = f"{'dataset[prefix]':28s}" + "".join(f"{k:>24s}" for k in COST_KEYS)
    print(header)
    for dataset, prefix in sorted(runs):
        entries = runs[(dataset, prefix)]
        row = [dataset, prefix]
        tag = f"{dataset}[{label(prefix)}]"
        line = f"{tag:28s}"
        for key in COST_KEYS:
            val = fmt([e.get(key) for _, e in entries])
            line += f"{val:>24s}"
            row.append(val)
        print(line)
        eff_rows.append(row)

    os.makedirs("results", exist_ok=True)
    with open("results/accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "prefix", "n_seeds", "seeds", "avg_inc_acc", "final_top1"])
        w.writerows(acc_rows)
    with open("results/efficiency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "prefix"] + COST_KEYS)
        w.writerows(eff_rows)
    print("\nWrote results/accuracy.csv and results/efficiency.csv")


if __name__ == "__main__":
    main()
