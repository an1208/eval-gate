"""
gate.py - the CI gate. Reads the saved eval results and fails the build if quality drops.

Three checks per metric, from evals/thresholds.json and evals/baseline.json:
    0. RELIABLE   no more than max_error_rate of the judge calls may have errored
    1. ABSOLUTE   the score must be at least the minimum in thresholds.json
    2. REGRESSION the score must not fall more than MAX_DROP below the committed baseline

Usage (run the evals first, they write evals/results/*.json):
    python -m evals.gate                     # check; exit code 1 if anything fails
    python -m evals.gate --update-baseline   # accept the current scores as the new baseline

On GitHub Actions the table is also written to the job summary.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
THRESHOLDS_PATH = HERE / "thresholds.json"
BASELINE_PATH = HERE / "baseline.json"


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def check(thresholds, results, baseline, max_drop, max_error_rate=0.2):
    """
    thresholds : {suite: {metric: minimum}}
    results    : {suite: {"summary": {metric: {"mean": ...}}} or None}
    baseline   : {suite: {metric: score}}
    Returns one row per metric with a PASS/FAIL status and a note.
    """
    rows = []
    for suite, metrics in thresholds.items():
        summary = (results.get(suite) or {}).get("summary") or {}
        for metric, minimum in metrics.items():
            entry = summary.get(metric) or {}
            score = entry.get("mean")
            n, errors = entry.get("n") or 0, entry.get("errors") or 0
            base = (baseline.get(suite) or {}).get(metric)

            if score is None:
                status, note = "FAIL", "no score (suite not run, or judge errors)"
            elif n and errors / n > max_error_rate:
                status, note = "FAIL", f"too many judge errors ({errors}/{n}), score is unreliable"
            elif score < minimum:
                status, note = "FAIL", f"below minimum {minimum:.2f}"
            elif base is not None and score < base - max_drop:
                status, note = "FAIL", f"regressed {base - score:.2f} from baseline {base:.2f}"
            else:
                status, note = "PASS", ""

            rows.append(
                {"suite": suite, "metric": metric, "score": score, "minimum": minimum,
                 "baseline": base, "status": status, "note": note}
            )
    return rows


def fmt(x):
    return "  n/a" if x is None else f"{x:5.2f}"


def print_table(rows):
    print(f"\n{'Suite':<11} {'Metric':<24} {'Score':>6} {'Min':>6} {'Base':>6}  Status")
    print("-" * 70)
    for r in rows:
        note = f"  {r['note']}" if r["note"] else ""
        print(f"{r['suite']:<11} {r['metric']:<24} {fmt(r['score']):>6} {fmt(r['minimum']):>6} {fmt(r['baseline']):>6}  {r['status']}{note}")


def write_github_summary(rows, passed):
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = [f"## EvalGate: {'PASSED' if passed else 'FAILED'}", "",
             "| Suite | Metric | Score | Min | Baseline | Status |", "|---|---|---|---|---|---|"]
    for r in rows:
        icon = "✅" if r["status"] == "PASS" else "❌"
        note = f" ({r['note']})" if r["note"] else ""
        lines.append(f"| {r['suite']} | {r['metric']} | {fmt(r['score']).strip()} | {fmt(r['minimum']).strip()} | {fmt(r['baseline']).strip()} | {icon} {r['status']}{note} |")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def update_baseline(thresholds, results):
    baseline = {}
    for suite, metrics in thresholds.items():
        summary = (results.get(suite) or {}).get("summary") or {}
        for metric in metrics:
            score = (summary.get(metric) or {}).get("mean")
            if score is not None:
                baseline.setdefault(suite, {})[metric] = round(score, 4)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Baseline written to {BASELINE_PATH}")
    return baseline


def main():
    parser = argparse.ArgumentParser(description="EvalGate CI gate")
    parser.add_argument("--update-baseline", action="store_true", help="save current scores as the baseline")
    args = parser.parse_args()

    config = load_json(THRESHOLDS_PATH)
    if config is None:
        sys.exit(f"Missing {THRESHOLDS_PATH}")
    max_drop = config.get("max_drop", 0.05)
    max_error_rate = config.get("max_error_rate", 0.2)
    thresholds = config["thresholds"]

    results = {suite: load_json(RESULTS_DIR / f"{suite}.json") for suite in thresholds}

    if args.update_baseline:
        update_baseline(thresholds, results)
        return

    baseline = load_json(BASELINE_PATH) or {}
    if not baseline:
        print("Note: no baseline.json yet, so only the absolute minimums are checked. "
              "Run with --update-baseline after a good run.")

    rows = check(thresholds, results, baseline, max_drop, max_error_rate)
    print_table(rows)

    failed = [r for r in rows if r["status"] == "FAIL"]
    passed = not failed
    write_github_summary(rows, passed)

    print(f"\nEvalGate {'PASSED' if passed else 'FAILED'}: {len(rows) - len(failed)}/{len(rows)} checks passed "
          f"(regression tolerance {max_drop:.2f})")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()