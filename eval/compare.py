"""Compare stored accuracy runs: how often each engine agrees with the HF baseline."""

import argparse
import json
from pathlib import Path


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["index"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Reference run, usually the HF baseline")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    baseline = load(args.baseline)
    report = {}
    for path in args.runs:
        rows = load(path)
        shared = sorted(set(rows) & set(baseline))
        agreement = sum(rows[index]["correct"] == baseline[index]["correct"] for index in shared)
        report[Path(path).name] = {
            "questions": len(shared),
            "correct": sum(rows[index]["correct"] for index in shared),
            "baseline_correct": sum(baseline[index]["correct"] for index in shared),
            "agreement_with_baseline": agreement,
            "identical_answers": sum(rows[index]["predicted"] == baseline[index]["predicted"] for index in shared),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
