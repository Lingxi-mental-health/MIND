"""CLI: external MDD-5k diagnostic evaluation (Table 2)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mind_v2.mind.evaluation.mdd5k import run_public_diagnostic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True,
                   help="JSONL with `dialogue` and `gold_diagnosis` per line")
    p.add_argument("--predictions", required=True,
                   help="JSONL with `case_id` and `predicted` per line")
    p.add_argument("--out", default="results/mdd5k.json")
    args = p.parse_args()

    pred_map = {}
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pred_map[row["case_id"]] = row["predicted"]

    cases = []
    with open(args.cases, "r", encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))

    res = run_public_diagnostic(
        cases=cases, predict_fn=lambda d: pred_map.get(d.get("case_id"), ""),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": round(res.accuracy * 100, 2),
            "macro_f1": round(res.macro_f1 * 100, 2),
            "confusion": res.confusion,
        }, f, indent=2)
    print(f"[mdd5k] wrote {args.out}")


if __name__ == "__main__":
    main()
