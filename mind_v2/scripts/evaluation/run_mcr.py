"""CLI: compute MCR + repetition over a results JSONL (Appendix D)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mind_v2.mind.evaluation.mcr import aggregate_mcr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="JSONL of per-episode trajectory dicts")
    args = p.parse_args()

    trajectories = []
    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            trajectories.append(row.get("trajectory", row))
    stats = aggregate_mcr(trajectories)
    print(json.dumps({
        "mcr": round(stats.mcr * 100, 2),
        "repetition": round(stats.repetition * 100, 2),
        "inquiry_turns": stats.inquiry_turns,
        "resolved_turns": stats.resolved_turns,
    }, indent=2))


if __name__ == "__main__":
    main()
