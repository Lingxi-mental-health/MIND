"""CLI: support-faithfulness evaluation (Table 4 + Appendix E)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mind_v2.mind.evaluation.support_faithfulness import (
    aggregate as aggregate_faith, judge_turn,
)
from mind_v2.mind.utils.llm_client import LLMClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--judge", default="deepseek_v32",
                   help="Judge identifier — see :mod:`mind.utils.llm_client`")
    p.add_argument("--judge_provider", default="deepseek")
    p.add_argument("--judge_model", default="deepseek-chat")
    args = p.parse_args()

    llm = LLMClient(provider=args.judge_provider, model=args.judge_model)
    scores = []
    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for turn in row.get("turns", []):
                if turn.get("is_diagnosis", False):
                    continue
                s = judge_turn(
                    llm=llm,
                    dialogue=turn.get("history", ""),
                    retrieved_supports=turn.get("supports", ""),
                    doctor_turn=turn.get("doctor_turn", ""),
                )
                scores.append(s)
    avg = aggregate_faith(scores)
    print(json.dumps({
        "FC": round(avg.fc, 2),
        "SG": round(avg.sg, 2),
        "PF": round(avg.pf, 2),
        "Avg": round(avg.avg, 2),
        "n_turns": len(scores),
        "judge": args.judge,
    }, indent=2))


if __name__ == "__main__":
    main()
