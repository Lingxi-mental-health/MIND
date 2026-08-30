"""CLI: assign reliability scores ρ_i to PRB primitives (§3.1.3)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mind_v2.mind.prb.entry import PRBEntry
from mind_v2.mind.prb.judge import judge_entry
from mind_v2.mind.utils.llm_client import LLMClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", required=True)
    p.add_argument("--out", dest="output_path", required=True)
    p.add_argument("--judge_llm", default="echo")
    p.add_argument("--model", default="deepseek-chat")
    args = p.parse_args()

    llm = LLMClient(provider=args.judge_llm, model=args.model)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    n = 0
    with open(args.input_path, "r", encoding="utf-8") as fin, \
         open(args.output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = PRBEntry.from_dict(json.loads(line))
            judge_entry(entry, llm=llm)
            fout.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")
            n += 1
    print(f"[judge_reliability] scored {n} primitives → {args.output_path}")


if __name__ == "__main__":
    main()
