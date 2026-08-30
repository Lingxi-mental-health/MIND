"""CLI: build clinical retrieval states q_i for every train-split dialogue.

Usage::

    python -m mind_v2.scripts.prb.build_clinical_states \
        --in  MIND/data/train.parquet \
        --out mind_v2/data/prb/states.jsonl \
        --builder_llm echo

The ``echo`` provider is the default deterministic mock and does *not*
require any network access — useful for static / dry-run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mind_v2.mind.prb.build import build_clinical_state
from mind_v2.mind.utils.llm_client import LLMClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", required=True, help="Parquet of training dialogues")
    p.add_argument("--out", dest="output_path", required=True)
    p.add_argument("--builder_llm", default="echo",
                   help="LLM provider: echo / openrouter / deepseek")
    p.add_argument("--model", default="moonshotai/kimi-k2",
                   help="Concrete model identifier (e.g. for openrouter)")
    p.add_argument("--limit", type=int, default=-1)
    args = p.parse_args()

    df = pd.read_parquet(args.input_path)
    llm = LLMClient(provider=args.builder_llm, model=args.model)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    n = 0
    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            if args.limit > 0 and n >= args.limit:
                break
            dialogue = row.get("dialogue") or row.get("conversation") or []
            if hasattr(dialogue, "tolist"):
                dialogue = dialogue.tolist()
            q_i = build_clinical_state(dialogue, llm=llm)
            f.write(json.dumps({
                "emr_id": row.get("emr_id", str(i)),
                "split": "train",
                "q_i": q_i,
                "next_question": row.get("next_question", ""),
                "gold_label": row.get("gold_label", ""),
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"[build_clinical_states] wrote {n} states to {args.output_path}")


if __name__ == "__main__":
    main()
