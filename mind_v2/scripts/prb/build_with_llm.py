"""CLI: synthesise PRB primitives for every clinical retrieval state.

Usage::

    python -m mind_v2.scripts.prb.build_with_llm \
        --in  data/prb/states.jsonl \
        --out data/prb/supports.jsonl \
        --builder_llm openrouter --model moonshotai/kimi-k2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mind_v2.mind.prb.build import synthesise_support
from mind_v2.mind.prb.entry import PRBEntry
from mind_v2.mind.utils.llm_client import LLMClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", required=True)
    p.add_argument("--out", dest="output_path", required=True)
    p.add_argument("--builder_llm", default="echo",
                   help="LLM provider: echo / openrouter / deepseek (Appendix F)")
    p.add_argument("--model", default="moonshotai/kimi-k2")
    args = p.parse_args()

    llm = LLMClient(provider=args.builder_llm, model=args.model)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    n = 0
    with open(args.input_path, "r", encoding="utf-8") as fin, \
         open(args.output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fields = synthesise_support(
                row["q_i"], llm=llm, next_question_hint=row.get("next_question", ""),
            )
            entry = PRBEntry(
                entry_id=str(uuid.uuid4()),
                q=row["q_i"],
                criterion=fields["criterion"],
                observed_evidence=fields["observed"],
                missing_check=fields["missing"],
                exclusion_cue=fields["exclusion"],
                next_inquiry_rationale=fields["rationale"],
                next_question=fields["next_question"],
                source_split=row.get("split", "train"),
                source_emr_id=row.get("emr_id", ""),
            )
            fout.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")
            n += 1
    print(f"[build_with_llm] wrote {n} primitives to {args.output_path}")


if __name__ == "__main__":
    main()
