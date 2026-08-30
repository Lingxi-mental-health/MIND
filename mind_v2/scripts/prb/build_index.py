"""CLI: build a BGE vector index over PRB primitives."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mind_v2.mind.prb.build import read_entries
from mind_v2.mind.prb.retriever import PRBRetriever


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", required=True)
    p.add_argument("--out", dest="output_dir", required=True)
    p.add_argument("--bge_model_path", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    entries = read_entries(args.input_path)
    PRBRetriever.build_index(
        entries,
        out_dir=args.output_dir,
        bge_model_path=args.bge_model_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"[build_index] indexed {len(entries)} entries → {args.output_dir}")


if __name__ == "__main__":
    main()
