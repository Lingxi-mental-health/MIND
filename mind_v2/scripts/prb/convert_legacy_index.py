"""CLI: convert a legacy pickle PRB index (``entries.pkl``) to JSONL.

The retriever refuses to load ``entries.pkl`` because unpickling executes
arbitrary Python. This one-off converter does the unpickling explicitly, behind
a required ``--i-trust-this-file`` flag, so the dangerous step is a deliberate
act on a file you have vouched for rather than a silent side effect of training.

Only run this on an index *you* built. If the index came from anywhere else,
rebuild it with ``mind_v2.scripts.prb.build_index`` instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mind_v2.mind.prb.entry import PRBEntry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index_dir", required=True, help="directory holding entries.pkl")
    p.add_argument(
        "--i-trust-this-file",
        action="store_true",
        help="required: confirm the pickle was produced locally by you",
    )
    args = p.parse_args()

    legacy_path = os.path.join(args.index_dir, "entries.pkl")
    out_path = os.path.join(args.index_dir, "entries.jsonl")

    if not os.path.exists(legacy_path):
        raise SystemExit(f"No legacy index at {legacy_path}")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} already exists; refusing to overwrite")
    if not args.i_trust_this_file:
        raise SystemExit(
            "Refusing to unpickle without --i-trust-this-file.\n"
            f"Unpickling {legacy_path} runs whatever code it contains. Pass the flag "
            "only if you built this index yourself; otherwise rebuild it with "
            "mind_v2.scripts.prb.build_index."
        )

    import pickle  # imported here so the unsafe path is explicit and local

    with open(legacy_path, "rb") as f:
        entries = pickle.load(f)

    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            data = e.as_dict() if isinstance(e, PRBEntry) else dict(e)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"[convert_legacy_index] {len(entries)} entries → {out_path}")
    print(f"[convert_legacy_index] you can now delete {legacy_path}")


if __name__ == "__main__":
    main()
