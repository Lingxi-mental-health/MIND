#!/usr/bin/env python3
"""Convert a legacy pickle RAG sidecar (``*.pkl``) to JSON.

``ragen/env/med_dialogue/rag_retriever.py`` refuses to load ``.pkl`` sidecars
because unpickling executes arbitrary Python. This one-off converter performs
the unpickling explicitly, behind a required ``--i-trust-this-file`` flag, so
the dangerous step is a deliberate act on a file you have vouched for rather
than a silent side effect of every training run.

Only run this on files *you* produced. Anything obtained from elsewhere should
be regenerated locally instead.

Usage::

    python scripts/convert_legacy_rag_index.py \\
        --path data/embeddings/retrieval_query_texts.pkl --i-trust-this-file
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="path to the legacy .pkl sidecar")
    p.add_argument(
        "--i-trust-this-file",
        action="store_true",
        help="required: confirm the pickle was produced locally by you",
    )
    args = p.parse_args()

    if not args.path.endswith(".pkl"):
        raise SystemExit("--path must point at a .pkl file")
    if not os.path.exists(args.path):
        raise SystemExit(f"No such file: {args.path}")

    out_path = args.path[:-4] + ".json"
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} already exists; refusing to overwrite")
    if not args.i_trust_this_file:
        raise SystemExit(
            "Refusing to unpickle without --i-trust-this-file.\n"
            f"Unpickling {args.path} runs whatever code it contains. Pass the flag "
            "only if you produced this file yourself; otherwise regenerate it."
        )

    import pickle  # imported here so the unsafe path is explicit and local

    with open(args.path, "rb") as f:
        obj = pickle.load(f)

    # Reject anything that is not plain JSON-serialisable data: if the pickle
    # held custom objects, converting it silently would hide that fact.
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

    n = len(obj) if hasattr(obj, "__len__") else "?"
    print(f"[convert_legacy_rag_index] {n} items → {out_path}")
    print(f"[convert_legacy_rag_index] you can now delete {args.path}")


if __name__ == "__main__":
    main()
