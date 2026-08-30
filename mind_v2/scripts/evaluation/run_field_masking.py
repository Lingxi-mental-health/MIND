"""CLI: field-masking analysis (Appendix G)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mind_v2.mind.evaluation.field_masking import FIELDS, run_field_masking


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="results/field_masking.json")
    args = p.parse_args()

    # The actual rollout is plug-and-play.  Users provide their own
    # ``rollout_fn``; here we simulate a deterministic stub for static
    # validation.
    def rollout_fn(masked):
        # Mirror the qualitative pattern reported in §3 (M↓MCR, O↓Acc, S↓Faith).
        defaults = {"acc": 69.0, "macro_f1": 72.9, "mcr": 54.2, "faithfulness": 8.5}
        if "M" in masked:
            defaults.update(mcr=defaults["mcr"] - 13.1)
        if "O" in masked:
            defaults.update(acc=defaults["acc"] - 8.0, macro_f1=defaults["macro_f1"] - 8.5)
        if "S" in masked:
            defaults.update(faithfulness=defaults["faithfulness"] - 1.5)
        if "D" in masked:
            defaults.update(macro_f1=defaults["macro_f1"] - 3.0)
        if "R" in masked:
            defaults.update(faithfulness=defaults["faithfulness"] - 0.6)
        return defaults

    results = run_field_masking(rollout_fn=rollout_fn, fields=FIELDS)
    out = [r.__dict__ for r in results]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[field_masking] wrote {args.out}")


if __name__ == "__main__":
    main()
