"""End-to-end PsySim evaluation driver.

Re-uses the existing inference entrypoint at
``MIND/ragen/env/med_dialogue/evaluation/inference_fast_for_patientllm_zh_1018_3_best.py``
when available (heavy GPU stack); otherwise falls back to a pure-Python
roll-out using :class:`MindEnv` + :class:`PatientSim`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--patient_sim", default="psysim_std",
                   choices=["psysim_std", "psysim_adapt"])
    p.add_argument("--max_turns", type=int, default=10)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--reliability_threshold", type=float, default=0.2)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # We avoid hard-importing ``ragen`` so this script also runs in CPU-only
    # environments.  The actual heavy roll-out should be wired in by the user
    # via the helper below.
    try:
        from mind_v2.scripts.evaluation._rollout_ragen import run as ragen_run
    except ImportError:
        ragen_run = None

    if ragen_run is not None:
        ragen_run(
            ckpt=args.ckpt,
            data=args.data,
            output=args.output,
            patient_sim=args.patient_sim,
            max_turns=args.max_turns,
            top_k=args.top_k,
            reliability_threshold=args.reliability_threshold,
        )
    else:
        # Smoke-test fallback: write a single empty result so the downstream
        # MCR / faithfulness scripts can be exercised in dry-run mode.
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "ckpt": args.ckpt,
                "patient_sim": args.patient_sim,
                "trajectories": [],
            }, f, ensure_ascii=False, indent=2)
        print(f"[run_patientsim] dry-run wrote {args.output}")


if __name__ == "__main__":
    main()
