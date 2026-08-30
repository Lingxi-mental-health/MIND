#!/usr/bin/env bash
# PsySim-Std / PsySim-Adapt evaluation harness (Table 1).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIND_V2_ROOT="$(cd "${HERE}/../.." && pwd)"

PATIENT_SIM="${1:-psysim_std}"
CKPT="${CKPT:-${MIND_V2_ROOT}/outputs/grpo/best}"
DATA="${DATA:-${MIND_V2_ROOT}/data/test.parquet}"
OUT="${OUT:-${MIND_V2_ROOT}/results/${PATIENT_SIM}.jsonl}"

cd "${MIND_V2_ROOT}/.."
exec python -m mind_v2.scripts.evaluation.run_patientsim \
    --ckpt "${CKPT}" --data "${DATA}" --output "${OUT}" \
    --patient_sim "${PATIENT_SIM}"
