#!/bin/bash
# Perlmutter smoke-test sequence for the HPL benchmark pipeline.
#
# Before launching any expensive 100-attempt experiment, run this to prove the
# full path works end to end:
#
#   1. one HPL evaluation  -> parse GFLOPs -> observe -> suggest attempt 2
#   2. three evaluations   (small sequential smoke run)
#
# Usage:  scripts/run_smoke_benchmark.sh [config.yaml]
#
# Monitor progress with:  watch -n 30 'squeue -u $USER'
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:-configs/perlmutter_smoke.yaml}"

echo "=============================================================="
echo "Smoke 1: single HPL evaluation with the random optimizer"
echo "  parent job -> 1 child HPL job -> parse -> attempt 2"
echo "=============================================================="
scripts/launch_benchmark.sh random "$CONFIG" --budget 1 --run-group smoke_single

echo "=============================================================="
echo "Smoke 2: three sequential HPL evaluations"
echo "=============================================================="
scripts/launch_benchmark.sh random "$CONFIG" --budget 3 --run-group smoke_three

cat <<'NOTE'

Jobs submitted. After they finish, verify:

    ls outputs/autotuning/smoke_single/evaluations.jsonl
    ls outputs/autotuning/smoke_three/evaluations.jsonl

Each record should contain a parsed objective (GFLOPs) with status COMPLETED.
NOTE
