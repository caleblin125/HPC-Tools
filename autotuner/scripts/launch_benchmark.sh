#!/bin/bash
# Submit a *parent* autotuner Slurm job for one optimizer.
#
# Usage:
#   scripts/launch_benchmark.sh <optimizer> <config.yaml> [--budget N] [--run-group G] [--time HH:MM:SS]
#
# The machine-specific account/partition/QoS/constraint/time are read from the
# experiment YAML and passed to sbatch as CLI flags (overriding the generic
# #SBATCH lines in scripts/autotune_<optimizer>.slurm). The Python controller
# then runs inside the parent job and submits the HPL child jobs itself.
#
# Example:
#   scripts/launch_benchmark.sh random configs/perlmutter_smoke.yaml --budget 1
#
# The parent job's --time should comfortably cover all sequential child runs.

set -euo pipefail

OPT="${1:?usage: launch_benchmark.sh <optimizer> <config.yaml> [--budget N] [--run-group G] [--time T]}"
shift
CONFIG="${1:?usage: launch_benchmark.sh <optimizer> <config.yaml> [--budget N] [--run-group G] [--time T]}"
shift

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

read_slurm() {
    python - "$CONFIG" "$1" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
slurm = cfg.get("slurm") or {}
value = slurm.get(sys.argv[2])
print(value if value is not None else "")
PY
}

ACCOUNT="$(read_slurm account)"
PARTITION="$(read_slurm partition)"
QOS="$(read_slurm qos)"
CONSTRAINT="$(read_slurm constraint)"
TIME_LIMIT="$(read_slurm time)"
VENV="${AUTOTUNE_VENV:-${REPO_ROOT}/.venv}"

SBATCH_ARGS=()
[ -n "$ACCOUNT" ] && SBATCH_ARGS+=(--account "$ACCOUNT")
[ -n "$PARTITION" ] && SBATCH_ARGS+=(--partition "$PARTITION")
[ -n "$QOS" ] && SBATCH_ARGS+=(--qos "$QOS")
[ -n "$CONSTRAINT" ] && SBATCH_ARGS+=(--constraint "$CONSTRAINT")
[ -n "$TIME_LIMIT" ] && SBATCH_ARGS+=(--time "$TIME_LIMIT")

DRIVER_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --budget) DRIVER_ARGS+=(--budget "$2"); shift 2 ;;
        --run-group) DRIVER_ARGS+=(--run-group "$2"); shift 2 ;;
        --time) SBATCH_ARGS+=(--time "$2"); shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

SCRIPT="scripts/autotune_${OPT}.slurm"
if [ ! -f "$SCRIPT" ]; then
    echo "No parent script for optimizer '${OPT}': $SCRIPT" >&2
    exit 2
fi

echo "Submitting autotune-${OPT} parent job"
echo "  config   : ${CONFIG}"
echo "  sbatch   : ${SBATCH_ARGS[*]:-<file defaults>}"
echo "  driver   : ${DRIVER_ARGS[*]:-<config defaults>}"
echo "  venv     : ${VENV}"

mkdir -p outputs/slurm

sbatch "${SBATCH_ARGS[@]}" \
    --export=ALL,AUTOTUNE_CONFIG="$CONFIG",AUTOTUNE_VENV="$VENV" \
    "$SCRIPT" "${DRIVER_ARGS[@]}"
