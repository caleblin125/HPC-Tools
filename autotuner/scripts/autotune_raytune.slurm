#!/bin/bash
# Parent autotuner job for Ray Tune.
#
# Machine-specific account/QoS/constraint/time settings are supplied at submit
# time by scripts/launch_benchmark.sh (or by hand with sbatch flags); the
# defaults below follow the standard NERSC structure (no explicit partition:
# the QoS + constraint select the queue).

#SBATCH -A m4007
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J autotune-raytune
#SBATCH -q shared
#SBATCH --constraint=cpu
#SBATCH --output=outputs/slurm/autotune_raytune_%j.out
#SBATCH --error=outputs/slurm/autotune_raytune_%j.err

set -euo pipefail

CONFIG="${AUTOTUNE_CONFIG:-${1:-}}"
if [ -z "$CONFIG" ]; then
    echo "ERROR: no experiment config. Set AUTOTUNE_CONFIG or pass the path as \$1." >&2
    exit 2
fi

if [ -n "${AUTOTUNE_VENV:-}" ]; then
    # shellcheck disable=SC1090
    source "${AUTOTUNE_VENV}/bin/activate"
fi

cd "${SLURM_SUBMIT_DIR}"

# Forward any additional arguments (e.g. --budget N, --run-group NAME).
shift || true
exec python -m hpc_autotuner.experiments.raytune --config "$CONFIG" "$@"
