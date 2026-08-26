#!/bin/bash
# Launch the full 100-attempt HPL benchmark for every optimizer that can
# represent the discrete/categorical HPL parameter space.
#
# The parameter space now includes the full HPL.dat set (N, NB, P -> Q,
# PMAP, PFACT, NBMIN, NDIV, RFACT, BCAST, DEPTH, SWAP, L1, U, EQUIL).
# DEAP and CMA-ES search a continuous space only, so they cannot represent
# the categorical parameters (notably P) and are intentionally excluded.
#
# 100 sequential full-node HPL runs can exceed the regular_1 wall limit
# (2 days). The parent jobs are given the maximum wall time and experiments
# are resumable: if a parent is interrupted, resubmit the same launch and it
# continues from evaluations.jsonl.
#
# Run the smoke test first (scripts/run_smoke_benchmark.sh) and verify
# outputs/autotuning/smoke_single|smoke_three before launching this.
#
# Usage: scripts/launch_full_benchmark.sh [config.yaml]
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:-configs/perlmutter_hpl.yaml}"
OPTIMIZERS=(random smac3 raytune hyperopt)
PARENT_TIME="${PARENT_TIME:-47:00:00}"

for OPT in "${OPTIMIZERS[@]}"; do
    echo "==> launching autotune-${OPT}: 100 runs, run-group=${OPT}, config=${CONFIG}"
    scripts/launch_benchmark.sh "$OPT" "$CONFIG" --budget 100 --run-group "$OPT" --time "$PARENT_TIME"
done

cat <<'NOTE'
All benchmark parents submitted. Monitor with:  squeue -u $USER
Results land in outputs/autotuning/<run_group>/{experiment.json,evaluations.jsonl}.
If a parent job hits the wall limit, rerun this script (or that optimizer's
launch) and it will resume from its evaluations.jsonl log.
NOTE
