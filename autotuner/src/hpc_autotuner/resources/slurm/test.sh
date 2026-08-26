#!/bin/bash

#SBATCH -J autotune_test
#SBATCH -N 1
#SBATCH -n 1
# Perlmutter rejects plain submissions; an explicit QoS/constraint is required.
#SBATCH -q shared
#SBATCH --constraint=cpu
#SBATCH --output=outputs/slurm/job_%j.out
#SBATCH --error=outputs/slurm/job_%j.err

set -euo pipefail

RUN_GROUP="AUTOTUNE_TEST"
DESCRIPTION="Testing HPC autotuner"

RUN_ROOT="${SLURM_SUBMIT_DIR}"
OUTPUT="${RUN_ROOT}/outputs"
OUTDIR="${OUTPUT}/${RUN_GROUP}"
OUTFILE="${OUTDIR}/${RUN_GROUP}_${SLURM_JOB_ID}.log"

mkdir -p "$OUTDIR"

echo \
    "$SLURM_JOB_ID : $DESCRIPTION - Using $SLURM_NTASKS tasks and $SLURM_JOB_NUM_NODES nodes" \
    >> "${OUTPUT}/descriptions.txt"

echo "Started running ${RUN_GROUP} at $(date)" \
    | tee -a "$OUTFILE"

start_time=$(date +%s)

echo "Hello from the compute node!"
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

sleep 5

echo "OBJECTIVE=5.0" | tee -a "$OUTFILE"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Finished running ${RUN_GROUP} at $(date)" \
    | tee -a "$OUTFILE"

echo "Elapsed time: ${elapsed} seconds" \
    | tee -a "$OUTFILE"