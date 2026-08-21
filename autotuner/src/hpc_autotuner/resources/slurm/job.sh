#!/bin/bash

#SBATCH -J __JOB_NAME__
#SBATCH -N __NODES__
#SBATCH -n __NTASKS__
#SBATCH --output=outputs/slurm/job_%j.out
#SBATCH --error=outputs/slurm/job_%j.err

set -euo pipefail

#####################################
# Job information
#####################################

RUN_GROUP="__RUN_GROUP__"
DESCRIPTION="__DESCRIPTION__"

#####################################
# File organization
#####################################

RUN_ROOT="${SLURM_SUBMIT_DIR}"
OUTPUT="${RUN_ROOT}/outputs"
OUTDIR="${OUTPUT}/${RUN_GROUP}"
OUTFILE="${OUTDIR}/${RUN_GROUP}_${SLURM_JOB_ID}.log"

mkdir -p "$OUTDIR"

echo \
    "$SLURM_JOB_ID : $DESCRIPTION - Using $SLURM_NTASKS tasks and $SLURM_JOB_NUM_NODES nodes" \
    >> "${OUTPUT}/descriptions.txt"

#####################################
# Start
#####################################

echo "Started running ${RUN_GROUP} at $(date)" \
    | tee -a "$OUTFILE"

start_time=$(date +%s)

#####################################
# Application
#####################################

__COMMAND__

#####################################
# End
#####################################

end_time=$(date +%s)

elapsed=$((end_time - start_time))

echo "Finished running ${RUN_GROUP} at $(date)" \
    | tee -a "$OUTFILE"

echo "Elapsed time: ${elapsed} seconds" \
    | tee -a "$OUTFILE"