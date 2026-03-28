#!/bin/bash
#SBATCH -N 1
#SBATCH -n 56
#SBATCH -c 1
#SBATCH --output=output/slurm/job_%j.out
#SBATCH --error=output/slurm/job_%j.err

FILENAME="SAMPLE_RUN"
DESCRIPTION="$FILENAME with some parameters"

ROOT=$(pwd)

echo "$SLURM_JOB_ID : $DESCRIPTION" >> output/descriptions.txt

OUTDIR=$ROOT/output/$FILENAME
mkdir -p $OUTDIR


