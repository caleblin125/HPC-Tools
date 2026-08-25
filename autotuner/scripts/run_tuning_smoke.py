#!/usr/bin/env python3
"""End-to-end Slurm-backed tuning smoke test (Perlmutter).

Runs a small real tuning experiment against the Slurm scheduler on Perlmutter.
Run from a Perlmutter login node inside the autotuner venv:

    .venv/bin/python scripts/run_tuning_smoke.py slurm_test --budget 2
    .venv/bin/python scripts/run_tuning_smoke.py hpl --budget 1

Outputs land under outputs/autotuning/<run-group>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hpc_autotuner.applications.hpl import HPLApplication
from hpc_autotuner.applications.slurm_test import SlurmTestApplication
from hpc_autotuner.optimizers.random import RandomOptimizer
from hpc_autotuner.runner.runner import Runner
from hpc_autotuner.schedulers.slurm import SlurmScheduler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slurm-backed tuning smoke test.")
    parser.add_argument("application", choices=["slurm_test", "hpl"], help="Application to tune")
    parser.add_argument("--budget", type=int, default=2, help="Number of evaluations")
    parser.add_argument("--run-group", default="tuning_smoke", help="Output run-group name")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the optimizer")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.home() / "HPC-Tools" / "autotuner"

    application = {
        "slurm_test": SlurmTestApplication(),
        "hpl": HPLApplication(),
    }[args.application]

    runner = Runner(
        optimizer=RandomOptimizer(application.parameters, seed=args.seed),
        application=application,
        scheduler=SlurmScheduler(project_root=root),
        storage=None,
        evaluation_budget=args.budget,
        project_root=root,
        run_group=args.run_group,
    )

    history = runner.run()

    print("\n=== results ===")
    for evaluation in history:
        print(
            f"eval {evaluation.evaluation_id} job={evaluation.job_id} "
            f"status={evaluation.status} success={evaluation.success} "
            f"objective={evaluation.objective} config={evaluation.configuration}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
