#!/usr/bin/env python3
"""End-to-end Slurm-backed tuning smoke test (Perlmutter).

Runs a small real tuning experiment against the Slurm scheduler on Perlmutter.
Run from a Perlmutter login node inside the autotuner venv:

    .venv/bin/python scripts/run_tuning_smoke.py slurm_test --budget 2
    .venv/bin/python scripts/run_tuning_smoke.py hpl --budget 1 \
        --fixed-config '{"N": 512, "NB": 96, "P": 1, "Q": 1}'

The runner's job template allocates one task per evaluation, so HPL must use a
P x Q grid that fits a single task (P=Q=1); pass --fixed-config for that.

Outputs land under outputs/autotuning/<run-group>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hpc_autotuner.applications.hpl import HPLApplication
from hpc_autotuner.applications.slurm_test import SlurmTestApplication
from hpc_autotuner.optimizers.random import RandomOptimizer
from hpc_autotuner.runner.runner import Runner
from hpc_autotuner.schedulers.slurm import SlurmScheduler


class FixedConfigOptimizer:
    """Duck-typed optimizer that always suggests a single fixed configuration."""

    def __init__(self, configuration: dict) -> None:
        self._configuration = dict(configuration)

    def suggest(self) -> dict:
        return dict(self._configuration)

    def observe(self, configuration: dict, evaluation: dict) -> None:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slurm-backed tuning smoke test.")
    parser.add_argument("application", choices=["slurm_test", "hpl"], help="Application to tune")
    parser.add_argument("--budget", type=int, default=2, help="Number of evaluations")
    parser.add_argument("--run-group", default="tuning_smoke", help="Output run-group name")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the optimizer")
    parser.add_argument(
        "--fixed-config",
        metavar="JSON",
        default=None,
        help="Pin every evaluation to this configuration (JSON object)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.home() / "HPC-Tools" / "autotuner"

    application = {
        "slurm_test": SlurmTestApplication(),
        "hpl": HPLApplication(),
    }[args.application]

    if args.fixed_config is not None:
        optimizer = FixedConfigOptimizer(json.loads(args.fixed_config))
    else:
        optimizer = RandomOptimizer(application.parameters, seed=args.seed)

    runner = Runner(
        optimizer=optimizer,
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
        line = (
            f"eval {evaluation.evaluation_id} job={evaluation.job_id} "
            f"status={evaluation.status} success={evaluation.success} "
            f"objective={evaluation.objective} config={evaluation.configuration}"
        )
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
