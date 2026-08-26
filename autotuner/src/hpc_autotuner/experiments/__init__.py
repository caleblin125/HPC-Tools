"""Experiment drivers: one thin module per optimizer.

Each module is just a ``python -m hpc_autotuner.experiments.<name> --config ...``
entry point that delegates to :func:`hpc_autotuner.experiments.common.driver_main`.
All Slurm execution, logging, and resume logic lives in the common runner.
"""

from hpc_autotuner.experiments.common import (
    build_application,
    build_optimizer,
    driver_main,
    run_configured_experiment,
    run_experiment,
)

__all__ = [
    "build_application",
    "build_optimizer",
    "driver_main",
    "run_configured_experiment",
    "run_experiment",
]
