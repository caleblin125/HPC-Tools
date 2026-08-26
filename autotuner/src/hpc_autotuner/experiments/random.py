"""Random Search experiment driver.

Run from the project root (or as a Slurm parent job)::

    python -m hpc_autotuner.experiments.random --config configs/perlmutter_hpl.yaml
"""

from __future__ import annotations

import sys

from hpc_autotuner.experiments.common import driver_main

OPTIMIZER = "random"


def main(argv: list[str] | None = None) -> int:
    return driver_main(OPTIMIZER, argv)


if __name__ == "__main__":
    sys.exit(main())
