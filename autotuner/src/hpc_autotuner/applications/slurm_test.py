from __future__ import annotations

import re
from typing import Any

from hpc_autotuner.applications.base import Application
from hpc_autotuner.core.parameter import Parameter


class SlurmTestApplication(Application):
    """A minimal Slurm-compatible test application for real cluster validation."""

    parameters = [
        Parameter("scale", "float", bounds=(0.5, 2.0)),
        Parameter("mode", "categorical", choices=["baseline", "fast"]),
    ]

    def command(self, configuration: dict[str, Any]) -> list[str]:
        scale = configuration.get("scale", 1.0)
        mode = configuration.get("mode", "baseline")
        # Use a non-login shell: `bash -l` sources the NERSC profile, whose
        # showquota banner crashes on compute nodes (no /usr/lpp/mmfs).
        # awk is always present, so we do not depend on a python interpreter.
        return [
            "bash",
            "-c",
            (
                "hostname; "
                "echo \"SLURM_JOB_ID=$SLURM_JOB_ID\"; "
                f"echo \"CONFIGURATION={{scale={scale}, mode={mode}}}\"; "
                f"objective=$(awk -v s={scale} -v m='{mode}' "
                "'BEGIN { print (m == \"fast\") ? s * 0.75 : s * 1.5 }'); "
                "echo \"OBJECTIVE=$objective\"; "
                "echo \"SUCCESS=true\"; "
            ),
        ]

    def parse_result(self, output: str) -> dict[str, Any]:
        objective = None
        success = True
        metrics: dict[str, float] = {}

        match = re.search(r"OBJECTIVE=([0-9.eE+-]+)", output)
        if match:
            objective = float(match.group(1))
            metrics["objective"] = objective

        match = re.search(r"SUCCESS=(true|false)", output, flags=re.IGNORECASE)
        if match:
            success = match.group(1).lower() == "true"

        return {"metrics": {"objective": objective}, "objective": objective, "success": success}
