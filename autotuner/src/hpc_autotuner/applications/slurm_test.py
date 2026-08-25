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
        return [
            "bash",
            "-lc",
            (
                "hostname; "
                "echo \"SLURM_JOB_ID=$SLURM_JOB_ID\"; "
                f"echo \"CONFIGURATION={{scale={scale}, mode={mode}}}\"; "
                f"objective=$(python - <<'PY'\n"
                f"import math\n"
                f"scale={scale}\n"
                f"mode='{mode}'\n"
                f"value = scale * (0.75 if mode == 'fast' else 1.5)\n"
                f"print(value)\n"
                "PY\n)\n"
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

        return {"metrics": metrics, "objective": objective, "success": success}
