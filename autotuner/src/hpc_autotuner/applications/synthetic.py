from __future__ import annotations

from typing import Any

from hpc_autotuner.applications.base import Application
from hpc_autotuner.core.parameter import Parameter


class SyntheticApplication(Application):
    """A tiny deterministic test application for local tuning."""

    parameters = [
        Parameter("x", "int", bounds=(0, 8)),
        Parameter("y", "float", bounds=(0.0, 2.0)),
        Parameter("mode", "categorical", choices=["fast", "slow"]),
    ]

    def command(self, configuration: dict[str, Any]) -> list[str]:
        return [
            "python",
            "-c",
            (
                "import json; "
                "x = json.loads('" + repr(str(configuration["x"])) + "'); "
                "y = float(" + repr(configuration["y"]) + "); "
                "m = '" + str(configuration["mode"]) + "'; "
                "runtime = (x + 1.0) * y + (0.5 if m == 'fast' else 1.5); "
                "print(f'METRICS: runtime={runtime}; objective={runtime}; success=true')"
            ),
        ]

    def parse_result(self, output: str) -> dict[str, Any]:
        metrics: dict[str, float] = {}
        objective = None
        success = True
        for chunk in output.split():
            if chunk.startswith("runtime="):
                value = float(chunk.split("=", 1)[1].rstrip(";"))
                metrics["runtime"] = value
                objective = value
            elif chunk.startswith("success="):
                success = chunk.split("=", 1)[1].lower() == "true"
        return {"metrics": metrics, "objective": objective, "success": success}
