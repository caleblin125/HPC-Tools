from __future__ import annotations

from typing import Any

from hpc_autotuner.optimizers.adapter import OptionalOptimizerAdapter


class CMAESOptimizer(OptionalOptimizerAdapter):
    package_name = "cmaes"
    import_name = "cmaes"

    def suggest(self) -> dict[str, Any]:
        raise NotImplementedError("CMA-ES integration is a future adapter; it is intentionally not executed in tests.")

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        return None
