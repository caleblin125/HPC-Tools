from __future__ import annotations

from typing import Any

from hpc_autotuner.optimizers.adapter import OptionalOptimizerAdapter


class DEAPOptimizer(OptionalOptimizerAdapter):
    package_name = "deap"
    import_name = "deap"

    def suggest(self) -> dict[str, Any]:
        raise NotImplementedError("DEAP integration is a future adapter; it is intentionally not executed in tests.")

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        return None
