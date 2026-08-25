from __future__ import annotations

from typing import Any

from hpc_autotuner.optimizers.adapter import OptionalOptimizerAdapter


class HyperoptOptimizer(OptionalOptimizerAdapter):
    package_name = "hyperopt"
    import_name = "hyperopt"

    def suggest(self) -> dict[str, Any]:
        raise NotImplementedError("Hyperopt integration is a future adapter; it is intentionally not executed in tests.")

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        return None
