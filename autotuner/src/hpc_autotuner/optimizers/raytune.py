from __future__ import annotations

from typing import Any

from hpc_autotuner.optimizers.adapter import OptionalOptimizerAdapter


class RayTuneOptimizer(OptionalOptimizerAdapter):
    package_name = "ray"
    import_name = "ray"

    def suggest(self) -> dict[str, Any]:
        raise NotImplementedError("Ray Tune integration is a future adapter; it is intentionally not executed in tests.")

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        return None
