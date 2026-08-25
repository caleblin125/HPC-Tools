from __future__ import annotations

from typing import Any

from hpc_autotuner.optimizers.adapter import OptionalOptimizerAdapter


class SMAC3Optimizer(OptionalOptimizerAdapter):
    package_name = "smac3"
    import_name = "smac"

    def suggest(self) -> dict[str, Any]:
        raise NotImplementedError("SMAC3 integration is a future adapter; it is intentionally not executed in tests.")

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        return None
