from __future__ import annotations

import importlib
from typing import Any

from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.optimizers.base import Optimizer
from hpc_autotuner.optimizers.util import loss_from_result


class OptionalOptimizerAdapter(Optimizer):
    """Base class for third-party optimizer integrations (optional deps).

    Subclasses declare the pip package name and import module. The backend is
    imported lazily so the rest of the framework works without any of the
    six optimizer libraries installed.
    """

    package_name: str = ""
    import_name: str = ""

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        seed: int | None = None,
        direction: str = "maximize",
        **kwargs: Any,
    ) -> None:
        if not self.package_name:
            raise ValueError(f"{self.__class__.__name__} must define package_name.")
        self.parameters = list(parameters or [])
        self.seed = seed
        self.direction = direction
        self.kwargs = dict(kwargs)
        self.backend = self._load_backend()

    def _load_backend(self) -> Any:
        try:
            return importlib.import_module(self.import_name or self.package_name)
        except ImportError as exc:
            raise ImportError(
                f"{self.__class__.__name__} requires the optional dependency "
                f"'{self.package_name}'. Install it with: "
                f"pip install 'hpc-autotuner[{self.package_name}]'"
            ) from exc

    def _to_loss(self, result: dict[str, Any]) -> float:
        """Translate an evaluation record to a minimization loss."""
        return loss_from_result(result, direction=self.direction)

    def suggest(self) -> dict[str, Any]:
        raise NotImplementedError

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        raise NotImplementedError

