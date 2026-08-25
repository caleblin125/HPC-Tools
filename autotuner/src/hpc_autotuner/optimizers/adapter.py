from __future__ import annotations

import importlib
from typing import Any

from hpc_autotuner.optimizers.base import Optimizer


class OptionalOptimizerAdapter(Optimizer):
    """Base class for third-party optimizer integrations that are optional dependencies."""

    package_name: str = ""
    import_name: str = ""

    def __init__(self, parameters: list[Any] | None = None, **kwargs: Any) -> None:
        if not self.package_name:
            raise ValueError(f"{self.__class__.__name__} must define package_name.")
        self.parameters = list(parameters or [])
        self.backend = self._load_backend()
        self.kwargs = dict(kwargs)

    def _load_backend(self) -> Any:
        try:
            return importlib.import_module(self.import_name or self.package_name)
        except ImportError as exc:  # pragma: no cover - dependency guard is exercised in tests
            raise ImportError(
                f"{self.__class__.__name__} requires the optional dependency '{self.package_name}'. "
                f"Install it with: pip install {self.package_name}"
            ) from exc

    def suggest(self) -> dict[str, Any]:
        raise NotImplementedError

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        raise NotImplementedError
