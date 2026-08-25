from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Optimizer(ABC):
    """Common optimizer interface.

    Optimizers are intentionally decoupled from Slurm or application details.
    They only consume configuration dicts and evaluation results.
    """

    @abstractmethod
    def suggest(self) -> dict[str, Any]:
        """Return the next configuration to evaluate."""

    @abstractmethod
    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        """Consume an observed evaluation result."""
