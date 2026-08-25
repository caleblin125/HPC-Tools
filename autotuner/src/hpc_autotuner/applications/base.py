from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hpc_autotuner.core.parameter import Parameter


class Application(ABC):
    """Defines a tunable HPC application and how to execute it."""

    parameters: list[Parameter] = []

    @abstractmethod
    def command(self, configuration: dict[str, Any]) -> list[str] | str:
        """Translate a configuration into a command representation."""

    @abstractmethod
    def parse_result(self, output: str) -> dict[str, Any]:
        """Parse stdout/stderr into metrics and objective information."""
