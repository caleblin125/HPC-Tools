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

    @property
    def tunable_parameters(self) -> list[Parameter]:
        """Parameters the optimizer is allowed to vary (not pinned)."""
        return [p for p in self.parameters if p.fixed_value is None]

    @property
    def fixed_values(self) -> dict[str, Any]:
        """Values pinned for every evaluation."""
        return {p.name: p.fixed_value for p in self.parameters if p.fixed_value is not None}

    def resolve_configuration(self, configuration: dict[str, Any]) -> dict[str, Any]:
        """Return the complete, logged configuration for an evaluation.

        Applications may override this to derive values (for example HPL's
        problem size ``N``) from the tunable parameters. The base
        implementation simply returns a copy of the input merged with fixed
        values.
        """
        resolved: dict[str, Any] = dict(self.fixed_values)
        resolved.update(configuration)
        return resolved

