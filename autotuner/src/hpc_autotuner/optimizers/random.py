from __future__ import annotations

import random
from typing import Any

from hpc_autotuner.core.parameter import Parameter

from .base import Optimizer


class RandomOptimizer(Optimizer):
    """A simple baseline optimizer that samples uniformly from the parameter space."""

    def __init__(self, parameters: list[Parameter], seed: int | None = None) -> None:
        self.parameters = list(parameters)
        self.rng = random.Random(seed)
        self.history: list[dict[str, Any]] = []

    def suggest(self) -> dict[str, Any]:
        configuration: dict[str, Any] = {}
        for parameter in self.parameters:
            value = parameter.sample(self.rng)
            configuration[parameter.name] = value
        if self.history and configuration in self.history:
            return self.suggest()
        self.history.append(configuration)
        return configuration

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        self.history.append(dict(configuration))
