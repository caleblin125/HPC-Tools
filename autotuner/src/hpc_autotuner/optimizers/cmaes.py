from __future__ import annotations

from typing import Any

import numpy as np

from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.core.space import ParameterSpace
from hpc_autotuner.optimizers.adapter import OptionalOptimizerAdapter


class CMAESOptimizer(OptionalOptimizerAdapter):
    """CMA-ES via the lightweight ``cmaes`` package.

    The optimizer works in the normalized ``[0, 1]`` hypercube mapped to the
    tunable parameter bounds. Evaluations are consumed one at a time: each
    :meth:`suggest` asks for a single candidate and each :meth:`observe`
    reports its loss, so the HPL evaluation loop stays strictly sequential.

    ``population_size`` defaults to 2 (the minimum that keeps CMA-ES's
    internal statistics well defined); every ``tell`` completes a generation.
    """

    package_name = "cmaes"
    import_name = "cmaes"

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        seed: int | None = None,
        direction: str = "maximize",
        sigma0: float = 0.25,
        population_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parameters, seed=seed, direction=direction, **kwargs)
        from cmaes import CMA

        self.space = ParameterSpace(self.parameters)
        if len(self.space.tunable) == 0:
            raise ValueError("CMA-ES requires at least one tunable parameter.")

        n_dims = len(self.space.tunable)
        self.es = CMA(
            mean=np.full(n_dims, 0.5, dtype=float),
            sigma=float(sigma0),
            bounds=np.array([[0.0, 1.0]] * n_dims, dtype=float),
            seed=seed or 0,
            population_size=population_size,  # None -> cmaes default (4 + 3*ln(d))
        )
        self.population_size = self.es.population_size
        self._pending: np.ndarray | None = None
        # ``cmaes`` requires full-generation tells, so results are buffered
        # until a generation completes. Evaluations still run one at a time;
        # the controller always observes attempt N before suggesting N+1.
        self._buffer: list[tuple[np.ndarray, float]] = []

    def suggest(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("CMA-ES cannot suggest a new point before the previous one is observed.")
        if len(self._buffer) >= self.population_size:
            self.es.tell(self._buffer)
            self._buffer.clear()
        x = self.es.ask()
        self._pending = x
        return self.space.from_vector(x, clip=True)

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        if self._pending is None:
            raise RuntimeError("CMA-ES observe() called without a pending suggestion.")
        loss = self._to_loss(result)
        self._buffer.append((self._pending, float(loss)))
        self._pending = None

