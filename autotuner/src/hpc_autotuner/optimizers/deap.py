from __future__ import annotations

import random
from typing import Any

from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.core.space import ParameterSpace
from hpc_autotuner.optimizers.adapter import OptionalOptimizerAdapter


class DEAPOptimizer(OptionalOptimizerAdapter):
    """A steady-state genetic algorithm built on DEAP primitives.

    The algorithm maintains a population of ``mu`` individuals; the first
    ``mu`` evaluations seed the population with random individuals, and every
    subsequent :meth:`suggest` produces one offspring via tournament
    selection, blend crossover and Gaussian mutation. Individuals live in the
    normalized ``[0, 1]`` hypercube mapped onto the tunable bounds.

    This keeps DEAP useful in the strictly-sequential HPL loop: one HPL
    evaluation per individual, with feedback every attempt.
    """

    package_name = "deap"
    import_name = "deap"

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        seed: int | None = None,
        direction: str = "maximize",
        mu: int = 8,
        cxpb: float = 0.7,
        mutpb: float = 0.3,
        sigma: float = 0.2,
        indpb: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(parameters, seed=seed, direction=direction, **kwargs)
        from deap import base, creator, tools

        self.space = ParameterSpace(self.parameters)
        if len(self.space.tunable) == 0:
            raise ValueError("DEAP requires at least one tunable parameter.")
        n_dim = len(self.space.tunable)

        # creator.create cannot be called twice with the same name.
        if not hasattr(creator, "FitnessHPL"):
            creator.create("FitnessHPL", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividualHPL"):
            creator.create("IndividualHPL", list, fitness=creator.FitnessHPL)

        self.rng = random.Random(seed)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", self.rng.uniform, 0.0, 1.0)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.IndividualHPL,
            self.toolbox.attr_float,
            n=n_dim,
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=sigma, indpb=indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        self.mu = int(mu)
        self.cxpb = float(cxpb)
        self.mutpb = float(mutpb)
        self.pop: list[Any] = []
        self._pending: Any = None

    def suggest(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("DEAP cannot suggest a new individual before the previous one is observed.")
        if len(self.pop) < self.mu:
            individual = self.toolbox.individual()
        else:
            individual = self._make_offspring()
        self._pending = individual
        return self.space.from_vector(list(individual), clip=True)

    def _make_offspring(self) -> Any:
        parent_a, parent_b = self.toolbox.select(self.pop, 2)
        child_a, child_b = self.toolbox.clone(parent_a), self.toolbox.clone(parent_b)
        if self.rng.random() < self.cxpb:
            child_a, child_b = self.toolbox.mate(child_a, child_b)
        if self.rng.random() < self.mutpb:
            child_a, = self.toolbox.mutate(child_a)
        del child_a.fitness.values
        return child_a

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        if self._pending is None:
            raise RuntimeError("DEAP observe() called without a pending suggestion.")
        individual = self._pending
        loss = self._to_loss(result)
        individual.fitness.values = (-loss,)
        self.pop.append(individual)
        self._pending = None
        if len(self.pop) > self.mu:
            worst = min(self.pop, key=lambda ind: ind.fitness.values)
            self.pop.remove(worst)

