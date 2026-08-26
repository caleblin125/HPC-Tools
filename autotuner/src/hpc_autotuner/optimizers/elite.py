"""Elitist random-restart local search over ordered discrete parameters.

Migrated from ``HPC-Tools/HPLtuning`` (the standalone ``EliteRandomSearch``
used to tune HPL in the previous project; see ``parameter_search.py``). The
algorithm is deliberately dependency-free: it keeps the best-scoring historical
configurations ("elites") and proposes new candidates by nudging a few
parameters by a small fraction of their ordered value range, falling back to
independent random samples until the elite population is seeded. This is an
evolutionary/local-search policy, not Bayesian optimization.

The original project's ``Q = TOTAL_TASKS // P`` handling is inherited from the
application layer: when ``P`` is tunable, the HPL application derives ``Q`` so
the grid always matches the Slurm task count.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.optimizers.base import Optimizer
from hpc_autotuner.optimizers.util import loss_from_result


class EliteSearchOptimizer(Optimizer):
    """Discrete elitist evolutionary search with no third-party dependency.

    Each parameter maps to an *ordered* list of allowed values:

    * ``int`` parameters span their bounds as a contiguous integer range,
    * ``categorical`` parameters use their declared ``choices``.

    ``float`` parameters are not supported (the search is discrete); use
    another optimizer for continuous spaces.
    """

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        seed: int | None = None,
        direction: str = "maximize",
        elite_count: int = 7,
        mutation_count: int = 3,
        mutation_fraction: float = 0.02,
        **kwargs: Any,
    ) -> None:
        self.parameters = list(parameters or [])
        self.seed = seed
        self.direction = direction
        self.elite_count = int(elite_count)
        self.mutation_count = int(mutation_count)
        self.mutation_fraction = float(mutation_fraction)
        self.rng = random.Random(seed)
        self._value_lists: dict[str, list[Any]] = {
            parameter.name: self._ordered_values(parameter) for parameter in self.parameters
        }
        #: Observed ``(configuration, loss, success)`` triples; losses are
        #: minimization losses computed via :func:`loss_from_result`.
        self.history: list[tuple[dict[str, Any], float, bool]] = []
        self._pending: dict[str, Any] | None = None
        self._elite_index = 0

    # ------------------------------------------------------------------
    # parameter mapping
    # ------------------------------------------------------------------

    @classmethod
    def _ordered_values(cls, parameter: Parameter) -> list[Any]:
        if parameter.kind == "int":
            low, high = int(parameter.bounds[0]), int(parameter.bounds[1])
            return list(range(low, high + 1))
        if parameter.kind == "categorical":
            return list(parameter.choices)
        raise NotImplementedError(
            "EliteSearchOptimizer is a discrete search and does not support float "
            f"parameter {parameter.name!r}. Use another optimizer for continuous "
            "spaces."
        )

    # ------------------------------------------------------------------
    # candidate generation (the EliteRandomSearch algorithm)
    # ------------------------------------------------------------------

    def _random_candidate(self) -> dict[str, Any]:
        """One independent uniform sample across all dimensions."""
        return {name: self.rng.choice(values) for name, values in self._value_lists.items()}

    def _mutate(self, configuration: Mapping[str, Any]) -> dict[str, Any]:
        """Nudge up to ``mutation_count`` parameters by a small index step.

        Mirrors ``EliteRandomSearch.mutate``: each chosen parameter moves by at
        most ``mutation_fraction * len(values)`` positions in its ordered list.
        """
        result = {
            name: configuration.get(name, self.rng.choice(values))
            for name, values in self._value_lists.items()
        }
        mutable = list(self._value_lists)
        changed = 0
        while mutable and changed < self.mutation_count:
            name = self.rng.choice(mutable)
            values = self._value_lists[name]
            current = result.get(name, self.rng.choice(values))
            try:
                index = values.index(current)
            except ValueError:
                index = self.rng.randrange(len(values))
            span = max(1, int(self.mutation_fraction * len(values)))
            new_index = max(0, min(len(values) - 1, index + self.rng.randint(-span, span)))
            result[name] = values[new_index]
            mutable.remove(name)
            changed += 1
        return result

    def _usable_history(self) -> list[tuple[dict[str, Any], float]]:
        """Successful observations, ready for elite selection."""
        return [(config, loss) for config, loss, success in self.history if success]

    def _best_elites(self) -> list[tuple[dict[str, Any], float]]:
        usable = self._usable_history()
        usable.sort(key=lambda pair: pair[1])
        return usable[: self.elite_count]

    def _propose(self) -> dict[str, Any]:
        """Seed the population with random points, then mutate elites.

        Until ``elite_count`` successful observations exist, proposals are
        independent random samples (like the original batch ``propose``).
        Afterwards each proposal mutates the next best elite in round-robin
        order, preserving the original search's diversity among the top
        candidates.
        """
        elites = self._best_elites()
        if len(elites) < self.elite_count:
            candidate = self._random_candidate()
        else:
            index = self._elite_index % self.elite_count
            self._elite_index += 1
            candidate = self._mutate(elites[index][0])

        # Avoid burning evaluations on a configuration that was already tried.
        seen = {tuple(sorted(config.items())) for config, _, _ in self.history}
        attempts = 0
        while tuple(sorted(candidate.items())) in seen and attempts < 10:
            candidate = self._random_candidate()
            attempts += 1
        return candidate

    # ------------------------------------------------------------------
    # Optimizer interface
    # ------------------------------------------------------------------

    def suggest(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError(
                "EliteSearchOptimizer cannot suggest before the previous point is observed."
            )
        candidate = self._propose()
        self._pending = candidate
        return dict(candidate)

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        if self._pending is None:
            raise RuntimeError(
                "EliteSearchOptimizer observe() called without a pending suggestion."
            )
        success = bool(result.get("success", False)) and result.get("objective") is not None
        loss = loss_from_result(result, direction=self.direction)
        self.history.append((dict(configuration), loss, success))
        self._pending = None

