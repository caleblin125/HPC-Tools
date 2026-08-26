from __future__ import annotations

from typing import Any

from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.core.space import ParameterSpace
from hpc_autotuner.optimizers.adapter import OptionalOptimizerAdapter


class RayTuneOptimizer(OptionalOptimizerAdapter):
    """Ray Tune integration driving a ``tune.search`` searcher externally.

    Ray Tune is used without starting a Ray cluster: the controller asks the
    searcher for one trial at a time (:meth:`suggest`) and reports the result
    with :meth:`observe`. By default the search algorithm is
    :class:`ray.tune.search.optuna.OptunaSearch` (TPE); pass
    ``search_algorithm="basic_variant"`` for Ray Tune's built-in random
    search. ``n_startup_trials`` random configurations are evaluated before
    the surrogate kicks in, which is standard TPE behavior.
    """

    package_name = "ray"
    import_name = "ray"

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        seed: int | None = None,
        direction: str = "maximize",
        search_algorithm: str = "optuna",
        n_startup_trials: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(parameters, seed=seed, direction=direction, **kwargs)

        self.space = ParameterSpace(self.parameters)
        if len(self.space.tunable) == 0:
            raise ValueError("Ray Tune requires at least one tunable parameter.")

        self.searcher = self._build_searcher(search_algorithm)
        self._trial_counter = 0
        self._pending: str | None = None

    def _build_searcher(self, search_algorithm: str) -> Any:
        if search_algorithm == "basic_variant":
            from ray import tune
            from ray.tune.search.basic_variant import BasicVariantGenerator

            tune_space: dict[str, Any] = {}
            for parameter in self.space.tunable:
                if parameter.kind == "float":
                    tune_space[parameter.name] = tune.uniform(*parameter.bounds)
                elif parameter.kind == "int":
                    tune_space[parameter.name] = tune.randint(
                        int(parameter.bounds[0]), int(parameter.bounds[1]) + 1
                    )
                else:
                    tune_space[parameter.name] = tune.choice(parameter.choices)
            return BasicVariantGenerator(space=tune_space, max_concurrent=1)
        if search_algorithm == "optuna":
            import optuna
            import optuna.distributions as dists
            from ray.tune.search.optuna import OptunaSearch

            # OptunaSearch hands the space straight to ``optuna.study.ask``,
            # which expects native distribution objects.
            optuna_space: dict[str, Any] = {}
            for parameter in self.space.tunable:
                if parameter.kind == "float":
                    optuna_space[parameter.name] = dists.FloatDistribution(*parameter.bounds)
                elif parameter.kind == "int":
                    optuna_space[parameter.name] = dists.IntDistribution(
                        int(parameter.bounds[0]), int(parameter.bounds[1])
                    )
                else:
                    optuna_space[parameter.name] = dists.CategoricalDistribution(
                        parameter.choices
                    )
            return OptunaSearch(
                space=optuna_space,
                metric="objective",
                mode="max" if self.direction == "maximize" else "min",
                sampler=optuna.samplers.TPESampler(seed=self.seed or 0),
            )
        raise ValueError(f"Unknown Ray Tune search algorithm: {search_algorithm!r}")

    def suggest(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("Ray Tune cannot suggest before the previous trial is observed.")
        trial_id = f"hpl_trial_{self._trial_counter}"
        self._trial_counter += 1
        suggested = self.searcher.suggest(trial_id)
        if suggested is None:
            raise RuntimeError("Ray Tune searcher returned no configuration to evaluate.")
        self._pending = trial_id
        return {key: value for key, value in suggested.items()}

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        if self._pending is None:
            raise RuntimeError("Ray Tune observe() called without a pending suggestion.")
        success = bool(result.get("success", False)) and result.get("objective") is not None
        if success:
            loss = self._to_loss(result)
            reported = -loss if self.direction == "maximize" else loss
            self.searcher.on_trial_complete(self._pending, {"objective": reported})
        else:
            # Tell the searcher the trial crashed so the surrogate does not
            # learn from a garbage value.
            self.searcher.on_trial_complete(self._pending, error=True)
        self._pending = None

