from __future__ import annotations

from typing import Any

import numpy as np

from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.core.space import ParameterSpace
from hpc_autotuner.optimizers.adapter import OptionalOptimizerAdapter


class HyperoptOptimizer(OptionalOptimizerAdapter):
    """Sequential Hyperopt (TPE) adapter.

    Hyperopt is driven through its low-level ``tpe.suggest`` protocol so that
    the controller keeps full ownership of the loop: :meth:`suggest` asks TPE
    for exactly one new trial and :meth:`observe` completes that trial's doc
    with the measured loss before it is committed to the ``Trials`` object.
    This gives Hyperopt the same strictly-sequential, one-HPL-run-at-a-time
    feedback as every other optimizer.
    """

    package_name = "hyperopt"
    import_name = "hyperopt"

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        seed: int | None = None,
        direction: str = "maximize",
        n_startup_jobs: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(parameters, seed=seed, direction=direction, **kwargs)
        from hyperopt import base as hb
        from hyperopt import hp

        self.space = ParameterSpace(self.parameters)
        if len(self.space.tunable) == 0:
            raise ValueError("Hyperopt requires at least one tunable parameter.")

        space_dict: dict[str, Any] = {}
        for parameter in self.space.tunable:
            if parameter.kind == "float":
                space_dict[parameter.name] = hp.uniform(parameter.name, *parameter.bounds)
            elif parameter.kind == "int":
                space_dict[parameter.name] = hp.quniform(
                    parameter.name, int(parameter.bounds[0]), int(parameter.bounds[1]), 1
                )
            elif parameter.kind == "categorical":
                space_dict[parameter.name] = hp.choice(parameter.name, parameter.choices)
            else:  # pragma: no cover - Parameter validates kind
                raise ValueError(f"Unsupported parameter kind: {parameter.kind}")

        self.domain = hb.Domain(lambda config: 0.0, space_dict)
        self.trials = hb.Trials()
        self.rstate = np.random.RandomState(seed or 0)
        self.n_startup_jobs = int(n_startup_jobs)
        self._pending: dict[str, Any] | None = None

    def suggest(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("Hyperopt cannot suggest before the previous trial is observed.")
        from hyperopt import tpe

        new_ids = self.trials.new_trial_ids(1)
        docs = tpe.suggest(new_ids, self.domain, self.trials, int(self.rstate.randint(2 ** 31 - 1)))
        if not docs:
            raise RuntimeError("Hyperopt TPE returned no new trial to evaluate.")
        doc = docs[0]
        self._pending = doc
        return self._config_from_doc(doc)

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        if self._pending is None:
            raise RuntimeError("Hyperopt observe() called without a pending suggestion.")
        from hyperopt import base as hb

        doc = self._pending
        doc["state"] = hb.JOB_STATE_DONE
        doc["result"] = {"loss": self._to_loss(result), "status": hb.STATUS_OK}
        self.trials.insert_trial_docs([doc])
        self.trials.refresh()
        self._pending = None

    # ------------------------------------------------------------------

    def _config_from_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        kind_by_name = {p.name: p for p in self.space.tunable}
        config: dict[str, Any] = {}
        for name, values in doc["misc"]["vals"].items():
            value = values[0]
            parameter = kind_by_name[name]
            if parameter.kind == "int":
                config[name] = int(round(float(value)))
            elif parameter.kind == "float":
                config[name] = float(value)
            else:
                config[name] = parameter.choices[int(value)]
        return config

