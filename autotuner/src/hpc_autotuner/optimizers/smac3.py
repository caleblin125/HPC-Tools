from __future__ import annotations

from typing import Any

from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.core.space import ParameterSpace
from hpc_autotuner.optimizers.adapter import OptionalOptimizerAdapter


def _native(value: Any) -> Any:
    """Convert numpy scalars (ConfigSpace returns them for categoricals) to
    plain Python types so configurations survive JSON serialization."""
    if hasattr(value, "item"):
        return value.item()
    return value


class SMAC3Optimizer(OptionalOptimizerAdapter):
    """SMAC3 (Bayesian optimization with random forests) adapter.

    SMAC3's public ``ask()``/``tell()`` protocol is used directly, so the
    controller keeps ownership of the sequential evaluation loop::

        info = smac.ask()          # TrialInfo wrapping a ConfigSpace Configuration
        smac.tell(info, TrialValue(cost=loss, ...))

    The default surrogate is SMAC3's RandomForest (requires the optional
    ``pyrfr`` package, which SMAC3 installs on Linux). Pass
    ``model="gaussian_process"`` to use a sklearn-based GP surrogate instead
    (useful on platforms where ``pyrfr`` cannot be built).
    """

    package_name = "smac"
    import_name = "smac"

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        seed: int | None = None,
        direction: str = "maximize",
        n_trials: int = 100,
        model: str = "default",
        **kwargs: Any,
    ) -> None:
        super().__init__(parameters, seed=seed, direction=direction, **kwargs)
        from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
        from smac import HyperparameterOptimizationFacade, Scenario

        self.space = ParameterSpace(self.parameters)
        if len(self.space.tunable) == 0:
            raise ValueError("SMAC3 requires at least one tunable parameter.")

        cs_space: dict[str, Any] = {}
        for parameter in self.space.tunable:
            if parameter.kind == "float":
                cs_space[parameter.name] = Float(parameter.name, parameter.bounds)
            elif parameter.kind == "int":
                cs_space[parameter.name] = Integer(
                    parameter.name, (int(parameter.bounds[0]), int(parameter.bounds[1]))
                )
            elif parameter.kind == "categorical":
                cs_space[parameter.name] = Categorical(parameter.name, parameter.choices)
            else:  # pragma: no cover - Parameter validates kind
                raise ValueError(f"Unsupported parameter kind: {parameter.kind}")

        self.cs = ConfigurationSpace(cs_space, seed=seed or 0)
        self.scenario = Scenario(self.cs, n_trials=int(n_trials), seed=seed or 0, deterministic=True)
        facade_kwargs: dict[str, Any] = {"overwrite": True}
        if model == "gaussian_process":
            facade_kwargs["model"] = self._build_gp_model()
        elif model != "default":
            raise ValueError(f"Unknown SMAC model: {model!r}")

        self.smac = HyperparameterOptimizationFacade(
            self.scenario, lambda config, seed=0: 0.0, **facade_kwargs
        )
        self._pending: Any = None

    def _build_gp_model(self) -> Any:
        from smac.model.gaussian_process.gaussian_process import GaussianProcess
        from smac.model.gaussian_process.kernels import (
            ConstantKernel,
            ProductKernel,
            RBFKernel,
            SumKernel,
            WhiteKernel,
        )

        kernel = SumKernel(
            ProductKernel(ConstantKernel(), RBFKernel()),
            ProductKernel(ConstantKernel(), WhiteKernel()),
        )
        return GaussianProcess(self.cs, kernel=kernel, seed=self.seed or 0)

    def suggest(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("SMAC3 cannot suggest before the previous trial is observed.")
        # ask() may return None when the intensifier wants to advance an
        # incumbent without a new configuration; tell(None, None) unblocks it.
        info = None
        for _ in range(4):
            info = self.smac.ask()
            if info is not None:
                break
            self.smac.tell(None, None)
        if info is None:
            raise RuntimeError("SMAC3 returned no configuration to evaluate.")
        self._pending = info
        return {key: _native(value) for key, value in dict(info.config).items()}

    def observe(self, configuration: dict[str, Any], result: Any) -> None:
        if self._pending is None:
            raise RuntimeError("SMAC3 observe() called without a pending suggestion.")
        from smac.runhistory import StatusType, TrialValue

        loss = self._to_loss(result)
        status = (
            StatusType.SUCCESS
            if result.get("objective") is not None and bool(result.get("success", False))
            else StatusType.CRASHED
        )
        self.smac.tell(
            self._pending,
            TrialValue(cost=loss, time=float(result.get("elapsed_seconds") or 0.0), status=status),
        )
        self._pending = None

