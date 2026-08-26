"""Parameter-space helpers shared by the optimizers and the experiment controller.

A :class:`ParameterSpace` wraps a list of :class:`~hpc_autotuner.core.parameter.Parameter`
objects and provides:

* separation of *tunable* and *fixed* parameters,
* validation and clipping of configurations against the declared bounds,
* uniform random sampling,
* conversion between configurations and normalized ``[0, 1]`` vectors (used by
  continuous optimizers such as CMA-ES and DEAP).

``memory_fraction`` for HPL is a plain ``float`` parameter; the *derived* HPL
problem size ``N`` is computed by the application, not by the optimizer.
"""

from __future__ import annotations

import random
from typing import Any, Iterable

from hpc_autotuner.core.parameter import Parameter


class ParameterSpace:
    """A validated collection of tunable and fixed parameters."""

    def __init__(self, parameters: Iterable[Parameter] | None = None) -> None:
        self.parameters = list(parameters or [])
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate parameter names: {names}")

    # -- decomposition ----------------------------------------------------

    @property
    def tunable(self) -> list[Parameter]:
        """Parameters that the optimizer is allowed to vary."""
        return [p for p in self.parameters if p.fixed_value is None]

    @property
    def fixed(self) -> dict[str, Any]:
        """Values that are pinned for every evaluation."""
        return {p.name: p.fixed_value for p in self.parameters if p.fixed_value is not None}

    @property
    def tunable_names(self) -> list[str]:
        return [p.name for p in self.tunable]

    # -- validation -------------------------------------------------------

    def validate(self, configuration: dict[str, Any]) -> dict[str, Any]:
        """Validate a configuration and return it normalized.

        Raises ``ValueError`` when a tunable value lies outside its declared
        bounds or a required parameter is missing.
        """
        normalized: dict[str, Any] = dict(configuration)
        for parameter in self.tunable:
            if parameter.name not in normalized:
                raise ValueError(f"Missing value for tunable parameter {parameter.name!r}")
            value = normalized[parameter.name]
            if parameter.kind == "int":
                lower, upper = int(parameter.bounds[0]), int(parameter.bounds[1])
                ivalue = int(value)
                if not (lower <= ivalue <= upper):
                    raise ValueError(
                        f"{parameter.name}={value!r} outside bounds ({lower}, {upper})"
                    )
                normalized[parameter.name] = ivalue
            elif parameter.kind == "float":
                lower, upper = float(parameter.bounds[0]), float(parameter.bounds[1])
                fvalue = float(value)
                if not (lower <= fvalue <= upper):
                    raise ValueError(
                        f"{parameter.name}={value!r} outside bounds ({lower}, {upper})"
                    )
                normalized[parameter.name] = fvalue
            else:
                if value not in parameter.choices:
                    raise ValueError(
                        f"{parameter.name}={value!r} not in choices {parameter.choices}"
                    )
        return normalized

    def clip(self, configuration: dict[str, Any]) -> dict[str, Any]:
        """Clamp tunable values to their declared bounds (no validation errors)."""
        clipped: dict[str, Any] = dict(configuration)
        for parameter in self.tunable:
            if parameter.name not in clipped:
                continue
            value = clipped[parameter.name]
            if parameter.kind in {"int", "float"}:
                lower, upper = parameter.bounds
                clipped[parameter.name] = min(max(value, lower), upper)
        return clipped

    # -- sampling ---------------------------------------------------------

    def sample(self, rng: random.Random | None = None) -> dict[str, Any]:
        """Return a full configuration (tunables sampled, fixed values included)."""
        rng = rng or random.Random()
        config: dict[str, Any] = dict(self.fixed)
        for parameter in self.tunable:
            config[parameter.name] = parameter.sample(rng)
        return config

    # -- normalized-vector mapping (numeric tunables only) ----------------

    def to_vector(self, configuration: dict[str, Any]) -> list[float]:
        """Map a configuration to a normalized ``[0, 1]`` vector.

        Only numeric (``int``/``float``) tunable parameters are mapped;
        categorical tunables are not supported and raise ``NotImplementedError``.
        """
        vector: list[float] = []
        for parameter in self.tunable:
            if parameter.kind not in {"int", "float"}:
                raise NotImplementedError(
                    "Continuous optimizers do not support categorical parameter "
                    f"{parameter.name!r}; use discrete optimizers for it."
                )
            value = float(configuration[parameter.name])
            lower, upper = parameter.bounds
            vector.append((value - lower) / (upper - lower) if upper > lower else 0.0)
        return vector

    def from_vector(self, vector: Iterable[float], *, clip: bool = True) -> dict[str, Any]:
        """Map a normalized ``[0, 1]`` vector back to a configuration dict."""
        values = list(vector)
        if len(values) != len(self.tunable):
            raise ValueError(
                f"vector has {len(values)} entries but {len(self.tunable)} tunable "
                "parameters are declared"
            )
        config: dict[str, Any] = dict(self.fixed)
        for parameter, entry in zip(self.tunable, values):
            lower, upper = parameter.bounds
            value = lower + (upper - lower) * float(entry)
            if clip:
                value = min(max(value, lower), upper)
            if parameter.kind == "int":
                value = int(round(value))
            config[parameter.name] = value
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": [p.to_dict() for p in self.parameters],
            "tunable": [p.name for p in self.tunable],
            "fixed": self.fixed,
        }

