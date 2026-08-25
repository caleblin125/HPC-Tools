from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

ParameterKind = Literal["int", "float", "categorical"]


@dataclass(frozen=True)
class Parameter:
    """A tunable scalar parameter for an HPC application."""

    name: str
    kind: ParameterKind
    bounds: tuple[float, float] | None = None
    choices: list[Any] | None = None
    log_scale: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ValueError("Parameter name must be a non-empty string.")

        allowed = {"int", "float", "categorical"}
        if self.kind not in allowed:
            raise ValueError(f"Unsupported parameter kind: {self.kind!r}")

        if self.kind in {"int", "float"}:
            if self.bounds is None:
                raise ValueError(f"Parameter {self.name!r} requires numeric bounds.")
            if len(self.bounds) != 2:
                raise ValueError(f"Parameter {self.name!r} bounds must have exactly two values.")
            lower, upper = float(self.bounds[0]), float(self.bounds[1])
            if not lower < upper:
                raise ValueError(
                    f"Parameter {self.name!r} invalid bounds: lower must be < upper."
                )
            if self.kind == "int":
                if not float(lower).is_integer() or not float(upper).is_integer():
                    raise ValueError(
                        f"Integer parameter {self.name!r} bounds must be integral."
                    )
            object.__setattr__(self, "bounds", (lower, upper))
        else:
            if not self.choices:
                raise ValueError(f"Categorical parameter {self.name!r} requires choices.")
            object.__setattr__(self, "choices", list(self.choices))

    def sample(self, rng: random.Random | None = None) -> Any:
        rng = rng or random.Random()
        if self.kind == "int":
            lower, upper = [int(v) for v in self.bounds]
            return rng.randint(lower, upper)
        if self.kind == "float":
            lower, upper = self.bounds
            return rng.uniform(lower, upper)
        return rng.choice(self.choices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "choices": list(self.choices) if self.choices is not None else None,
            "log_scale": self.log_scale,
            "metadata": self.metadata,
        }
