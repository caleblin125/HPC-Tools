from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ObjectiveSpec:
    """Defines the optimization direction of a scalar objective."""

    name: str
    direction: Literal["minimize", "maximize"] = "minimize"

    def __post_init__(self) -> None:
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("Objective direction must be 'minimize' or 'maximize'.")


@dataclass(frozen=True)
class ResultMetric:
    """A single raw metric or objective value reported by an application."""

    name: str
    value: float
    unit: str | None = None
