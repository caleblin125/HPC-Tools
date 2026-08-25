"""Small, dependency-free search policies for discrete parameter spaces.

The module deliberately knows nothing about Slurm, HPL, files, or result
formats beyond a caller-supplied score and validity key.  A benchmark adapter
is responsible for rendering a candidate, evaluating it, and saving a record.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from random import Random
import re
from typing import Any, Iterable, Mapping


class Param:
    """Mutable discrete parameter used by benchmark/template adapters.

    ``Parameter`` below is the immutable search-space definition consumed by
    search policies.  This companion class holds one selected value and can
    render it into a text template, so adapter code does not need to carry its
    own duplicate parameter implementation.
    """

    def __init__(self, name: str, values: Iterable[Any]):
        self.name = name
        self.values = list(values)
        if not self.values:
            raise ValueError(f"Parameter {name!r} has no values")
        self.rand = random.choice(self.values)

    def generate(self) -> None:
        self.rand = random.choice(self.values)

    def move(self, jump: float) -> bool:
        """Move by at most ``jump`` of the ordered value-list length."""
        try:
            index = self.values.index(self.rand)
        except ValueError:
            index = random.randint(0, len(self.values) - 1)
        span = max(1, int(jump * len(self.values)))
        new_index = max(0, min(len(self.values) - 1, index + random.randint(-span, span)))
        self.rand = self.values[new_index]
        return new_index != index

    def zero(self) -> None:
        self.rand = self.values[0]

    def next(self, jump: float = 0.0, overflow: bool = True) -> bool:
        """Advance by one or more discrete positions."""
        jump_max = max(1, int(jump * len(self.values)))
        next_index = self.values.index(self.rand) + random.randint(1, jump_max)
        self.rand = self.values[next_index % len(self.values) if overflow else min(next_index, len(self.values) - 1)]
        return len(self.values) <= next_index

    def replace(self, text: str) -> str:
        return re.sub(f"<{re.escape(self.name)}>", str(self.rand), text)

    def copy(self) -> "Param":
        clone = Param(self.name, self.values)
        clone.rand = self.rand
        return clone

    def __str__(self) -> str:
        return f"{self.name}: {self.rand}"

    __repr__ = __str__


@dataclass(frozen=True)
class Parameter:
    """One ordered, discrete search dimension."""

    name: str
    values: tuple[Any, ...]

    def __init__(self, name: str, values: Iterable[Any]):
        values = tuple(values)
        if not values:
            raise ValueError(f"Parameter {name!r} has no values")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class Candidate:
    """A proposed point and the policy that produced it."""

    values: dict[str, Any]
    origin: str


class EliteRandomSearch:
    """Elitist random-restart local search over ordered discrete parameters.

    Each batch mutates the highest-scoring valid historical candidates, then
    fills remaining slots with independent random samples.  It is a simple
    evolutionary/local-search policy, not Bayesian optimization.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        *,
        elite_count: int = 7,
        mutation_count: int = 3,
        mutation_fraction: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError("At least one parameter is required")
        self.elite_count = elite_count
        self.mutation_count = mutation_count
        self.mutation_fraction = mutation_fraction
        self.rng = Random(seed)

    def random_candidate(self) -> Candidate:
        return Candidate(
            {parameter.name: self.rng.choice(parameter.values) for parameter in self.parameters},
            "random",
        )

    def mutate(self, values: Mapping[str, Any]) -> Candidate:
        # Historical records commonly contain score/status metadata.  A
        # candidate contains only dimensions from this search space.
        result = {
            parameter.name: values.get(parameter.name, self.rng.choice(parameter.values))
            for parameter in self.parameters
        }
        movable = list(self.parameters)
        changed = 0
        while movable and changed < self.mutation_count:
            parameter = self.rng.choice(movable)
            current = result.get(parameter.name, self.rng.choice(parameter.values))
            try:
                index = parameter.values.index(current)
            except ValueError:
                index = self.rng.randrange(len(parameter.values))
            span = max(1, int(self.mutation_fraction * len(parameter.values)))
            new_index = max(0, min(len(parameter.values) - 1, index + self.rng.randint(-span, span)))
            result[parameter.name] = parameter.values[new_index]
            movable.remove(parameter)
            changed += 1
        return Candidate(result, "mutated-elite")

    def propose(
        self,
        history: Iterable[Mapping[str, Any]],
        count: int,
        *,
        score_key: str = "score",
        valid_key: str = "valid",
    ) -> list[Candidate]:
        """Return candidates without evaluating them.

        Records missing a valid numeric score are ignored for elite selection.
        When there is no usable history, all proposals are random, so a new
        search needs no special initialization path.
        """
        if count < 1:
            return []
        usable = [
            record for record in history
            if record.get(valid_key) and isinstance(record.get(score_key), (int, float))
        ]
        usable.sort(key=lambda record: record[score_key], reverse=True)
        proposals: list[Candidate] = []
        for record in usable[:self.elite_count]:
            if len(proposals) == count:
                break
            proposals.append(self.mutate(record))
        while len(proposals) < count:
            proposals.append(self.random_candidate())
        return proposals
