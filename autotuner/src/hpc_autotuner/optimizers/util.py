"""Shared helpers for the optimizer adapters.

All adapters speak one common protocol::

    config = optimizer.suggest()
    result = <run the application>          # dict evaluation record
    optimizer.observe(config, result)

The controller passes ``result`` as the JSON-able evaluation record written to
``evaluations.jsonl`` (see :class:`hpc_autotuner.core.evaluation.Evaluation`).
Adaptors translate that record into whatever their native library expects.
"""

from __future__ import annotations

from typing import Any

#: Objective value handed to a minimizing optimizer when an evaluation failed
#: (no objective was extracted). It is deliberately huge and identical for all
#: adapters so failed HPL runs are treated uniformly.
FAILURE_PENALTY = 1e9


def loss_from_result(result: dict[str, Any], direction: str = "maximize") -> float:
    """Convert an evaluation record into a *minimization* loss.

    ``direction`` is the experiment's objective direction ("maximize" for
    GFLOPs). For a successful evaluation the loss is ``-objective`` when
    maximizing; failed evaluations (no objective) always map to
    :data:`FAILURE_PENALTY` so they are never mistaken for good points.
    """
    objective = result.get("objective")
    success = bool(result.get("success", False))
    if objective is None or not success:
        return FAILURE_PENALTY
    value = float(objective)
    return -value if direction == "maximize" else value


def objective_value(result: dict[str, Any]) -> float:
    """Raw objective value for a successful evaluation (``NaN`` otherwise)."""
    objective = result.get("objective")
    success = bool(result.get("success", False))
    if objective is None or not success:
        return float("nan")
    return float(objective)
