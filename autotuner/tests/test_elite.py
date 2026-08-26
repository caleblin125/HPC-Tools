"""Unit tests for the migrated Elite Search optimizer (from HPC-Tools/HPLtuning)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.optimizers.elite import EliteSearchOptimizer


def _result(objective, success=True):
    return {
        "objective": objective,
        "success": success,
        "metrics": {"gflops": objective if objective is not None else 0.0},
        "status": "COMPLETED" if success else "FAILED",
    }


def test_elite_seeds_random_population_then_mutates():
    opt = EliteSearchOptimizer([Parameter("x", "int", bounds=(0, 100))], seed=7)
    # The first `elite_count` suggestions seed the population with random points.
    for _ in range(opt.elite_count):
        cfg = opt.suggest()
        assert 0 <= cfg["x"] <= 100
        opt.observe(cfg, _result(float(cfg["x"])))
    # Once seeded, suggestions mutate the elites (still within bounds).
    cfg = opt.suggest()
    assert 0 <= cfg["x"] <= 100
    assert len(opt.history) == opt.elite_count


def test_elite_improves_on_smooth_objective():
    opt = EliteSearchOptimizer(
        [Parameter("x", "int", bounds=(0, 100))],
        seed=7,
        direction="maximize",
        mutation_fraction=0.1,
    )
    observed = []
    for _ in range(60):
        cfg = opt.suggest()
        x = cfg["x"]
        objective = 100.0 - (x - 50) ** 2
        observed.append((x, objective))
        opt.observe(cfg, _result(objective))
    best_x, _ = max(observed, key=lambda pair: pair[1])
    assert abs(best_x - 50) <= 15


def test_elite_respects_bounds_and_categoricals():
    opt = EliteSearchOptimizer(
        [
            Parameter("a", "int", bounds=(1, 8)),
            Parameter("b", "categorical", choices=[1, 2, 4, 8, 16]),
        ],
        seed=3,
    )
    for _ in range(12):
        cfg = opt.suggest()
        assert 1 <= cfg["a"] <= 8
        assert cfg["b"] in [1, 2, 4, 8, 16]
        opt.observe(cfg, _result(float(cfg["a"] + cfg["b"])))


def test_elite_survives_failed_observation():
    opt = EliteSearchOptimizer([Parameter("x", "int", bounds=(0, 10))], seed=1)
    cfg = opt.suggest()
    opt.observe(cfg, _result(None, success=False))
    cfg2 = opt.suggest()
    assert 0 <= cfg2["x"] <= 10


def test_elite_rejects_float_parameter():
    with pytest.raises(NotImplementedError, match="float"):
        EliteSearchOptimizer([Parameter("memory_fraction", "float", bounds=(0.80, 0.96))])


def test_elite_drives_full_hpl_space():
    from hpc_autotuner.experiments.common import build_application, build_optimizer
    from hpc_autotuner.experiments.config import ExperimentConfig

    config_path = Path(__file__).resolve().parents[1] / "configs" / "perlmutter_smoke.yaml"
    app = build_application(ExperimentConfig.from_yaml(config_path))
    opt = build_optimizer("elite", app.tunable_parameters, seed=42, direction="maximize", n_trials=100)
    for index in range(8):
        candidate = opt.suggest()
        resolved = app.resolve_configuration(dict(candidate))
        assert resolved["P"] * resolved["Q"] == 128
        assert resolved["N"] > 0 and "BCAST" in resolved
        opt.observe(resolved, _result(float(index) + 1.0))


def test_elite_registered_and_buildable():
    from hpc_autotuner.experiments.common import build_optimizer
    from hpc_autotuner.optimizers import OPTIMIZERS

    assert "elite" in OPTIMIZERS
    optimizer = build_optimizer(
        "elite",
        [Parameter("x", "int", bounds=(0, 10))],
        seed=1,
        direction="maximize",
        n_trials=100,
    )
    assert isinstance(optimizer, EliteSearchOptimizer)
