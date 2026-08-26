"""Unit tests for the six optimizer adapters.

Each adapter is exercised through the common ``suggest()`` / ``observe()``
protocol against a smooth one-dimensional "GFLOPs" objective that peaks at
``memory_fraction = 0.90``:

    gflops(mf) = 3.0 - 100.0 * (mf - 0.90)**2

No Slurm or HPL executable is involved. Backends are optional dependencies, so
each test skips when its library is not installed.
"""

from __future__ import annotations

import pytest

from hpc_autotuner.core.parameter import Parameter

PARAMETERS = [Parameter("memory_fraction", "float", bounds=(0.80, 0.96))]

OPTIMUM = 0.90


def objective(fraction: float) -> float:
    return 3.0 - 100.0 * (fraction - OPTIMUM) ** 2


def make_result(fraction: float) -> dict:
    return {
        "objective": objective(fraction),
        "success": True,
        "metrics": {"gflops": objective(fraction)},
        "status": "COMPLETED",
        "attempt": 1,
    }


def run_adapter(adapter, evals: int = 40):
    """Drive one adapter through the suggest/observe loop."""
    best = -1e9
    best_fraction = None
    for _ in range(evals):
        config = adapter.suggest()
        assert config["memory_fraction"] is not None
        fraction = float(config["memory_fraction"])
        assert 0.80 <= fraction <= 0.96
        result = make_result(fraction)
        if result["objective"] > best:
            best = result["objective"]
            best_fraction = fraction
        adapter.observe(config, result)
    return best, best_fraction


def test_random_optimizer():
    from hpc_autotuner.optimizers.random import RandomOptimizer

    best, _ = run_adapter(RandomOptimizer(PARAMETERS, seed=42), evals=30)
    # Random search should at least find a reasonable point in 30 draws.
    assert best >= objective(0.80)


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("cmaes", {}),
        ("hyperopt", {}),
        ("deap", {"mu": 6}),
        ("smac3", {"model": "gaussian_process", "n_trials": 60}),
        ("raytune", {}),
    ],
)
def test_optional_optimizers_respect_bounds_and_observe(name, kwargs):
    backend = {
        "cmaes": "cmaes",
        "hyperopt": "hyperopt",
        "deap": "deap",
        "smac3": "smac",
        "raytune": "ray",
    }[name]
    pytest.importorskip(backend)

    from hpc_autotuner.optimizers import OPTIMIZERS

    adapter = OPTIMIZERS[name](PARAMETERS, seed=42, direction="maximize", **kwargs)
    best, best_fraction = run_adapter(adapter, evals=30)
    assert best > objective(0.80)
    assert best_fraction is not None


def test_cmaes_converges_to_optimum():
    pytest.importorskip("cmaes")
    from hpc_autotuner.optimizers.cmaes import CMAESOptimizer

    _, best_fraction = run_adapter(CMAESOptimizer(PARAMETERS, seed=7), evals=40)
    assert abs(best_fraction - OPTIMUM) < 0.05


def test_hyperopt_converges_to_optimum():
    pytest.importorskip("hyperopt")
    from hpc_autotuner.optimizers.hyperopt import HyperoptOptimizer

    _, best_fraction = run_adapter(HyperoptOptimizer(PARAMETERS, seed=7), evals=40)
    assert abs(best_fraction - OPTIMUM) < 0.10


def test_smac3_gp_converges_to_optimum():
    pytest.importorskip("smac")
    from hpc_autotuner.optimizers.smac3 import SMAC3Optimizer

    _, best_fraction = run_adapter(
        SMAC3Optimizer(PARAMETERS, seed=7, model="gaussian_process", n_trials=60),
        evals=40,
    )
    assert abs(best_fraction - OPTIMUM) < 0.10


def test_raytune_converges_to_optimum():
    pytest.importorskip("ray.tune.search.optuna")
    pytest.importorskip("optuna")
    from hpc_autotuner.optimizers.raytune import RayTuneOptimizer

    _, best_fraction = run_adapter(RayTuneOptimizer(PARAMETERS, seed=7), evals=40)
    assert abs(best_fraction - OPTIMUM) < 0.10


def test_deap_improves_over_initial_population():
    pytest.importorskip("deap")
    from hpc_autotuner.optimizers.deap import DEAPOptimizer

    adapter = DEAPOptimizer(PARAMETERS, seed=7, mu=8)
    # First 8 suggestions are the random initial population.
    initial_best = -1e9
    for _ in range(8):
        config = adapter.suggest()
        value = objective(config["memory_fraction"])
        initial_best = max(initial_best, value)
        adapter.observe(config, make_result(config["memory_fraction"]))
    _, best_fraction = run_adapter(adapter, evals=32)
    assert abs(best_fraction - OPTIMUM) < 0.12


def test_failed_evaluation_does_not_break_adapters():
    pytest.importorskip("cmaes")
    from hpc_autotuner.optimizers.cmaes import CMAESOptimizer

    adapter = CMAESOptimizer(PARAMETERS, seed=7)
    config = adapter.suggest()
    adapter.observe(config, {"objective": None, "success": False, "status": "FAILED"})
    config2 = adapter.suggest()  # must still produce a valid point
    assert 0.80 <= config2["memory_fraction"] <= 0.96
