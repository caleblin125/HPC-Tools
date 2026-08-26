"""Unit tests for the plotting data transformation layer."""

from __future__ import annotations

import json

from hpc_autotuner.plotting.core import load_experiments, transform


def _records(gflops: list[float | None]) -> list[dict]:
    return [
        {
            "attempt": i + 1,
            "status": "COMPLETED" if value is not None else "FAILED",
            "success": value is not None,
            "objective": value,
            "metrics": {"gflops": value, "runtime": 1.0},
        }
        for i, value in enumerate(gflops)
    ]


def test_transform_raw():
    xs, ys = transform(_records([1.0, 0.5, 3.0]), y="objective", aggregate="raw")
    assert xs == [1, 2, 3]
    assert ys == [1.0, 0.5, 3.0]


def test_transform_cummax():
    xs, ys = transform(_records([1.0, 0.5, 3.0, 2.0]), y="objective", aggregate="cummax")
    assert xs == [1, 2, 3, 4]
    assert ys == [1.0, 1.0, 3.0, 3.0]


def test_transform_skips_failed_attempts():
    xs, ys = transform(_records([1.0, None, 3.0]), y="objective", aggregate="cummax")
    assert xs == [1, 3]
    assert ys == [1.0, 3.0]


def test_transform_from_metrics_field():
    xs, ys = transform(_records([1.0, 2.0]), y="gflops", aggregate="cummax")
    assert ys == [1.0, 2.0]


def test_load_experiments_discovers_groups(tmp_path):
    for group, values in [("random", [1.0, 2.0]), ("cmaes", [0.5, 1.5])]:
        group_dir = tmp_path / group
        group_dir.mkdir(parents=True, exist_ok=True)
        with (group_dir / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
            for record in _records(values):
                handle.write(json.dumps(record) + "\n")

    experiments = load_experiments(tmp_path)
    assert {exp["name"] for exp in experiments} == {"random", "cmaes"}


def test_load_experiments_missing_input(tmp_path):
    import pytest

    from hpc_autotuner.plotting.core import load_experiments

    with pytest.raises(FileNotFoundError):
        load_experiments(tmp_path / "nope")
    with pytest.raises(ValueError):
        load_experiments(tmp_path)  # exists but has no evaluations.jsonl
