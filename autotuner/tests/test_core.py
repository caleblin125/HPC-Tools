import json
from pathlib import Path

import pytest

from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.core.paths import autotuning_output_dir, output_dir, project_root, run_group_dir
from hpc_autotuner.core.result import ObjectiveSpec, ResultMetric
from hpc_autotuner.optimizers.random import RandomOptimizer
from hpc_autotuner.storage.filesystem import FilesystemStorage


def test_parameter_validation_and_sampling():
    param = Parameter("learning_rate", "float", bounds=(1e-4, 1e-1))
    assert param.kind == "float"
    assert 1e-4 <= param.sample() <= 1e-1

    cat = Parameter("solver", "categorical", choices=["lbfgs", "sgd"])
    assert cat.sample() in {"lbfgs", "sgd"}

    with pytest.raises(ValueError):
        Parameter("bad_int", "int", bounds=(5, 3))

    with pytest.raises(ValueError):
        Parameter("bad_kind", "unknown", bounds=(0, 1))


def test_random_optimizer_suggests_valid_configs():
    params = [
        Parameter("x", "int", bounds=(0, 10)),
        Parameter("y", "float", bounds=(0.0, 1.0)),
        Parameter("mode", "categorical", choices=["a", "b"]),
    ]
    optimizer = RandomOptimizer(params, seed=123)

    cfg1 = optimizer.suggest()
    cfg2 = optimizer.suggest()

    assert set(cfg1) == {"x", "y", "mode"}
    assert cfg1 != cfg2
    assert 0 <= cfg1["x"] <= 10
    assert 0.0 <= cfg1["y"] <= 1.0
    assert cfg1["mode"] in {"a", "b"}


def test_path_helpers_follow_slurm_submission_dir(monkeypatch):
    monkeypatch.setenv("SLURM_SUBMIT_DIR", "/tmp/project-root")
    assert project_root() == Path("/tmp/project-root").resolve()
    assert output_dir() == Path("/tmp/project-root/outputs")
    assert autotuning_output_dir() == Path("/tmp/project-root/outputs/autotuning")
    assert run_group_dir("RUN_123") == Path("/tmp/project-root/outputs/RUN_123")


def test_filesystem_storage_round_trip(tmp_path):
    storage = FilesystemStorage(root=tmp_path / "experiment")
    storage.write_experiment_metadata({"name": "demo"})
    storage.append_evaluation({"evaluation_id": 7, "job_id": "123", "objective": 2.5, "success": True})

    payload = json.loads((tmp_path / "experiment" / "experiment.json").read_text())
    assert payload["name"] == "demo"

    lines = (tmp_path / "experiment" / "evaluations.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["objective"] == 2.5


def test_objective_model_supports_direction():
    objective = ObjectiveSpec(name="runtime", direction="minimize")
    assert objective.direction == "minimize"
    assert ResultMetric(name="runtime", value=12.4).value == 12.4
