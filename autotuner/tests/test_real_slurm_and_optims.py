import importlib

import pytest

from hpc_autotuner.applications.hpl import HPLApplication
from hpc_autotuner.applications.slurm_test import SlurmTestApplication
from hpc_autotuner.optimizers.adapter import OptionalOptimizerAdapter
from hpc_autotuner.optimizers.cmaes import CMAESOptimizer
from hpc_autotuner.optimizers.deap import DEAPOptimizer
from hpc_autotuner.optimizers.hyperopt import HyperoptOptimizer
from hpc_autotuner.optimizers.raytune import RayTuneOptimizer
from hpc_autotuner.optimizers.smac3 import SMAC3Optimizer


def test_slurm_test_application_parses_output():
    app = SlurmTestApplication()
    parsed = app.parse_result(
        "hostname\nSLURM_JOB_ID=1234\nCONFIGURATION={scale=1.25, mode=fast}\nOBJECTIVE=2.0\nSUCCESS=true\n"
    )
    assert parsed["objective"] == 2.0
    assert parsed["success"] is True


def test_hpl_application_has_expected_parameter_space():
    app = HPLApplication()
    assert app.executable.exists() is False or app.executable.name in {"xhpl", "bin"}
    assert {p.name for p in app.parameters} == {"N", "NB", "P", "Q"}


def test_optional_optimizer_adapters_are_declared():
    adapter_classes = [
        SMAC3Optimizer,
        CMAESOptimizer,
        HyperoptOptimizer,
        DEAPOptimizer,
        RayTuneOptimizer,
    ]
    for cls in adapter_classes:
        assert issubclass(cls, OptionalOptimizerAdapter)
        assert cls.package_name


def test_optional_optimizers_fail_gracefully_when_dependency_missing():
    missing = [
        SMAC3Optimizer,
        CMAESOptimizer,
        HyperoptOptimizer,
        DEAPOptimizer,
        RayTuneOptimizer,
    ]
    for cls in missing:
        with pytest.raises(ImportError):
            cls(parameters=[{}])
