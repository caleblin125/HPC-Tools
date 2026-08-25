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


def test_hpl_application_parses_results_table():
    app = HPLApplication()
    sample = (
        "================================================================================\n"
        "T/V                N    NB     P     Q                 Time                 Gflops\n"
        "--------------------------------------------------------------------------------\n"
        "WR00C2R2         256    64     1     1               0.01               1.083e-02\n"
        "--------------------------------------------------------------------------------\n"
    )
    parsed = app.parse_result(sample)
    assert parsed["objective"] == 1.083e-02
    assert parsed["success"] is True
    assert parsed["metrics"]["gflops"] == 1.083e-02
    assert parsed["metrics"]["runtime"] == 0.01


def test_hpl_application_renders_hpl_dat_with_values():
    app = HPLApplication()
    command = app.command({"N": 1024, "NB": 128, "P": 2, "Q": 2})
    assert isinstance(command, str)
    assert "1024          Ns" in command
    assert "128         NBs" in command
    assert "2          Ps" in command
    assert "2          Qs" in command
    assert "<<'HPLDAT'" in command
    assert "HPLinpack benchmark input file" in command
    assert str(app.executable) in command
