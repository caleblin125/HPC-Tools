"""Unit tests: parameter bounds, memory_fraction -> N, HPL config, parsing."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from hpc_autotuner.applications.hpl import HPLApplication
from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.core.space import ParameterSpace
from tests.conftest import fake_hpl_output


# ---------------------------------------------------------------------------
# parameter bounds
# ---------------------------------------------------------------------------


def test_benchmark_parameter_space_is_bounded():
    app = HPLApplication.for_benchmark(node_memory_bytes=512 * 1024**3)
    tunable = app.tunable_parameters
    assert [p.name for p in tunable] == ["memory_fraction"]
    (mf,) = tunable
    assert mf.kind == "float"
    assert mf.bounds == (0.80, 0.96)
    assert app.fixed_values["NB"] == 192
    assert app.fixed_values["P"] == 8
    assert app.fixed_values["Q"] == 16


def test_parameter_space_rejects_out_of_bounds_config():
    space = ParameterSpace([Parameter("memory_fraction", "float", bounds=(0.80, 0.96))])
    space.validate({"memory_fraction": 0.90})
    with pytest.raises(ValueError):
        space.validate({"memory_fraction": 0.50})
    with pytest.raises(ValueError):
        space.validate({"memory_fraction": 1.10})
    with pytest.raises(ValueError):
        space.validate({"other": 1.0})  # missing required tunable


def test_parameter_space_clips_to_bounds():
    space = ParameterSpace([Parameter("x", "float", bounds=(0.0, 1.0))])
    assert space.clip({"x": 1.7})["x"] == 1.0
    assert space.clip({"x": -0.4})["x"] == 0.0


def test_parameter_space_vector_roundtrip():
    space = ParameterSpace([Parameter("x", "float", bounds=(0.80, 0.96))])
    config = space.from_vector([0.5])
    assert config["x"] == pytest.approx(0.88)
    assert space.to_vector(config)[0] == pytest.approx(0.5)
    int_space = ParameterSpace([Parameter("nb", "int", bounds=(64, 256))])
    assert int_space.from_vector([1.0])["nb"] == 256


# ---------------------------------------------------------------------------
# memory_fraction -> N conversion
# ---------------------------------------------------------------------------


def test_memory_fraction_to_n_matches_hpl_faq():
    # HPL FAQ: memory ~= N^2 * 8 bytes (matrix of doubles). A 4 GiB node at
    # fraction 1.0 has ~4.29e9 bytes -> N ~= sqrt(4.29e9 / 8) ~= 23170.
    app = HPLApplication.for_benchmark(node_memory_bytes=4 * 1024**3)
    n = app.memory_fraction_to_n(1.0)
    expected = int(math.sqrt((4 * 1024**3) / 8.0))
    assert n == expected


def test_memory_fraction_scales_n_quadratically():
    app = HPLApplication.for_benchmark(node_memory_bytes=512 * 1024**3)
    n_low = app.memory_fraction_to_n(0.80)
    n_high = app.memory_fraction_to_n(0.96)
    assert n_low < n_high
    # N is quadratic in memory: doubling memory only multiplies N by sqrt(2).
    assert n_high / n_low == pytest.approx(math.sqrt(0.96 / 0.80), rel=0.01)
    # Both stay inside the node's physical memory at the documented 8 bytes/elem.
    assert n_high**2 * 8 <= 0.96 * 512 * 1024**3


def test_resolve_configuration_records_derived_values():
    app = HPLApplication.for_benchmark(
        node_memory_bytes=8 * 1024**3,
        fixed={"NB": 128, "P": 8, "Q": 16},
    )
    resolved = app.resolve_configuration({"memory_fraction": 0.90})
    assert resolved["memory_fraction"] == 0.90
    assert resolved["NB"] == 128
    assert resolved["P"] == 8
    assert resolved["Q"] == 16
    assert resolved["target_memory_bytes"] == int(0.90 * 8 * 1024**3)
    assert resolved["N"] == app.memory_fraction_to_n(0.90)
    # The HPL.dat algorithm parameters are present for reproducibility.
    assert resolved["BCAST"] == 1
    assert resolved["PFACT"] == 0


def test_resolve_configuration_rejects_out_of_bounds_fraction():
    app = HPLApplication.for_benchmark(node_memory_bytes=8 * 1024**3)
    with pytest.raises(ValueError):
        app.resolve_configuration({"memory_fraction": 0.5})
    with pytest.raises(ValueError):
        app.resolve_configuration({"memory_fraction": 0.99})

# ---------------------------------------------------------------------------
# HPL.dat configuration generation
# ---------------------------------------------------------------------------


def test_command_renders_hpl_dat_with_derived_n():
    app = HPLApplication.for_benchmark(
        node_memory_bytes=8 * 1024**3,
        fixed={"NB": 256, "P": 2, "Q": 2},
    )
    resolved = app.resolve_configuration({"memory_fraction": 0.85})
    command = app.command(resolved)
    assert isinstance(command, str)
    assert f"{resolved['N']}          Ns" in command
    assert "256         NBs" in command
    assert "2          Ps" in command
    assert "2          Qs" in command
    assert "<<'HPLDAT'" in command
    assert "HPLinpack benchmark input file" in command
    assert "BCASTs" in command


def test_command_requires_resolved_configuration():
    app = HPLApplication.for_benchmark(node_memory_bytes=8 * 1024**3)
    with pytest.raises(KeyError):
        app.command({"memory_fraction": 0.85})  # N missing


def test_legacy_mode_preserves_original_parameter_space():
    app = HPLApplication()
    assert {p.name for p in app.parameters} == {"N", "NB", "P", "Q"}
    command = app.command({"N": 1024, "NB": 128, "P": 2, "Q": 2})
    assert "1024          Ns" in command
    assert "128         NBs" in command


# ---------------------------------------------------------------------------
# full HPL parameter space (mirrors the HPLtuning work)
# ---------------------------------------------------------------------------

FULL_TUNABLES = [
    "N", "NB", "P", "PMAP", "PFACT", "NBMIN", "NDIV",
    "RFACT", "BCAST", "DEPTH", "SWAP", "L1", "U", "EQUIL",
]


def test_full_parameter_space_is_built():
    app = HPLApplication.for_benchmark(node_memory_bytes=8 * 1024**3, ntasks=128, tunable=FULL_TUNABLES)
    names = [p.name for p in app.tunable_parameters]
    for expected in ("N", "NB", "P", "BCAST", "PFACT", "NBMIN", "NDIV", "RFACT", "DEPTH", "SWAP", "L1", "U", "EQUIL"):
        assert expected in names

    p = next(p for p in app.parameters if p.name == "P")
    assert p.kind == "categorical"
    assert p.choices == [1, 2, 4, 8, 16, 32, 64, 128]

    # Q is derived from P (P*Q == ntasks), never an independent parameter.
    assert all(q.name != "Q" for q in app.parameters)

    # N bounds come from the memory model band [0.80, 0.96].
    n = next(p for p in app.parameters if p.name == "N")
    lo = HPLApplication.for_benchmark(node_memory_bytes=8 * 1024**3).memory_fraction_to_n(0.80)
    hi = HPLApplication.for_benchmark(node_memory_bytes=8 * 1024**3).memory_fraction_to_n(0.96)
    assert n.bounds == (lo, hi)

    # Remaining parameters stay fixed and are recorded.
    assert app.fixed_values["SWAP_THRESH"] == 64
    assert app.fixed_values["ALIGN"] == 8


def test_q_is_derived_from_tunable_p():
    app = HPLApplication.for_benchmark(node_memory_bytes=8 * 1024**3, ntasks=128, tunable=["N", "P"])
    assert app.resolve_configuration({"N": 30000, "P": 8})["Q"] == 16
    assert app.resolve_configuration({"N": 30000, "P": 32})["Q"] == 4
    with pytest.raises(ValueError, match="divide"):
        app.resolve_configuration({"N": 30000, "P": 7})  # 7 does not divide 128


def test_full_configuration_resolves_and_renders_tunable_algorithm_params():
    app = HPLApplication.for_benchmark(node_memory_bytes=8 * 1024**3, ntasks=128, tunable=FULL_TUNABLES)
    resolved = app.resolve_configuration({
        "N": 30000, "NB": 256, "P": 16, "PMAP": 1, "PFACT": 1, "NBMIN": 2,
        "NDIV": 3, "RFACT": 1, "BCAST": 2, "DEPTH": 3, "SWAP": 1, "L1": 1,
        "U": 0, "EQUIL": 0,
    })
    assert resolved["Q"] == 8
    assert resolved["target_memory_bytes"] == int(30000**2 * 8)

    command = app.command(resolved)
    assert "256         NBs" in command
    assert "16          Ps" in command
    assert "8          Qs" in command
    # Tunable algorithm choices must land in HPL.dat (not the fixed defaults).
    assert "2            BCASTs" in command
    assert "1            PFACTs" in command
    assert "3            DEPTHs" in command


def test_p_and_q_cannot_both_be_tunable():
    with pytest.raises(ValueError, match="P and Q"):
        HPLApplication.for_benchmark(tunable=["N", "P", "Q"])


def test_benchmark_config_declares_full_tunable_set():
    from hpc_autotuner.experiments.common import build_application
    from hpc_autotuner.experiments.config import ExperimentConfig

    config_path = Path(__file__).resolve().parents[1] / "configs" / "perlmutter_hpl.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    app = build_application(config)
    names = [p.name for p in app.tunable_parameters]
    assert "N" in names and "NB" in names and "P" in names and "BCAST" in names
    assert "Q" not in names  # derived from P
    assert config.slurm.ntasks == 128
    # NERSC convention: the QoS + constraint select the queue; no explicit
    # partition is needed (and the partition name is machine-version-specific).
    assert config.slurm.qos == "shared"
    assert config.slurm.partition is None
    assert config.slurm.account == "m4007"


# ---------------------------------------------------------------------------
# HPL result parsing
# ---------------------------------------------------------------------------


def test_parse_result_extracts_gflops():
    app = HPLApplication()
    parsed = app.parse_result(fake_hpl_output(n=4096, gflops=3.25, runtime=0.4))
    assert parsed["success"] is True
    assert parsed["objective"] == pytest.approx(3.25)
    assert parsed["metrics"]["gflops"] == pytest.approx(3.25)
    assert parsed["metrics"]["runtime"] == pytest.approx(0.4)


def test_parse_result_uses_last_results_row():
    output = fake_hpl_output(gflops=1.0) + fake_hpl_output(gflops=2.0)
    parsed = HPLApplication().parse_result(output)
    assert parsed["objective"] == pytest.approx(2.0)


def test_parse_result_fails_on_garbage():
    parsed = HPLApplication().parse_result("srun: error: could not start xhpl\n")
    assert parsed["success"] is False
    assert parsed["objective"] is None

