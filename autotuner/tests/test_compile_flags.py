"""Unit tests for the generic non-HPL example application (compiler flags)."""

from __future__ import annotations

import pytest

from hpc_autotuner.applications.compile_flags import CompileFlagsApplication


def _app(**kwargs) -> CompileFlagsApplication:
    return CompileFlagsApplication(**kwargs)


def test_compile_flags_renders_command_with_flags():
    app = _app(executable="./bench")
    cmd = app.command({"opt_level": 2, "arch": "native", "lto": True})
    assert "-O2" in cmd
    assert "-march=native" in cmd
    assert "-flto" in cmd
    assert "./bench" in cmd

    cmd2 = app.command({"opt_level": 0, "arch": "avx2", "lto": False})
    assert "-O0" in cmd2
    assert "-march=avx2" in cmd2
    assert "-flto" not in cmd2
    assert "CFLAGS='-O0 -march=avx2'" in cmd2


def test_compile_flags_command_cds_into_build_dir():
    app = _app()
    cmd = app.command({"opt_level": 3, "arch": "x86-64", "lto": False, "build_dir": "my_app"})
    assert "cd my_app" in cmd


def test_compile_flags_parse_result():
    app = _app()
    parsed = app.parse_result("CFLAGS='-O2 -march=native'\nscore: 123.45\n")
    assert parsed["success"] is True
    assert parsed["objective"] == pytest.approx(123.45)
    assert parsed["metrics"]["score"] == pytest.approx(123.45)

    parsed2 = app.parse_result("build failed\n")
    assert parsed2["success"] is False
    assert parsed2["objective"] is None


def test_compile_flags_resolve_configuration_merges_fixed():
    app = _app(fixed={"build_dir": "app/"})
    resolved = app.resolve_configuration({"opt_level": 3, "arch": "x86-64", "lto": False})
    assert resolved["build_dir"] == "app/"
    assert resolved["opt_level"] == 3


def test_compile_flags_registered_and_buildable():
    from hpc_autotuner.experiments.common import build_application
    from hpc_autotuner.experiments.config import ExperimentConfig

    config = ExperimentConfig(application_type="compile_flags", executable="./bench")
    app = build_application(config)
    assert isinstance(app, CompileFlagsApplication)
    assert app.executable == "./bench"

    # Unknown application types are rejected with a helpful message.
    with pytest.raises(ValueError, match="compile_flags"):
        build_application(ExperimentConfig(application_type="nope"))


def test_compile_flags_space_supports_discrete_optimizers():
    from hpc_autotuner.core.space import ParameterSpace

    space = ParameterSpace(_app().parameters)
    names = [p.name for p in space.tunable]
    assert names == ["opt_level", "arch", "lto"]
    # Categorical tunables are intentional here; continuous optimizers would
    # raise NotImplementedError (documented in the application docstring).
    with pytest.raises(NotImplementedError):
        space.to_vector({"opt_level": 2, "arch": "native", "lto": True})
