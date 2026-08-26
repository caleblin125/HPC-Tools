"""Example non-HPL application: tune compiler flags for a build-and-run benchmark.

This class exists to demonstrate that the framework is not tied to HPL. The
tunable parameters map directly to compiler flags; the fixed configuration can
carry the build directory; :meth:`command` rebuilds the project with the chosen
flags and runs a benchmark; :meth:`parse_result` extracts a scalar metric.

Because the tunables include *categorical* parameters, use one of the discrete
optimizers (random, smac3, raytune, hyperopt) with this application. The
continuous optimizers (CMA-ES, DEAP) require purely numeric tunables (see
``ParameterSpace.to_vector``).
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from hpc_autotuner.applications.base import Application
from hpc_autotuner.core.parameter import Parameter


class CompileFlagsApplication(Application):
    """Tune ``CFLAGS`` for an arbitrary ``make && ./benchmark`` workflow.

    Tunable parameters:

    * ``opt_level``  (int)        -> ``-O<level>``
    * ``arch``       (categorical) -> ``-march=<arch>``
    * ``lto``        (categorical) -> ``-flto`` when enabled

    The configuration may carry a fixed ``build_dir`` (default ``.``). The
    objective is whatever scalar metric the benchmark prints, parsed from a
    line shaped like ``score: 123.45`` (also accepts ``gflops``/``runtime``).
    """

    parameters = [
        Parameter("opt_level", "int", bounds=(0, 3), metadata={"flag": "-O"}),
        Parameter("arch", "categorical", choices=["native", "x86-64", "avx2"], metadata={"flag": "-march="}),
        Parameter("lto", "categorical", choices=[False, True], metadata={"flag": "-flto"}),
    ]

    objective_metric = "score"
    objective_direction = "maximize"

    def __init__(self, executable: str = "./benchmark", fixed: dict[str, Any] | None = None) -> None:
        self.executable = executable
        self._fixed = dict(fixed or {})

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "CompileFlagsApplication":
        """Build from an :class:`~hpc_autotuner.experiments.config.ExperimentConfig`."""
        return cls(executable=config.executable or "./benchmark", fixed=config.fixed)

    # ------------------------------------------------------------------
    # Application interface
    # ------------------------------------------------------------------

    def resolve_configuration(self, configuration: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(self._fixed)
        resolved.update(configuration)
        return resolved

    def _render_flags(self, configuration: dict[str, Any]) -> str:
        flags = [f"-O{int(configuration['opt_level'])}"]
        flags.append(f"-march={configuration['arch']}")
        if configuration.get("lto"):
            flags.append("-flto")
        return " ".join(flags)

    def command(self, configuration: dict[str, Any]) -> str:
        cflags = self._render_flags(configuration)
        build_dir = str(configuration.get("build_dir", "."))
        prefix = f"cd {shlex.quote(build_dir)}\n" if build_dir != "." else ""
        return (
            "set -e\n"
            f"{prefix}"
            f"echo \"CFLAGS='{cflags}'\"\n"
            "make clean >/dev/null 2>&1 || true\n"
            f"CFLAGS='{cflags}' make >/dev/null 2>&1 || true\n"
            f"{self.executable}\n"
        )

    def parse_result(self, output: str) -> dict[str, Any]:
        match = re.search(
            r"(?:score|gflops|flops|runtime)\s*[:=]\s*([0-9.eE+-]+)",
            output,
            flags=re.IGNORECASE,
        )
        if not match:
            return {"metrics": {}, "objective": None, "success": False}
        value = float(match.group(1))
        return {"metrics": {"score": value}, "objective": value, "success": True}
