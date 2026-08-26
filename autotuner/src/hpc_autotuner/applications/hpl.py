from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from hpc_autotuner.applications.base import Application
from hpc_autotuner.core.parameter import Parameter

_BYTES_PER_DOUBLE = 8


class HPLApplication(Application):
    """Adapter for the HPL benchmark executable (``xhpl``).

    The adapter supports two parameter spaces:

    * ``legacy`` mode (the default): ``N``, ``NB``, ``P``, ``Q`` are all
      tunable. This mode is kept for the original runner and smoke tests.
    * benchmark mode (:meth:`for_benchmark`): ``memory_fraction`` is the only
      tunable; ``NB``/``P``/``Q`` and every HPL algorithm choice are fixed.
      The problem size ``N`` is *derived* from the requested memory fraction.

    Memory model
    ------------
    The HPL FAQ (https://www.netlib.org/benchmark/hpl/faqs.html) states that
    "the amount of memory used by HPL is essentially the size of the
    coefficient matrix", i.e. for an ``N x N`` matrix of doubles::

        memory_bytes ~= N**2 * 8

    Solving for ``N``::

        N = floor(sqrt(target_memory_bytes / 8))

    ``memory_fraction`` therefore does **not** scale linearly with ``N``;
    doubling the memory only grows ``N`` by ``sqrt(2)``. The fraction band
    ``[0.80, 0.96]`` intentionally leaves the OS headroom recommended by the
    FAQ ("as a rule of thumb, 80 % of the total amount of memory is a good
    guess").
    """

    HPL_ROOT = Path("/global/common/software/m4007/opt/hpl-2.3")

    BYTES_PER_DOUBLE = _BYTES_PER_DOUBLE
    DEFAULT_MEMORY_FACTOR = 1.0
    DEFAULT_NODE_MEMORY_BYTES = 512 * 1024**3  # Perlmutter CPU node (~512 GiB)
    DEFAULT_MEMORY_FRACTION_BOUNDS = (0.80, 0.96)

    # Fixed HPL.dat algorithm choices used in benchmark mode. They are kept
    # out of the tunable parameter space so every optimizer sees the same
    # execution path; they are still recorded in each evaluation log.
    DEFAULT_FIXED_HPL_PARAMS: dict[str, Any] = {
        "PMAP": 0,
        "PFACT": 0,
        "NBMIN": 4,
        "NDIV": 4,
        "RFACT": 0,
        "BCAST": 1,
        "DEPTH": 1,
        "SWAP": 0,
        "SWAP_THRESH": 64,
        "L1": 0,
        "U": 1,
        "EQUIL": 1,
        "ALIGN": 8,
    }

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(self, executable: str | Path | None = None) -> None:
        # Legacy mode: N/NB/P/Q are all tunable (backwards compatible).
        self._executable_override = Path(executable) if executable else None
        self.parameters = [
            Parameter("N", "int", bounds=(64, 4096)),
            Parameter("NB", "int", bounds=(32, 256)),
            Parameter("P", "int", bounds=(1, 32)),
            Parameter("Q", "int", bounds=(1, 32)),
        ]
        self.node_memory_bytes: int = self.DEFAULT_NODE_MEMORY_BYTES
        self.memory_factor: float = self.DEFAULT_MEMORY_FACTOR
        self.memory_fraction_bounds: tuple[float, float] = self.DEFAULT_MEMORY_FRACTION_BOUNDS
        self.fixed_hpl_params: dict[str, Any] = dict(self.DEFAULT_FIXED_HPL_PARAMS)


    @classmethod
    def for_benchmark(
        cls,
        *,
        executable: str | Path | None = None,
        node_memory_bytes: int = DEFAULT_NODE_MEMORY_BYTES,
        memory_factor: float = DEFAULT_MEMORY_FACTOR,
        memory_fraction_bounds: tuple[float, float] = DEFAULT_MEMORY_FRACTION_BOUNDS,
        fixed: dict[str, Any] | None = None,
    ) -> "HPLApplication":
        """Build an HPL application configured for the benchmark experiment.

        ``memory_fraction`` (a float in ``memory_fraction_bounds``) is the only
        tunable; everything else is fixed. ``fixed`` may override the default
        ``NB``/``P``/``Q`` values and any HPL.dat algorithm choice.
        """
        app = cls(executable=executable)
        app.node_memory_bytes = int(node_memory_bytes)
        app.memory_factor = float(memory_factor)
        app.memory_fraction_bounds = tuple(float(v) for v in memory_fraction_bounds)
        app.fixed_hpl_params.update(fixed or {})
        lo, hi = app.memory_fraction_bounds
        if not 0.0 < lo < hi:
            raise ValueError(f"Invalid memory_fraction bounds: {app.memory_fraction_bounds}")

        nb = int(app.fixed_hpl_params.get("NB", 192))
        p = int(app.fixed_hpl_params.get("P", 8))
        q = int(app.fixed_hpl_params.get("Q", 16))
        fixed_params = [
            Parameter("NB", "int", bounds=(1, 1024), fixed_value=nb),
            Parameter("P", "int", bounds=(1, 128), fixed_value=p),
            Parameter("Q", "int", bounds=(1, 128), fixed_value=q),
        ]
        app.parameters = [Parameter("memory_fraction", "float", bounds=(lo, hi)), *fixed_params]
        return app

    # ------------------------------------------------------------------
    # memory -> N conversion (documented in the class docstring)
    # ------------------------------------------------------------------

    def memory_bytes_to_n(self, memory_bytes: float) -> int:
        """Convert a target memory footprint to the largest fitting ``N``.

        ``memory_bytes ~= N**2 * BYTES_PER_DOUBLE * memory_factor``
        """
        denominator = self.BYTES_PER_DOUBLE * self.memory_factor
        if memory_bytes <= denominator:
            raise ValueError(f"memory_bytes={memory_bytes} too small for an HPL matrix")
        return int(math.sqrt(memory_bytes / denominator))

    def memory_fraction_to_n(self, memory_fraction: float) -> int:
        """Derive the HPL problem size ``N`` from ``memory_fraction``."""
        return self.memory_bytes_to_n(memory_fraction * self.node_memory_bytes)

    def target_memory_bytes(self, memory_fraction: float) -> int:
        return int(memory_fraction * self.node_memory_bytes)

    # ------------------------------------------------------------------
    # configuration resolution
    # ------------------------------------------------------------------

    def resolve_configuration(self, configuration: dict[str, Any]) -> dict[str, Any]:
        """Return the full logged configuration for an evaluation.

        Merges the fixed HPL parameters with the tunable values suggested by
        the optimizer and derives ``N`` (and ``target_memory_bytes``) from
        ``memory_fraction``. In legacy mode the configuration is returned as
        provided.
        """
        resolved: dict[str, Any] = dict(configuration)
        if "memory_fraction" not in resolved:
            return resolved
        memory_fraction = float(resolved["memory_fraction"])
        if not self.memory_fraction_bounds[0] <= memory_fraction <= self.memory_fraction_bounds[1]:
            raise ValueError(
                f"memory_fraction={memory_fraction!r} outside bounds "
                f"{self.memory_fraction_bounds}"
            )
        resolved["target_memory_bytes"] = self.target_memory_bytes(memory_fraction)
        resolved["N"] = self.memory_fraction_to_n(memory_fraction)
        for key, value in self.fixed_hpl_params.items():
            resolved.setdefault(key, value)
        return resolved


    @property
    def executable(self) -> Path:
        if self._executable_override is not None:
            return self._executable_override
        candidates = [
            self.HPL_ROOT / "bin" / "xhpl",
            self.HPL_ROOT / "xhpl",
            self.HPL_ROOT / "src" / "xhpl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return self.HPL_ROOT / "xhpl"

    def command(self, configuration: dict[str, Any]) -> str:
        """Render the bash snippet that stages HPL.dat and runs ``srun xhpl``.

        ``configuration`` must be a *resolved* configuration (see
        :meth:`resolve_configuration`) so that ``N`` is always present.
        """
        n = int(configuration["N"])
        nb = int(configuration.get("NB", 64))
        p = int(configuration.get("P", 1))
        q = int(configuration.get("Q", 1))
        params = self.fixed_hpl_params
        executable = str(self.executable)

        # Stage a per-job HPL.dat on the shared filesystem (under the Slurm
        # submission directory) because srun cannot always chdir into a node-
        # local /tmp path ("couldn't chdir to /tmp/...: going to /tmp instead").
        # HPL reads HPL.dat from its working directory.
        return (
            'RUNDIR="${SLURM_SUBMIT_DIR:-$PWD}/outputs/hpl_runs/${SLURM_JOB_ID:-local}"\n'
            'mkdir -p "$RUNDIR"\n'
            'echo "HPL run dir: $RUNDIR"\n'
            'ln -sf "$RUNDIR/HPL.dat" /tmp/HPL.dat\n'
            "cat > \"$RUNDIR/HPL.dat\" <<'HPLDAT'\n"
            "HPLinpack benchmark input file\n"
            "Innovative Computing Laboratory, University of Tennessee\n"
            "HPL.out      output file name (if any)\n"
            "6            device out (6=stdout,7=stderr,file)\n"
            "1            # of problems sizes (N)\n"
            f"{n}          Ns\n"
            "1            # of NBs\n"
            f"{nb}         NBs\n"
            f"{int(params.get('PMAP', 0))}            PMAP process mapping (0=Row-,1=Column-major)\n"
            "1            # of process grids (P x Q)\n"
            f"{p}          Ps\n"
            f"{q}          Qs\n"
            "16.0         threshold\n"
            "1            # of panel fact\n"
            f"{int(params.get('PFACT', 0))}            PFACTs (0=left, 1=Crout, 2=Right)\n"
            "1            # of recursive stopping criterium\n"
            f"{int(params.get('NBMIN', 4))}            NBMINs (>= 1)\n"
            "1            # of panels in recursion\n"
            f"{int(params.get('NDIV', 4))}            NDIVs\n"
            "1            # of recursive panel fact.\n"
            f"{int(params.get('RFACT', 0))}            RFACTs (0=left, 1=Crout, 2=Right)\n"
            "1            # of broadcast\n"
            f"{int(params.get('BCAST', 1))}            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)\n"
            "1            # of lookahead depth\n"
            f"{int(params.get('DEPTH', 1))}            DEPTHs (>=0)\n"
            f"{int(params.get('SWAP', 0))}            SWAP (0=bin-exch,1=long,2=mix)\n"
            f"{int(params.get('SWAP_THRESH', 64))}           swapping threshold\n"
            f"{int(params.get('L1', 0))}            L1 in (0=transposed,1=no-transposed) form\n"
            f"{int(params.get('U', 1))}            U  in (0=transposed,1=no-transposed) form\n"
            f"{int(params.get('EQUIL', 1))}            Equilibration (0=no,1=yes)\n"
            f"{int(params.get('ALIGN', 8))}            memory alignment in double (> 0)\n"
            "HPLDAT\n"
            'cd "$RUNDIR"\n'
            f"srun {executable}\n"
            'rm -f /tmp/HPL.dat\n'
            'rm -rf "$RUNDIR"\n'
        )

    def parse_result(self, output: str) -> dict[str, Any]:
        metrics: dict[str, float] = {}
        objective = None

        # HPL v2.3 prints one results-table row per (N, NB) problem, e.g.:
        #   WR00C2R2         256    64     1     1               0.01               1.083e-02
        # Keep the last row (the final measurement).
        rows = re.findall(
            r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.eE+-]+)\s*$",
            output,
            flags=re.MULTILINE,
        )
        if rows:
            _, _, _, _, _, runtime, gflops = rows[-1]
            objective = float(gflops)
            metrics["gflops"] = objective
            metrics["runtime"] = float(runtime)
        else:
            # Fallback for HPL variants that emit an explicit tag.
            match = re.search(r"HPLinGFLops\s*[:=]\s*([0-9.eE+-]+)", output, flags=re.IGNORECASE)
            if match:
                objective = float(match.group(1))
                metrics["gflops"] = objective

        return {"metrics": metrics, "objective": objective, "success": objective is not None}
