from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hpc_autotuner.applications.base import Application
from hpc_autotuner.core.parameter import Parameter


class HPLApplication(Application):
    """Adapter for the HPL benchmark in /global/common/software/m4007/opt/hpl-2.3."""

    HPL_ROOT = Path("/global/common/software/m4007/opt/hpl-2.3")

    parameters = [
        Parameter("N", "int", bounds=(64, 4096)),
        Parameter("NB", "int", bounds=(32, 256)),
        Parameter("P", "int", bounds=(1, 32)),
        Parameter("Q", "int", bounds=(1, 32)),
    ]

    @property
    def executable(self) -> Path:
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
        n = int(configuration.get("N", 256))
        nb = int(configuration.get("NB", 64))
        p = int(configuration.get("P", 1))
        q = int(configuration.get("Q", 1))
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
            "0            PMAP process mapping (0=Row-,1=Column-major)\n"
            "1            # of process grids (P x Q)\n"
            f"{p}          Ps\n"
            f"{q}          Qs\n"
            "16.0         threshold\n"
            "1            # of panel fact\n"
            "0            PFACTs (0=left, 1=Crout, 2=Right)\n"
            "1            # of recursive stopping criterium\n"
            "4            NBMINs (>= 1)\n"
            "1            # of panels in recursion\n"
            "4            NDIVs\n"
            "1            # of recursive panel fact.\n"
            "0            RFACTs (0=left, 1=Crout, 2=Right)\n"
            "1            # of broadcast\n"
            "1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)\n"
            "1            # of lookahead depth\n"
            "1            DEPTHs (>=0)\n"
            "0            SWAP (0=bin-exch,1=long,2=mix)\n"
            "64           swapping threshold\n"
            "0            L1 in (0=transposed,1=no-transposed) form\n"
            "1            U  in (0=transposed,1=no-transposed) form\n"
            "1            Equilibration (0=no,1=yes)\n"
            "8            memory alignment in double (> 0)\n"
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
