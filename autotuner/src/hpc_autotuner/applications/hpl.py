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

    def command(self, configuration: dict[str, Any]) -> list[str]:
        n = int(configuration.get("N", 256))
        nb = int(configuration.get("NB", 128))
        p = int(configuration.get("P", 1))
        q = int(configuration.get("Q", 1))
        executable = str(self.executable)
        return [
            "bash",
            "-lc",
            (
                f"cd {self.HPL_ROOT} && "
                f"cp -f HPL.dat HPL.dat.bak 2>/dev/null || true && "
                f"python - <<'PY'\n"
                f"from pathlib import Path\n"
                f"path = Path('HPL.dat')\n"
                f"path.write_text('''HPLinpack benchmark input file\n'\n"
                f"'Innovative Computing Laboratory, University of Tennessee.\n'\n"
                f"'\n'\n"
                f"'  1  1  1  1\n'\n"
                f"'  1  1  1\n'\n"
                f"'  1\n'\n"
                f"'  1\n'\n"
                f"'  1\n'\n"
                f"'\n'\n"
                f"f'{{n}}\n'\n"
                f"f'{{nb}}\n'\n"
                f"f'{{p}}\n'\n"
                f"f'{{q}}\n'\n"
                f"'\n'\n"
                f"'\n'\n"
                f"'\n'\n" 
                f"'\n'\n"
                f"'\n'\n"
                f"'\n'\n"
                f"'\n'\n"
                f"''')\n"
                f"PY\n"
                f"{executable}"
            ),
        ]

    def parse_result(self, output: str) -> dict[str, Any]:
        metrics: dict[str, float] = {}
        objective = None

        match = re.search(r"HPLinGFLops\s*[:=]\s*([0-9.]+)", output, flags=re.IGNORECASE)
        if match:
            objective = float(match.group(1))
            metrics["gflops"] = objective

        match = re.search(r"time\s*[:=]\s*([0-9.]+)", output, flags=re.IGNORECASE)
        if match:
            metrics["runtime"] = float(match.group(1))

        return {"metrics": metrics, "objective": objective, "success": objective is not None}
