import os
import subprocess
import time
from pathlib import Path

from .base import Scheduler


class SlurmScheduler(Scheduler):
    """Standard Slurm scheduler wrapper supporting submission and status polling."""

    def __init__(self, project_root: str | Path | None = None, polling_interval: float = 2.0):
        self.project_root = Path(project_root or os.environ.get("SLURM_SUBMIT_DIR", Path.cwd())).resolve()
        self.polling_interval = polling_interval

    def submit(
        self,
        script: str | Path,
        environment: dict[str, str] | None = None,
    ) -> str:
        command = ["sbatch", "--parsable"]
        if environment:
            for key, value in environment.items():
                command.extend(["--export", f"{key}={value}"])
        command.append(str(script))

        result = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")

        job_id = result.stdout.strip()
        return job_id.split(";")[0]

    def wait(self, job_id: str) -> None:
        while True:
            state = self.status(job_id)
            if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "BOOT_FAIL", "NODE_FAIL"}:
                return
            time.sleep(self.polling_interval)

    def status(self, job_id: str) -> str:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            check=False,
        )
        state = result.stdout.strip()
        if state:
            return state.upper()

        sacct = subprocess.run(
            ["sacct", "-j", job_id, "-n", "-o", "State%20"],
            capture_output=True,
            text=True,
            check=False,
        )
        if sacct.returncode == 0:
            lines = [line.strip() for line in sacct.stdout.splitlines() if line.strip()]
            if lines:
                return lines[-1].split()[0].upper() if lines[-1].split() else "UNKNOWN"
        return "COMPLETED"
