import os
import subprocess
from pathlib import Path

from .base import Scheduler


class SlurmScheduler(Scheduler):

    def __init__(self, project_root=None):
        self.project_root = Path(
            project_root
            or os.environ.get(
                "SLURM_SUBMIT_DIR",
                Path.cwd(),
            )
        ).resolve()

    def submit(
        self,
        script: Path,
        environment: dict[str, str] | None = None,
    ) -> str:

        command = [
            "sbatch",
            "--parsable",
        ]

        if environment:
            for key, value in environment.items():
                command.extend(
                    ["--export", f"{key}={value}"]
                )

        command.append(str(script))

        result = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=True,
        )

        job_id = result.stdout.strip()

        # Some Slurm configurations return:
        # 123456
        # or
        # 123456;cluster
        return job_id.split(";")[0]

    def wait(self, job_id: str) -> None:

        while True:

            result = subprocess.run(
                [
                    "squeue",
                    "-j",
                    job_id,
                    "-h",
                    "-o",
                    "%T",
                ],
                capture_output=True,
                text=True,
            )

            state = result.stdout.strip()

            if not state:
                return

            print(
                f"Job {job_id}: {state}",
                flush=True,
            )

            import time
            time.sleep(2)