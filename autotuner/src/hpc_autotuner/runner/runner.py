from __future__ import annotations

import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hpc_autotuner.applications.base import Application
from hpc_autotuner.core.evaluation import Evaluation
from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.core.paths import output_dir, project_root as project_root_fn
from hpc_autotuner.optimizers.base import Optimizer
from hpc_autotuner.schedulers.base import Scheduler
from hpc_autotuner.storage.filesystem import FilesystemStorage


class Runner:
    """Coordinates optimizer, application, scheduler, and storage."""

    def __init__(
        self,
        optimizer: Optimizer,
        application: Application,
        scheduler: Scheduler,
        storage: FilesystemStorage | None = None,
        evaluation_budget: int = 10,
        project_root: str | Path | None = None,
        run_group: str = "default",
        polling_interval: float = 60.0,
    ) -> None:
        self.optimizer = optimizer
        self.application = application
        self.scheduler = scheduler
        root_path = Path(project_root) if project_root is not None else project_root_fn()
        self.project_root = root_path.resolve()
        if storage is not None and getattr(storage, "root", None) is not None:
            inferred_run_group = storage.root.name if storage.root.name not in {"", "."} else run_group
        else:
            inferred_run_group = run_group
        self.run_group = inferred_run_group or "default"
        canonical_root = self.project_root / "outputs" / "autotuning" / self.run_group
        self.storage = storage or FilesystemStorage(root=canonical_root, run_group=self.run_group)
        self.storage.root = canonical_root
        self.evaluation_budget = evaluation_budget
        self.polling_interval = polling_interval
        self._history: list[Evaluation] = []

    def _ensure_outputs(self) -> None:
        self.project_root.mkdir(parents=True, exist_ok=True)
        output_dir_path = self.project_root / "outputs"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "slurm").mkdir(parents=True, exist_ok=True)
        autotuning_dir = output_dir_path / "autotuning"
        autotuning_dir.mkdir(parents=True, exist_ok=True)
        (autotuning_dir / self.run_group).mkdir(parents=True, exist_ok=True)
        if self.storage.root != autotuning_dir / self.run_group:
            self.storage.root.mkdir(parents=True, exist_ok=True)

    def _render_job_script(self, configuration: dict[str, Any], evaluation_id: int, description: str) -> str:
        command = self.application.command(configuration)
        if isinstance(command, str):
            shell_command = command
        else:
            shell_command = " ".join(shlex.quote(part) for part in command)

        script = f'''#!/bin/bash
#SBATCH -J autotune_{evaluation_id}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -q shared
#SBATCH --constraint=cpu
#SBATCH --output=outputs/slurm/job_%j.out
#SBATCH --error=outputs/slurm/job_%j.err

set -euo pipefail

RUN_GROUP="{self.run_group}"
DESCRIPTION="{description}"
RUN_ROOT="${{SLURM_SUBMIT_DIR:-{self.project_root}}}"
OUTPUT="${{RUN_ROOT}}/outputs"
OUTDIR="${{OUTPUT}}/${{RUN_GROUP}}"
OUTFILE="${{OUTDIR}}/${{RUN_GROUP}}_${{SLURM_JOB_ID}}.log"

mkdir -p "$OUTDIR"

echo "$SLURM_JOB_ID : $DESCRIPTION - Using $SLURM_NTASKS tasks and $SLURM_JOB_NUM_NODES nodes" >> "${{OUTPUT}}/descriptions.txt"

echo "Started running ${{RUN_GROUP}} at $(date)" | tee -a "$OUTFILE"
start_time=$(date +%s)

{{
    {shell_command}
}} >> "$OUTFILE" 2>&1

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Finished running ${{RUN_GROUP}} at $(date)" | tee -a "$OUTFILE"
echo "Elapsed time: ${{elapsed}} seconds" | tee -a "$OUTFILE"
'''
        return script

    def _parse_application_output(self, raw_output: str) -> dict[str, Any]:
        if not raw_output.strip():
            return {"metrics": {}, "objective": None, "success": False, "status": "NO_OUTPUT"}
        parsed = self.application.parse_result(raw_output)
        if not isinstance(parsed, dict):
            raise TypeError("Application parse_result() must return a dictionary.")
        return parsed

    def _persist_experiment(self) -> None:
        self.storage.write_experiment_metadata({
            "run_group": self.run_group,
            "evaluation_budget": self.evaluation_budget,
            "polling_interval": self.polling_interval,
            "project_root": str(self.project_root),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    def _create_evaluation(self, configuration: dict[str, Any]) -> Evaluation:
        evaluation_id = len(self._history) + 1
        return Evaluation(evaluation_id=evaluation_id, configuration=dict(configuration))

    def run(self) -> list[Evaluation]:
        self._ensure_outputs()
        self._persist_experiment()

        for _ in range(self.evaluation_budget):
            configuration = self.optimizer.suggest()
            evaluation = self._create_evaluation(configuration)
            description = json.dumps({"configuration": configuration}, sort_keys=True)
            script_text = self._render_job_script(configuration, evaluation.evaluation_id, description)
            script_path = self.project_root / "outputs" / "slurm" / f"job_{evaluation.evaluation_id}.sh"
            script_path.write_text(script_text)

            try:
                job_id = self.scheduler.submit(str(script_path))
            except Exception as exc:  # pragma: no cover - exercised by scheduler failures in integration tests
                evaluation.success = False
                evaluation.status = "FAILED_SUBMISSION"
                evaluation.metadata["submission_error"] = str(exc)
                self.storage.append_evaluation(evaluation.to_dict())
                self._history.append(evaluation)
                self.optimizer.observe(configuration, evaluation.to_dict())
                continue

            evaluation.job_id = job_id
            evaluation.status = "SUBMITTED"
            evaluation.submitted_at = datetime.now(timezone.utc).isoformat()

            try:
                self.scheduler.wait(job_id)
            except Exception as exc:  # pragma: no cover
                evaluation.success = False
                evaluation.status = "WAIT_ERROR"
                evaluation.metadata["wait_error"] = str(exc)
            else:
                evaluation.status = self.scheduler.status(job_id) if hasattr(self.scheduler, "status") else "COMPLETED"
                raw_output = ""
                log_path = self.project_root / "outputs" / self.run_group / f"{self.run_group}_{job_id}.log"
                if log_path.exists():
                    raw_output = log_path.read_text(encoding="utf-8", errors="replace")
                result = self._parse_application_output(raw_output)
                evaluation.metrics = result.get("metrics", {})
                evaluation.objective = result.get("objective")
                evaluation.success = bool(result.get("success", True))
                evaluation.metadata.update({"result": result})
                if evaluation.objective is None and result.get("metrics"):
                    evaluation.objective = next(iter(result["metrics"].values()), None)

            self.storage.append_evaluation(evaluation.to_dict())
            self.optimizer.observe(configuration, evaluation.to_dict())
            self._history.append(evaluation)

        return self._history
