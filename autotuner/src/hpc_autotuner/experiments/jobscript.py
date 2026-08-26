"""Render Slurm job scripts from the shared template.

The template lives in ``src/hpc_autotuner/resources/slurm/job.sh`` and is the
single source of truth for the *child* HPL job wrapper: it creates the run
directory, writes the run-group log, appends to ``outputs/descriptions.txt``,
executes the application command, and saves stdout/stderr via the ``#SBATCH
--output/--error`` directives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hpc_autotuner.experiments.config import SlurmConfig

_TEMPLATE = Path(__file__).resolve().parents[1] / "resources" / "slurm" / "job.sh"


def render_job_script(
    *,
    slurm: SlurmConfig,
    job_name: str,
    run_group: str,
    description: str,
    command: str,
    template: str | Path | None = None,
) -> str:
    """Render the child job script for one evaluation."""
    source = Path(template) if template else _TEMPLATE
    text = source.read_text(encoding="utf-8")

    def line(directive: str, value: str | None) -> str:
        return f"#SBATCH {directive}={value}\n" if value else ""

    account_line = line("--account", slurm.account)
    partition_line = line("--partition", slurm.partition)
    time_line = line("--time", slurm.time)
    exclusive_line = "#SBATCH --exclusive\n" if slurm.exclusive else ""
    cpus_line = line("--cpus-per-task", str(slurm.cpus_per_task)) if slurm.cpus_per_task else ""

    replacements = {
        "__JOB_NAME__": job_name,
        "__NODES__": str(slurm.nodes),
        "__NTASKS__": str(slurm.ntasks),
        "__QOS__": slurm.qos,
        "__CONSTRAINT__": slurm.constraint,
        "__ACCOUNT_LINE__": account_line,
        "__PARTITION_LINE__": partition_line,
        "__TIME_LINE__": time_line,
        "__EXCLUSIVE_LINE__": exclusive_line + cpus_line,
        "__RUN_GROUP__": run_group,
        "__DESCRIPTION__": description,
        "__COMMAND__": command,
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)

    if "__" in text.replace("__COMMAND__", ""):  # pragma: no cover - defensive
        leftover = [token for token in text.split() if token.startswith("__") and token.endswith("__")]
        raise RuntimeError(f"Unfilled template placeholders: {sorted(set(leftover))}")
    return text


def build_child_script(
    *,
    slurm: SlurmConfig,
    run_group: str,
    attempt: int,
    configuration: dict[str, Any],
    command: str,
    template: str | Path | None = None,
) -> str:
    """Convenience wrapper used by the experiment controller."""
    description = _describe(configuration)
    return render_job_script(
        slurm=slurm,
        job_name=f"hpl_a{attempt}",
        run_group=run_group,
        description=description,
        command=command,
        template=template,
    )


def _describe(configuration: dict[str, Any]) -> str:
    import json

    return json.dumps({"configuration": configuration}, sort_keys=True)
