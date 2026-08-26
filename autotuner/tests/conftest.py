"""Shared fixtures and mock infrastructure for the unit tests.

The mock scheduler simulates the Slurm contract without ever invoking sbatch
or bash: submitting a rendered job script returns a fake job id and (unless
configured otherwise) writes a synthetic HPL result log into the expected
``outputs/<run_group>/<run_group>_<job_id>.log`` location so the experiment
controller's parse step behaves exactly as it does on a cluster.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hpc_autotuner.experiments.config import SlurmConfig
from hpc_autotuner.schedulers.base import Scheduler


def fake_hpl_output(
    n: int = 1024,
    nb: int = 192,
    p: int = 1,
    q: int = 1,
    gflops: float = 2.5,
    runtime: float = 0.35,
) -> str:
    """Synthetic xhpl 2.3 results-table output."""
    return (
        "================================================================================\n"
        "T/V                N    NB     P     Q               Time                 Gflops\n"
        "--------------------------------------------------------------------------------\n"
        f"WR11L4L4       {n:>7} {nb:>5} {p:>5} {q:>5}           {runtime:>9.2f}    {gflops:>14.6e}\n"
        "--------------------------------------------------------------------------------\n"
        "||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=   1.11537336e-02 ...... PASSED\n"
        "Finished      1 tests with the following results:\n"
        "              1 tests completed and passed residual checks,\n"
    )


class MockScheduler(Scheduler):
    """Deterministic in-process scheduler for unit tests.

    ``job_id`` values are deterministic per scheduler instance. On submit, a
    synthetic HPL output log is written so the controller can parse it.
    Configure failure behavior with ``fail_submissions`` (number of submits to
    reject) and ``fail_result_submits`` (1-based submit indices whose result
    log is unparseable).
    """

    def __init__(
        self,
        *,
        fail_submissions: int = 0,
        fail_result_submits: set[int] | None = None,
        status_after_wait: str = "COMPLETED",
        job_id_prefix: str = "42",
    ) -> None:
        self.fail_submissions = fail_submissions
        self.fail_result_submits = set(fail_result_submits or set())
        self.status_after_wait = status_after_wait
        self.job_id_prefix = job_id_prefix
        self._counter = 0
        self.submitted_scripts: list[str] = []
        self.job_ids: list[str] = []
        self.states: dict[str, str] = {}

    def submit(self, script: str | Path, environment: dict[str, str] | None = None) -> str:
        script_path = Path(script)
        if len(self.submitted_scripts) < self.fail_submissions:
            self.submitted_scripts.append(str(script))
            raise RuntimeError(f"synthetic sbatch failure for {script_path.name}")
        self._counter += 1
        job_id = f"{self.job_id_prefix}{self._counter}"
        self.submitted_scripts.append(str(script_path))
        self.job_ids.append(job_id)
        self.states[job_id] = "RUNNING"

        # Simulate the child job writing its run-group log. The controller
        # looks for <outputs>/<run_group>/<run_group>_<job_id>.log, i.e. one
        # level above the slurm/ scripts directory.
        run_group = self._run_group_from_script(script_path)
        outfile = script_path.parent.parent / run_group / f"{run_group}_{job_id}.log"
        outfile.parent.mkdir(parents=True, exist_ok=True)
        if self._counter in self.fail_result_submits:
            outfile.write_text("no results table here\n", encoding="utf-8")
        else:
            outfile.write_text(fake_hpl_output(gflops=2.5 + self._counter * 0.1), encoding="utf-8")
        return job_id

    @staticmethod
    def _run_group_from_script(script_path: Path) -> str:
        text = script_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("RUN_GROUP="):
                return line.split("=", 1)[1].strip().strip('"')
        return "default"

    def wait(self, job_id: str) -> None:
        self.states[job_id] = self.status_after_wait

    def status(self, job_id: str) -> str:
        return self.states.get(job_id, self.status_after_wait)


@pytest.fixture
def slurm_config() -> SlurmConfig:
    return SlurmConfig(ntasks=1, polling_interval=0.0)


@pytest.fixture
def mock_scheduler():
    return MockScheduler()


def write_experiment_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
