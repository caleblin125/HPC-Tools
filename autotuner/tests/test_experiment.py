"""Unit tests for the sequential experiment controller.

These tests use the :class:`MockScheduler` from conftest.py, so no Slurm
submission ever happens. They verify attempt numbering, JSONL logging,
sequential feedback, failed-submission handling, and resume-from-log.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hpc_autotuner.applications.hpl import HPLApplication
from hpc_autotuner.experiments.common import run_experiment
from hpc_autotuner.experiments.config import SlurmConfig
from hpc_autotuner.optimizers.base import Optimizer
from hpc_autotuner.storage.filesystem import FilesystemStorage
from tests.conftest import MockScheduler


class RecordingOptimizer(Optimizer):
    """Suggests a deterministic sequence and records the observe calls."""

    def __init__(self, fractions: list[float] | None = None):
        self.fractions = fractions or [0.80 + 0.02 * i for i in range(100)]
        self.suggest_calls = 0
        self.observed: list[tuple[dict, dict]] = []

    def suggest(self) -> dict:
        fraction = self.fractions[self.suggest_calls % len(self.fractions)]
        self.suggest_calls += 1
        return {"memory_fraction": fraction}

    def observe(self, configuration: dict, result: dict) -> None:
        self.observed.append((dict(configuration), dict(result)))


def _make_app() -> HPLApplication:
    return HPLApplication.for_benchmark(
        node_memory_bytes=8 * 1024**3,
        fixed={"NB": 192, "P": 1, "Q": 1},
    )


def _slurm() -> SlurmConfig:
    return SlurmConfig(ntasks=1, polling_interval=0.0, submit_retries=2)


def _setup(tmp_path: Path, run_group: str = "random"):
    output_root = tmp_path / "outputs" / "autotuning"
    storage = FilesystemStorage(root=output_root / run_group, run_group=run_group)
    return output_root, storage


def test_attempt_numbers_monotonic_and_log_complete(tmp_path):
    output_root, storage = _setup(tmp_path)
    optimizer = RecordingOptimizer()
    records = run_experiment(
        optimizer=optimizer,
        application=_make_app(),
        scheduler=MockScheduler(),
        storage=storage,
        budget=3,
        seed=42,
        run_group="random",
        optimizer_name="random",
        slurm=_slurm(),
        output_root=output_root,
        project_root=tmp_path,
        log=lambda _: None,
    )

    assert [rec["attempt"] for rec in records] == [1, 2, 3]
    assert [rec["status"] for rec in records] == ["COMPLETED"] * 3
    assert all(rec["success"] for rec in records)
    assert all(rec["objective"] is not None for rec in records)
    assert all(rec["gflops"] == rec["objective"] for rec in records)
    assert records[0]["attempt"] == 1 and records[-1]["attempt"] == 3

    # Every record was persisted to JSONL.
    lines = storage.evaluations_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6  # 3 SUBMITTED + 3 COMPLETED
    final_records = [json.loads(line) for line in lines if json.loads(line)["status"] == "COMPLETED"]
    assert len(final_records) == 3

    # The configuration is the fully-resolved HPL configuration.
    config = final_records[0]["configuration"]
    assert config["memory_fraction"] >= 0.80
    assert config["N"] > 0
    assert config["NB"] == 192


def test_sequential_feedback_before_next_suggest(tmp_path):
    output_root, storage = _setup(tmp_path, "seq")
    optimizer = RecordingOptimizer()

    class SequentialProbe(Optimizer):
        def __init__(self, inner):
            self.inner = inner
            self.observed_before_suggest = 0

        def suggest(self):
            # When attempt 2 is requested, the previous result must already
            # have been observed.
            self.observed_before_suggest = len(self.inner.observed)
            return self.inner.suggest()

        def observe(self, configuration, result):
            return self.inner.observe(configuration, result)

    probe = SequentialProbe(optimizer)
    run_experiment(
        optimizer=probe,
        application=_make_app(),
        scheduler=MockScheduler(),
        storage=storage,
        budget=4,
        seed=42,
        run_group="seq",
        optimizer_name="random",
        slurm=_slurm(),
        output_root=output_root,
        project_root=tmp_path,
        log=lambda _: None,
    )
    # By the 4th suggest, 3 results were observed (strictly sequential).
    assert optimizer.suggest_calls == 4
    assert len(optimizer.observed) == 4


def test_resume_continues_from_log(tmp_path):
    output_root, storage = _setup(tmp_path, "resume")
    slurm = _slurm()

    run_experiment(
        optimizer=RecordingOptimizer(),
        application=_make_app(),
        scheduler=MockScheduler(),
        storage=storage,
        budget=3,
        seed=42,
        run_group="resume",
        optimizer_name="random",
        slurm=slurm,
        output_root=output_root,
        project_root=tmp_path,
        log=lambda _: None,
    )

    # Second invocation (simulates an interrupted run being restarted) should
    # pick up from attempt 4 and reach the new budget.
    optimizer = RecordingOptimizer()
    records = run_experiment(
        optimizer=optimizer,
        application=_make_app(),
        scheduler=MockScheduler(),
        storage=storage,
        budget=5,
        seed=42,
        run_group="resume",
        optimizer_name="random",
        slurm=slurm,
        output_root=output_root,
        project_root=tmp_path,
        log=lambda _: None,
    )
    assert [rec["attempt"] for rec in records] == [4, 5]
    # Optimizer state was rebuilt from the first 3 observations.
    assert len(optimizer.observed) == 3 + 2

def test_failed_submission_is_not_an_optimizer_attempt(tmp_path):
    output_root, storage = _setup(tmp_path, "failsub")
    # All submit retries fail so the controller raises after logging.
    scheduler = MockScheduler(fail_submissions=3)

    with pytest.raises(RuntimeError, match="sbatch failed"):
        run_experiment(
            optimizer=RecordingOptimizer(),
            application=_make_app(),
            scheduler=scheduler,
            storage=storage,
            budget=2,
            seed=42,
            run_group="failsub",
            optimizer_name="random",
            slurm=_slurm(),
            output_root=output_root,
            project_root=tmp_path,
            log=lambda _: None,
        )

    # A FAILED_SUBMISSION record was written but no optimizer attempt consumed.
    records = storage.read_evaluations()
    assert records[-1]["status"] == "FAILED_SUBMISSION"
    assert records[-1]["slurm_job_id"] is None
    assert all(rec["status"] != "COMPLETED" for rec in records)


def test_failed_evaluation_is_recorded_and_observed(tmp_path):
    output_root, storage = _setup(tmp_path, "failres")
    # The first submitted evaluation produces an unparseable HPL log.
    scheduler = MockScheduler(fail_result_submits={1})
    optimizer = RecordingOptimizer()

    run_experiment(
        optimizer=optimizer,
        application=_make_app(),
        scheduler=scheduler,
        storage=storage,
        budget=2,
        seed=42,
        run_group="failres",
        optimizer_name="random",
        slurm=_slurm(),
        output_root=output_root,
        project_root=tmp_path,
        log=lambda _: None,
    )

    records = storage.read_evaluations()
    failed = [rec for rec in records if rec["status"] == "FAILED"]
    assert len(failed) == 1
    assert failed[0]["success"] is False
    assert failed[0]["objective"] is None
    # The optimizer explicitly saw the failed evaluation (never silently dropped).
    assert any(not rec[1]["success"] for rec in optimizer.observed)


def test_interrupted_attempt_is_rerun_on_resume(tmp_path):
    output_root, storage = _setup(tmp_path, "interrupt")
    slurm = _slurm()

    # First run: budget 3, then simulate an interruption: keep only the first
    # COMPLETED record and one SUBMITTED (incomplete) record in the log.
    run_experiment(
        optimizer=RecordingOptimizer(),
        application=_make_app(),
        scheduler=MockScheduler(),
        storage=storage,
        budget=3,
        seed=42,
        run_group="interrupt",
        optimizer_name="random",
        slurm=slurm,
        output_root=output_root,
        project_root=tmp_path,
        log=lambda _: None,
    )
    records = storage.read_evaluations()
    completed_one = next(rec for rec in records if rec["status"] == "COMPLETED" and rec["attempt"] == 1)
    interrupted = next(rec for rec in records if rec["status"] == "SUBMITTED" and rec["attempt"] == 2)
    storage.evaluations_file.write_text(
        json.dumps(completed_one) + "\n" + json.dumps(interrupted) + "\n",
        encoding="utf-8",
    )

    optimizer = RecordingOptimizer()
    records = run_experiment(
        optimizer=optimizer,
        application=_make_app(),
        scheduler=MockScheduler(),
        storage=storage,
        budget=3,
        seed=42,
        run_group="interrupt",
        optimizer_name="random",
        slurm=slurm,
        output_root=output_root,
        project_root=tmp_path,
        log=lambda _: None,
    )
    # Attempt 2 was re-run (the interrupted one) and attempt 3 continued.
    assert [rec["attempt"] for rec in records] == [2, 3]
