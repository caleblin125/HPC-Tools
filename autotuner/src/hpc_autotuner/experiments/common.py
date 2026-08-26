"""Common sequential experiment controller.

This is the single execution loop shared by all six optimizer drivers. The
flow for every attempt is::

    config = optimizer.suggest()
    resolve/validate configuration
    render child Slurm script
    submit -> wait -> parse GFLOPs
    write JSONL evaluation record
    optimizer.observe(config, record)

Attempts run strictly sequentially: the optimizer sees the result of attempt
N before it is asked for attempt N+1. Experiments are resumable from their
``evaluations.jsonl`` log.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hpc_autotuner.applications.base import Application
from hpc_autotuner.applications.hpl import HPLApplication
from hpc_autotuner.core.evaluation import Evaluation
from hpc_autotuner.core.space import ParameterSpace
from hpc_autotuner.experiments.config import ExperimentConfig
from hpc_autotuner.experiments.jobscript import build_child_script
from hpc_autotuner.optimizers import OPTIMIZERS
from hpc_autotuner.optimizers.base import Optimizer
from hpc_autotuner.schedulers.base import Scheduler
from hpc_autotuner.storage.filesystem import FilesystemStorage

#: Slurm job states that mean the job finished (terminal).
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "BOOT_FAIL",
    "NODE_FAIL",
    "PREEMPTED",
    "DEADLINE",
}

Logger = Callable[[str], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


#: Factory that builds an :class:`Application` from an experiment config.
ApplicationFactory = Callable[[ExperimentConfig], Application]

#: Registered application types, keyed by ``config.application_type``.
_APPLICATION_FACTORIES: dict[str, ApplicationFactory] = {}


def register_application(kind: str, factory: ApplicationFactory) -> None:
    """Register a new application type so it can be selected from an experiment
    YAML via ``application.type: <kind>``.

    This is the extension point for using the framework with a new workload
    (e.g. compiler-flag tuning, a runtime-knob benchmark, ...). Subclass
    :class:`~hpc_autotuner.applications.base.Application`, then::

        register_application("my_app", my_app_factory)
    """
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("application kind must be a non-empty string")
    if not callable(factory):
        raise TypeError("application factory must be callable")
    _APPLICATION_FACTORIES[kind] = factory


def _build_hpl(config: ExperimentConfig) -> Application:
    return HPLApplication.for_benchmark(
        executable=config.executable,
        node_memory_bytes=config.node_memory_bytes,
        memory_factor=config.memory_factor,
        memory_fraction_bounds=config.memory_fraction_bounds,
        ntasks=config.slurm.ntasks,
        tunable=config.tunable,
        fixed=config.fixed,
    )


def _build_compile_flags(config: ExperimentConfig) -> Application:
    from hpc_autotuner.applications.compile_flags import CompileFlagsApplication

    return CompileFlagsApplication.from_config(config)


#: Application types available out of the box.
register_application("hpl", _build_hpl)
register_application("compile_flags", _build_compile_flags)


def build_application(config: ExperimentConfig) -> Application:
    """Build the application named by ``config.application_type``.

    Register new applications with :func:`register_application`; see also the
    bundled :mod:`hpc_autotuner.applications.compile_flags` example.
    """
    factory = _APPLICATION_FACTORIES.get(config.application_type)
    if factory is None:
        raise ValueError(
            f"Unsupported application type {config.application_type!r}; "
            f"registered types: {sorted(_APPLICATION_FACTORIES)}"
        )
    return factory(config)


def build_optimizer(
    optimizer_name: str,
    parameters: list[Any],
    *,
    seed: int | None = None,
    direction: str = "maximize",
    n_trials: int = 100,
) -> Optimizer:
    """Construct the optimizer adapter for ``optimizer_name``."""
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {optimizer_name!r}; choose from {sorted(OPTIMIZERS)}")
    cls = OPTIMIZERS[optimizer_name]
    return cls(parameters, seed=seed, direction=direction, n_trials=n_trials)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _query_slurm_times(job_id: str) -> dict[str, Any] | None:
    """Best-effort Submit/Start/End timestamps from ``sacct``."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "-n", "-P", "-o", "Submit,Start,End"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3:
            continue
        submit, start, end = parts
        if submit in {"", "Unknown"} or start in {"", "Unknown"} or end in {"", "Unknown"}:
            continue

        def parse(iso: str) -> datetime | None:
            try:
                return datetime.fromisoformat(iso)
            except ValueError:
                return None

        submit_dt, start_dt, end_dt = parse(submit), parse(start), parse(end)
        if not (submit_dt and start_dt and end_dt):
            continue
        return {
            "submit": submit,
            "start": start,
            "end": end,
            "queue_time": (start_dt - submit_dt).total_seconds(),
            "compute_time": (end_dt - start_dt).total_seconds(),
        }
    return None


def _submit_with_retries(
    scheduler: Scheduler, script_path: Path, *, retries: int, log: Logger
) -> str:
    last_error: Exception | None = None
    for attempt_no in range(1, retries + 1):
        try:
            return scheduler.submit(str(script_path))
        except Exception as exc:
            last_error = exc
            if attempt_no < retries:
                backoff = 5 * attempt_no
                log(f"    sbatch failed (attempt {attempt_no}/{retries}): {exc}; retrying in {backoff}s")
                time.sleep(backoff)
    assert last_error is not None
    raise RuntimeError(f"sbatch failed after {retries} attempts: {last_error}") from last_error

# ---------------------------------------------------------------------------
# the experiment loop
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    optimizer: Optimizer,
    application: Application,
    scheduler: Scheduler,
    storage: FilesystemStorage,
    budget: int,
    seed: int,
    run_group: str,
    optimizer_name: str,
    slurm: Any,
    output_root: Path,
    project_root: Path,
    log: Logger | None = None,
) -> list[dict[str, Any]]:
    """Run up to ``budget`` sequential evaluations and return all records."""
    log = log or (lambda message: print(message, flush=True))
    if budget < 1:
        raise ValueError("budget must be >= 1")

    # -- output layout ----------------------------------------------------
    slurm_out_dir = output_root.parent / "slurm"
    slurm_out_dir.mkdir(parents=True, exist_ok=True)
    run_group_dir = output_root.parent / run_group
    run_group_dir.mkdir(parents=True, exist_ok=True)
    storage.root.mkdir(parents=True, exist_ok=True)

    space = ParameterSpace(application.tunable_parameters)

    # -- resume from existing log ------------------------------------------
    existing = storage.read_evaluations()
    # The log contains transient SUBMITTED records plus the final record per
    # attempt; only the *last* record per attempt describes its state.
    by_attempt: dict[int, dict[str, Any]] = {}
    for rec in existing:
        by_attempt[int(rec.get("attempt", 0))] = rec
    terminal = [by_attempt[a] for a in sorted(by_attempt) if by_attempt[a].get("status") in TERMINAL_STATES]
    incomplete = next(
        (
            by_attempt[a]
            for a in sorted(by_attempt)
            if a > 0
            and by_attempt[a].get("job_id")
            and by_attempt[a].get("status") not in TERMINAL_STATES
        ),
        None,
    )

    resume_config: dict[str, Any] | None = None
    if incomplete is not None:
        start_attempt = int(incomplete["attempt"])
        resume_config = incomplete["configuration"]
        replay = [rec for rec in terminal if rec["attempt"] != start_attempt]
    else:
        start_attempt = max((int(rec["attempt"]) for rec in terminal if rec.get("attempt")), default=0) + 1
        replay = terminal

    # Rebuild optimizer state from history (all adapters learn via observe()).
    for record in replay:
        if "configuration" in record:
            optimizer.observe(record["configuration"], record)

    if resume_config is not None:
        log(f"[resume] continuing from attempt {start_attempt} with a re-run of the interrupted evaluation.")
    elif start_attempt > 1:
        log(f"[resume] continuing from attempt {start_attempt} ({len(terminal)} completed records found).")

    # -- metadata -----------------------------------------------------------
    storage.write_experiment_metadata(_build_metadata(
        optimizer_name=optimizer_name,
        run_group=run_group,
        seed=seed,
        budget=budget,
        application=application,
        slurm=slurm,
        space=space,
        created_at=_now(),
        resumed_from=len(existing),
    ))


    # -- main loop -----------------------------------------------------------
    records: list[dict[str, Any]] = []
    attempt = start_attempt
    while attempt <= budget:
        if resume_config is not None:
            tunable = resume_config
            resume_config = None
        else:
            tunable = optimizer.suggest()

        config = application.resolve_configuration(dict(tunable))
        space.validate(config)
        _validate_grid(config, slurm)

        command = application.command(config)
        if isinstance(command, list):
            command = " ".join(shlex.quote(part) for part in command)

        script_text = build_child_script(
            slurm=slurm,
            run_group=run_group,
            attempt=attempt,
            configuration=config,
            command=command,
        )
        script_path = slurm_out_dir / f"job_attempt_{attempt}.sh"
        script_path.write_text(script_text, encoding="utf-8")

        log(f"[attempt {attempt}/{budget}] submitting {optimizer_name} config: "
            f"{json.dumps(config, sort_keys=True)}")
        try:
            job_id = _submit_with_retries(scheduler, script_path, retries=slurm.submit_retries, log=log)
        except Exception as exc:
            record = {
                "attempt": attempt,
                "evaluation_id": attempt,
                "optimizer": optimizer_name,
                "run_group": run_group,
                "configuration": config,
                "slurm_job_id": None,
                "status": "FAILED_SUBMISSION",
                "success": False,
                "objective": None,
                "metrics": {},
                "error": str(exc),
                "submitted_at": _now(),
            }
            storage.append_evaluation(record)
            log(f"[attempt {attempt}] SUBMISSION FAILED: {exc}")
            raise  # resumable: the config was not consumed by the optimizer
        # NOTE: attempt numbers are assigned only after a successful sbatch.

        evaluation = Evaluation(
            evaluation_id=attempt,
            attempt=attempt,
            optimizer=optimizer_name,
            run_group=run_group,
            configuration=config,
            job_id=job_id,
            status="SUBMITTED",
            submitted_at=_now(),
        )
        storage.append_evaluation(evaluation.to_dict())

        wait_started = time.monotonic()
        try:
            scheduler.wait(job_id)
        except Exception as exc:  # pragma: no cover - scheduler failure path
            evaluation.status = "WAIT_ERROR"
            evaluation.success = False
            evaluation.error = str(exc)
        else:
            evaluation.status = scheduler.status(job_id)
        wall_seconds = time.monotonic() - wait_started


        # Timestamps / timing from sacct when available.
        timings = _query_slurm_times(job_id)
        if timings:
            evaluation.queue_time = timings["queue_time"]
            evaluation.compute_time = timings["compute_time"]
            evaluation.started_at = timings["start"]
        evaluation.finished_at = _now()

        # Parse the application log written by the child job.
        log_path = run_group_dir / f"{run_group}_{job_id}.log"
        raw_output = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        parsed = application.parse_result(raw_output) if raw_output else {}
        if not isinstance(parsed, dict):
            parsed = {}
        metrics = parsed.get("metrics") or {}
        objective = parsed.get("objective")
        parse_ok = bool(parsed.get("success", False)) and objective is not None

        evaluation.metrics = metrics
        evaluation.objective = objective
        evaluation.success = parse_ok and evaluation.status == "COMPLETED"
        evaluation.elapsed_seconds = (
            float(metrics.get("runtime", wall_seconds))
            if metrics.get("runtime") is not None
            else wall_seconds
        )
        if not parse_ok:
            evaluation.error = "HPL output did not contain a parseable result"
        evaluation.metadata["output_log"] = str(log_path) if log_path.exists() else None
        evaluation.metadata["wall_seconds"] = round(wall_seconds, 3)
        # A completed Slurm job with an unparseable application log is a FAILED
        # evaluation; a job that itself failed keeps its Slurm terminal state.
        if not parse_ok and evaluation.status == "COMPLETED":
            evaluation.status = "FAILED"
        if evaluation.status in TERMINAL_STATES and evaluation.status != "COMPLETED":
            evaluation.success = False
        if evaluation.status not in TERMINAL_STATES:
            evaluation.status = "FAILED"

        record = evaluation.to_dict()
        record["gflops"] = metrics.get("gflops", objective)
        record["runtime_seconds"] = evaluation.elapsed_seconds
        storage.append_evaluation(record)
        records.append(record)

        optimizer.observe(config, record)
        log(
            f"[attempt {attempt}/{budget}] job={job_id} status={evaluation.status} "
            f"success={evaluation.success} objective={evaluation.objective} "
            f"queue={evaluation.queue_time}s compute={evaluation.compute_time}s"
        )
        attempt += 1

    storage.write_experiment_metadata(_build_metadata(
        optimizer_name=optimizer_name,
        run_group=run_group,
        seed=seed,
        budget=budget,
        application=application,
        slurm=slurm,
        space=space,
        created_at=None,
        completed_at=_now(),
        resumed_from=len(existing),
    ))
    log(f"[done] experiment '{run_group}' completed {len(records)} evaluations.")
    return records


def run_configured_experiment(
    optimizer_name: str,
    config: ExperimentConfig,
    *,
    budget: int | None = None,
    seed: int | None = None,
    run_group: str | None = None,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    log: Logger | None = None,
) -> list[dict[str, Any]]:
    """Wire the optimizer, application, scheduler, and storage from a config."""
    from hpc_autotuner.schedulers.slurm import SlurmScheduler

    log = log or (lambda message: print(message, flush=True))
    if budget is not None:
        config.budget = budget
    if seed is not None:
        config.seed = seed
    if run_group:
        config.run_group = run_group
    if optimizer_name:
        config.optimizer = optimizer_name

    root = Path(project_root) if project_root else config.resolved_project_root()
    output = Path(output_root) if output_root else config.resolved_output_root()
    run_group = config.run_group or config.optimizer

    log(f"project_root={root}")
    log(f"output_root={output}")
    log(f"optimizer={config.optimizer} budget={config.budget} seed={config.seed}")

    application = build_application(config)
    optimizer = build_optimizer(
        config.optimizer,
        application.tunable_parameters,
        seed=config.seed,
        direction=config.objective_direction,
        n_trials=config.budget,
    )
    scheduler = SlurmScheduler(project_root=root, polling_interval=config.slurm.polling_interval)
    storage = FilesystemStorage(root=output / run_group, run_group=run_group)

    return run_experiment(
        optimizer=optimizer,
        application=application,
        scheduler=scheduler,
        storage=storage,
        budget=config.budget,
        seed=config.seed,
        run_group=run_group,
        optimizer_name=config.optimizer,
        slurm=config.slurm,
        output_root=output,
        project_root=root,
        log=log,
    )


def _validate_grid(config: dict[str, Any], slurm: Any) -> None:
    """Ensure the HPL process grid matches the Slurm task count."""
    if "P" in config and "Q" in config:
        grid = int(config["P"]) * int(config["Q"])
        if grid != int(slurm.ntasks):
            raise ValueError(
                f"HPL grid P x Q = {config['P']} x {config['Q']} = {grid} does not match "
                f"the Slurm allocation of {slurm.ntasks} tasks"
            )


def _build_metadata(
    *,
    optimizer_name: str,
    run_group: str,
    seed: int,
    budget: int,
    application: Application,
    slurm: Any,
    space: ParameterSpace,
    created_at: str | None,
    completed_at: str | None = None,
    resumed_from: int = 0,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "run_group": run_group,
        "optimizer": optimizer_name,
        "seed": seed,
        "budget": budget,
        "resumed_from_records": resumed_from,
        "objective": {
            "metric": getattr(application, "objective_metric", "gflops"),
            "direction": getattr(application, "objective_direction", "maximize"),
        },
        "parameter_space": space.to_dict(),
        "application": {
            "type": application.__class__.__name__,
            "parameters": [p.to_dict() for p in application.parameters],
        },
        "slurm": {
            "nodes": slurm.nodes,
            "ntasks": slurm.ntasks,
            "qos": slurm.qos,
            "constraint": slurm.constraint,
            "account": slurm.account,
            "partition": slurm.partition,
            "time": slurm.time,
            "exclusive": slurm.exclusive,
        },
    }
    if hasattr(application, "node_memory_bytes"):
        metadata["application"]["node_memory_bytes"] = application.node_memory_bytes
        metadata["application"]["memory_factor"] = application.memory_factor
        metadata["application"]["memory_fraction_bounds"] = list(application.memory_fraction_bounds)
    if created_at:
        metadata["created_at"] = created_at
    if completed_at:
        metadata["completed_at"] = completed_at
    return metadata


def driver_main(optimizer_name: str, argv: list[str] | None = None) -> int:
    """Shared CLI entry point for the six optimizer drivers."""
    import argparse

    parser = argparse.ArgumentParser(description=f"Run the {optimizer_name} HPL tuning experiment.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--budget", type=int, default=None, help="Override the evaluation budget.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument("--run-group", default=None, help="Override the output run group.")
    parser.add_argument("--output-root", default=None, help="Override the outputs directory.")
    parser.add_argument("--project-root", default=None, help="Override the project root.")
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    try:
        run_configured_experiment(
            optimizer_name,
            config,
            budget=args.budget,
            seed=args.seed,
            run_group=args.run_group,
            project_root=args.project_root,
            output_root=args.output_root,
        )
    except Exception as exc:  # noqa: BLE001 - top-level driver reports and exits non-zero
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0

