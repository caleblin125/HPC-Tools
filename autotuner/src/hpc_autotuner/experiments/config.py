"""Experiment configuration: machine-specific settings live here, never in code.

A benchmark experiment is defined by a YAML file that captures the Slurm
allocation, the HPL executable and memory model, the objective, and the
optimizer budget/seed. None of the account/partition/QoS/module paths are
hard-coded in the framework; they are read from this file (or from command
line overrides) so the same code runs unchanged on Perlmutter and elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SlurmConfig:
    account: str | None = None
    partition: str | None = None
    qos: str = "shared"
    constraint: str = "cpu"
    time: str = "00:30:00"
    exclusive: bool = False
    nodes: int = 1
    ntasks: int = 128
    cpus_per_task: int | None = None
    submit_retries: int = 3
    polling_interval: float = 10.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SlurmConfig":
        clean = {k: v for k, v in (data or {}).items() if v is not None}
        return cls(**clean)


@dataclass
class ExperimentConfig:
    name: str = "hpl-benchmark"
    optimizer: str = "random"
    run_group: str = "random"
    budget: int = 100
    seed: int = 42
    objective_metric: str = "gflops"
    objective_direction: str = "maximize"
    application_type: str = "hpl"
    executable: str | None = None
    node_memory_bytes: int = 512 * 1024**3
    memory_factor: float = 1.0
    memory_fraction_bounds: tuple[float, float] = (0.80, 0.96)
    tunable: list[str] | None = None
    fixed: dict[str, Any] = field(default_factory=dict)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    output_root: str | None = None
    project_root: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Experiment config not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        experiment = raw.get("experiment", {})
        objective = raw.get("objective", {})
        application = raw.get("application", {})
        slurm = SlurmConfig.from_dict(raw.get("slurm", {}))
        outputs = raw.get("outputs", {}) or {}

        def _memory_bytes(value: Any, default: int) -> int:
            if value is None:
                return default
            if isinstance(value, str):
                return _parse_size(value)
            return int(value)

        return cls(
            name=str(experiment.get("name", "hpl-benchmark")),
            optimizer=str(experiment.get("optimizer", "random")),
            run_group=str(experiment.get("run_group", experiment.get("optimizer", "random"))),
            budget=int(experiment.get("budget", 100)),
            seed=int(experiment.get("seed", 42)),
            objective_metric=str(objective.get("metric", "gflops")),
            objective_direction=str(objective.get("direction", "maximize")),
            application_type=str(application.get("type", "hpl")),
            executable=application.get("executable"),
            node_memory_bytes=_memory_bytes(application.get("node_memory_bytes"), 512 * 1024**3),
            memory_factor=float(application.get("memory_factor", 1.0)),
            memory_fraction_bounds=tuple(
                float(v) for v in application.get("memory_fraction_bounds", [0.80, 0.96])
            ),
            tunable=list(application.get("tunable")) if application.get("tunable") else None,
            fixed=dict(application.get("fixed", {}) or {}),
            slurm=slurm,
            output_root=outputs.get("root"),
            project_root=raw.get("project_root"),
        )

    def resolved_project_root(self) -> Path:
        if self.project_root:
            return Path(self.project_root).resolve()
        return Path(os.environ.get("SLURM_SUBMIT_DIR", Path.cwd())).resolve()

    def resolved_output_root(self) -> Path:
        if self.output_root:
            return Path(self.output_root).resolve()
        return self.resolved_project_root() / "outputs" / "autotuning"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["slurm"] = asdict(self.slurm)
        return payload


def _parse_size(value: str) -> int:
    """Parse a human size like ``"512GiB"`` or ``"64GB"`` into bytes."""
    value = value.strip()
    multiplier = 1
    suffix = ""
    for unit, factor in [
        ("GiB", 1024**3),
        ("GB", 10**9),
        ("MiB", 1024**2),
        ("MB", 10**6),
        ("KiB", 1024),
        ("KB", 10**3),
    ]:
        if value.upper().endswith(unit.upper()):
            multiplier = factor
            suffix = unit
            break
    number = value[: len(value) - len(suffix)] if suffix else value
    return int(float(number) * multiplier)
