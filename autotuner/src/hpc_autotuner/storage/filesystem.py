from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hpc_autotuner.core.paths import autotuning_output_dir


class FilesystemStorage:
    """Filesystem-backed storage for experiment metadata and evaluations."""

    def __init__(self, root: str | Path | None = None, run_group: str | None = None) -> None:
        self.root = Path(root) if root is not None else autotuning_output_dir() / (run_group or "default")
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def experiment_file(self) -> Path:
        return self.root / "experiment.json"

    @property
    def evaluations_file(self) -> Path:
        return self.root / "evaluations.jsonl"

    def write_experiment_metadata(self, metadata: dict[str, Any]) -> None:
        self.experiment_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    def append_evaluation(self, evaluation: dict[str, Any]) -> None:
        self.evaluations_file.parent.mkdir(parents=True, exist_ok=True)
        with self.evaluations_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evaluation, sort_keys=True) + "\n")

    def read_evaluations(self) -> list[dict[str, Any]]:
        if not self.evaluations_file.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.evaluations_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
