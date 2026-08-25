from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Scheduler(ABC):
    """Abstract scheduler interface for Slurm-like job execution."""

    @abstractmethod
    def submit(self, script: str | Any, environment: dict[str, str] | None = None) -> str:
        """Submit a script and return the job identifier."""

    @abstractmethod
    def wait(self, job_id: str) -> None:
        """Wait for a job to finish."""

    @abstractmethod
    def status(self, job_id: str) -> str:
        """Return the current status of a job."""
