from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Evaluation:
    """Represents a single attempted configuration evaluation."""

    evaluation_id: int
    configuration: dict[str, Any]
    job_id: str | None = None
    objective: float | None = None
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str | None = None
    submitted_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.submitted_at is None:
            self.submitted_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
