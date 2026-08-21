from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evaluation:
    evaluation_id: int
    configuration: dict[str, Any]

    job_id: str | None = None

    objective: float | None = None

    success: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )