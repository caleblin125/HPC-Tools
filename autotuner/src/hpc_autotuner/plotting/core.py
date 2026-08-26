"""Generalized experiment analysis and plotting.

The plot utility is deliberately independent of HPL/GFLOPs: it reads the
``evaluations.jsonl`` records produced by the experiment controller and plots
*any* field against *any* other field, optionally with a rolling aggregate.
This makes it reusable for runtime, queue time, memory utilization, NB, N,
and any other recorded metric.

Example::

    hpc-tune plot --input outputs/autotuning \\
        --x attempt --y gflops --aggregate cummax --output cummax_gflops.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

Aggregate = Literal["raw", "cummax", "cummin", "cummean"]


def read_evaluations(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL evaluations file."""
    import json

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_experiments(input_dir: str | Path) -> list[dict[str, Any]]:
    """Discover optimizer experiment directories under ``input_dir``.

    Each subdirectory containing an ``evaluations.jsonl`` becomes one
    experiment named after the directory. Records are de-duplicated by attempt
    (the last record per attempt wins), which makes resumable experiments plot
    cleanly.
    """
    input_dir = Path(input_dir)
    experiments: list[dict[str, Any]] = []
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for group_dir in sorted(input_dir.iterdir()):
        eval_file = group_dir / "evaluations.jsonl"
        if not group_dir.is_dir() or not eval_file.exists():
            continue
        records = read_evaluations(eval_file)
        experiments.append({"name": group_dir.name, "path": group_dir, "records": records})
    if not experiments:
        raise ValueError(f"No evaluations.jsonl found under {input_dir}")
    return experiments


def field_value(record: dict[str, Any], field: str) -> Any:
    """Extract a field, falling back to the ``metrics`` dict."""
    if field in record:
        return record[field]
    metrics = record.get("metrics") or {}
    if field in metrics:
        return metrics[field]
    return None


def transform(
    records: Iterable[dict[str, Any]],
    x: str = "attempt",
    y: str = "objective",
    aggregate: str = "raw",
) -> tuple[list[float], list[float]]:
    """Transform records into plottable ``(xs, ys)`` sequences.

    Records without a valid ``x`` or ``y`` value are skipped. ``x`` values are
    kept unique (first occurrence wins) so every point appears once.
    """
    rows: list[tuple[float, float]] = []
    seen_x: set[float] = set()
    for record in records:
        x_value = field_value(record, x)
        y_value = field_value(record, y)
        if x_value is None or y_value is None:
            continue
        try:
            xf, yf = float(x_value), float(y_value)
        except (TypeError, ValueError):
            continue
        if xf in seen_x:
            continue
        seen_x.add(xf)
        rows.append((xf, yf))

    rows.sort(key=lambda pair: pair[0])
    xs = [pair[0] for pair in rows]
    ys = [pair[1] for pair in rows]

    if aggregate in ("cummax", "cummin", "cummean"):
        cumulative: list[float] = []
        for index in range(len(ys)):
            window = ys[: index + 1]
            if aggregate == "cummax":
                cumulative.append(max(window))
            elif aggregate == "cummin":
                cumulative.append(min(window))
            else:
                cumulative.append(sum(window) / len(window))
        ys = cumulative
    elif aggregate != "raw":
        raise ValueError(f"Unknown aggregate {aggregate!r}; use raw/cummax/cummin/cummean")
    return xs, ys


def plot_metric(
    experiments: Iterable[dict[str, Any]],
    x: str = "attempt",
    y: str = "objective",
    aggregate: str = "cummax",
    ax: Any = None,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
) -> Any:
    """Plot one or more experiments on a single axes.

    ``experiments`` is a list of ``{"name": ..., "records": [...]}`` dicts as
    returned by :func:`load_experiments`.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for experiment in experiments:
        xs, ys = transform(experiment["records"], x=x, y=y, aggregate=aggregate)
        if not xs:
            continue
        label = f"{experiment['name']} ({aggregate})"
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, label=label)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if aggregate != "raw":
        ax.set_title(title or f"{y} ({aggregate}) vs {x}")
    else:
        ax.set_title(title or f"{y} vs {x}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    return ax


def plot_experiments(
    input_dir: str | Path,
    x: str = "attempt",
    y: str = "objective",
    aggregate: str = "cummax",
    output: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
) -> Any:
    """Discover experiments under ``input_dir`` and plot them together."""
    import matplotlib.pyplot as plt

    experiments = load_experiments(input_dir)
    fig, ax = plt.subplots(figsize=figsize)
    plot_metric(experiments, x=x, y=y, aggregate=aggregate, ax=ax, title=title)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
    else:
        fig.show()
    return fig
