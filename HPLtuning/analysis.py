#!/usr/bin/env python3
"""Analyze HPL/HPL-AI autotuning results.

Run from an autotuning workspace, for example:
  python /home/caleb/HPC-Tools/HPLtuning/analysis.py --results hpl_ai_results.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=None,
                        help="JSON results file (auto-detect if omitted)")
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    results = args.results
    if results is None:
        for name in ("hpl_ai_results.json", "hpl_config_results.json"):
            candidate = Path(name)
            if candidate.exists():
                results = candidate
                break
    if results is None or not results.exists():
        raise SystemExit("No results JSON found; pass --results PATH.")

    df = pd.read_json(results)
    if df.empty:
        raise SystemExit(f"No results in {results}")
    df = df.sort_values("Config Id").copy()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    valid = df[df["Possible"].fillna(False) & df["GFlops"].notna()].copy()
    if valid.empty:
        raise SystemExit("No valid completed configurations yet.")
    valid["Best so far"] = valid["GFlops"].cummax()

    # Config ID is evaluation order; Slurm IDs may have gaps after restarts.
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(valid["Config Id"], valid["GFlops"], "o-", lw=1.1, ms=4,
            color="#4c78a8", label="completed trial")
    ax.step(valid["Config Id"], valid["Best so far"], where="post", lw=2.4,
            color="#f58518", label="best so far")
    best = valid.loc[valid["GFlops"].idxmax()]
    ax.scatter(best["Config Id"], best["GFlops"], s=80, zorder=3,
               color="#e45756", label=f"best: {best['GFlops']:.3f} GFLOP/s")
    ax.annotate(f"{best['GFlops']:.1f}", (best["Config Id"], best["GFlops"]),
                xytext=(5, 7), textcoords="offset points")
    ax.set(title=f"{results.stem}: tuning progress", xlabel="configuration evaluation order",
           ylabel="GFLOP/s")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(args.plots_dir / "tuning_progress.png", dpi=200)
    plt.close(fig)

    reserved = {"Config Id", "Config File", "Output File", "GFlops", "Job Id", "State", "Possible"}
    parameters = [c for c in valid.columns if c not in reserved | {"Best so far"}
                  and pd.api.types.is_numeric_dtype(valid[c])]
    for parameter in parameters:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
        ax.scatter(valid[parameter], valid["GFlops"], alpha=.8)
        ax.set(xlabel=parameter, ylabel="GFLOP/s", title=f"Performance vs. {parameter}")
        ax.grid(alpha=.25)
        fig.savefig(args.plots_dir / f"{parameter}.png", dpi=200)
        plt.close(fig)

    print(f"Completed valid trials: {len(valid)} / {len(df)}")
    print("Best:", best[["Config Id", "GFlops", *parameters]].to_dict())
    print("Progress plot:", args.plots_dir / "tuning_progress.png")


if __name__ == "__main__":
    main()
