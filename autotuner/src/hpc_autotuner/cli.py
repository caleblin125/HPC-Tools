from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hpc_autotuner.plotting as plotting


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune HPC applications with Slurm-backed evaluations.")
    subparsers = parser.add_subparsers(dest="command")

    plot = subparsers.add_parser("plot", help="Plot experiment evaluation logs.")
    plot.add_argument("--input", required=True, help="Directory containing <group>/evaluations.jsonl.")
    plot.add_argument("--x", default="attempt", help="Field for the x axis (default: attempt).")
    plot.add_argument("--y", default="objective", help="Field for the y axis (default: objective).")
    plot.add_argument(
        "--aggregate",
        default="cummax",
        choices=["raw", "cummax", "cummin", "cummean"],
        help="Aggregation applied to the y values (default: cummax).",
    )
    plot.add_argument("--output", default=None, help="Write the figure to this PNG path.")
    plot.add_argument("--title", default=None, help="Optional plot title.")

    parser.add_argument("--version", action="store_true", help="Show version information.")
    parser.add_argument("run", nargs="?", help="Run an experiment configuration or experiment name.")
    parser.add_argument("--config", default=None, help="Optional path to an experiment configuration file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "plot":
        try:
            plotting.plot_experiments(
                input_dir=args.input,
                x=args.x,
                y=args.y,
                aggregate=args.aggregate,
                output=args.output,
                title=args.title,
            )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.version:
        print("hpc-tune 0.1.0")
        return 0

    if args.run is None:
        parser.print_help()
        return 0

    print(f"Requested run: {args.run}")
    if args.config:
        print(f"Using config: {Path(args.config).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

