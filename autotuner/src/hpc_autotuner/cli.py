from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune HPC applications with Slurm-backed evaluations.")
    parser.add_argument("--version", action="store_true", help="Show version information.")
    parser.add_argument("run", nargs="?", help="Run an experiment configuration or experiment name.")
    parser.add_argument("--config", default=None, help="Optional path to an experiment configuration file.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print("hpc-tune 0.1.0")
        return

    if args.run is None:
        parser.print_help()
        return

    print(f"Requested run: {args.run}")
    if args.config:
        print(f"Using config: {Path(args.config).resolve()}")


if __name__ == "__main__":
    main()
