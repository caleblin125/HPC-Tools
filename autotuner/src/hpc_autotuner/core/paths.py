from pathlib import Path
import os


def project_root() -> Path:
    """
    Return the project root.

    Prefer SLURM_SUBMIT_DIR when running inside a Slurm job.
    Otherwise walk upward from this source file.
    """
    submit_dir = os.environ.get("SLURM_SUBMIT_DIR")

    if submit_dir:
        return Path(submit_dir).resolve()

    # paths.py:
    # src/hpc_autotuner/core/paths.py
    #
    # parents[0] = core
    # parents[1] = hpc_autotuner
    # parents[2] = src
    # parents[3] = project root
    return Path(__file__).resolve().parents[3]


def output_dir() -> Path:
    return project_root() / "outputs"


def slurm_output_dir() -> Path:
    return output_dir() / "slurm"


def autotuning_output_dir() -> Path:
    return output_dir() / "autotuning"


def run_group_dir(run_group: str) -> Path:
    return output_dir() / run_group