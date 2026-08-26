import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from hpc_autotuner.applications.base import Application
from hpc_autotuner.core.evaluation import Evaluation
from hpc_autotuner.core.parameter import Parameter
from hpc_autotuner.optimizers.base import Optimizer
from hpc_autotuner.runner.runner import Runner
from hpc_autotuner.schedulers.base import Scheduler
from hpc_autotuner.storage.filesystem import FilesystemStorage


def _bash_available() -> bool:
    if shutil.which("bash") is None:
        return False
    try:
        result = subprocess.run(["bash", "-c", "true"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


_BASH_AVAILABLE = _bash_available()



class FakeApp(Application):
    parameters = [
        Parameter("n", "int", bounds=(1, 4)),
        Parameter("scale", "float", bounds=(0.5, 2.0)),
    ]

    def command(self, configuration):
        return ["python", "-c", f"print('METRICS: n={configuration['n']} scale={configuration['scale']} runtime={configuration['n'] * configuration['scale']}; objective={configuration['n'] * configuration['scale'] * 3}')"]

    def parse_result(self, output):
        metrics = {}
        for token in output.split():
            if token.startswith("runtime="):
                metrics["runtime"] = float(token.split("=", 1)[1].rstrip(";"))
            elif token.startswith("objective="):
                metrics["objective"] = float(token.split("=", 1)[1].rstrip(";"))
        return {"metrics": metrics, "objective": metrics["objective"], "success": True}


class FakeOptimizer(Optimizer):
    def __init__(self):
        self._configs = [{"n": 1, "scale": 1.0}, {"n": 2, "scale": 1.5}]

    def suggest(self):
        return self._configs.pop(0)

    def observe(self, configuration, result):
        return None


class FakeScheduler(Scheduler):
    def submit(self, script, environment=None):
        root = Path(script).resolve().parents[2]
        env = dict(__import__("os").environ)
        env.update({
            "SLURM_JOB_ID": "1234",
            "SLURM_NTASKS": "1",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_SUBMIT_DIR": str(root),
        })
        subprocess.run(["bash", script], cwd=root, check=True, capture_output=True, text=True, env=env)
        return "1234"

    def wait(self, job_id):
        return None

    def status(self, job_id):
        return "COMPLETED"


@dataclass
class DummyJob:
    script: str
    environment: dict | None = None


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash interpreter required to run rendered job scripts")
def test_runner_end_to_end(tmp_path):
    storage = FilesystemStorage(root=tmp_path / "run")
    app = FakeApp()
    optimizer = FakeOptimizer()
    scheduler = FakeScheduler()
    runner = Runner(
        optimizer=optimizer,
        application=app,
        scheduler=scheduler,
        storage=storage,
        evaluation_budget=2,
        project_root=tmp_path,
    )

    history = runner.run()
    assert len(history) == 2
    assert all(item.success for item in history)
    assert history[0].job_id == "1234"
    assert history[0].objective is not None
    assert (tmp_path / "outputs" / "autotuning" / "run").exists()
