# HPC Autotuner

A generic, Slurm-backed autotuning framework for HPC applications. A parent
controller job asks an optimizer for a configuration, runs it as a child Slurm
job, parses a scalar performance metric from the output, and feeds the result
back to the optimizer — repeated for a fixed, sequential evaluation budget.

The framework is **application-agnostic**. The "application" is any executable
or build-and-run workflow whose behaviour depends on a set of parameters and
whose performance reduces to a scalar metric. The bundled, fully configured
example is the **HPL** (High Performance Linpack) benchmark, but the same
machinery tunes other workloads — compiler/build flags, runtime knobs, MPI or
thread configuration, I/O parameters, and so on. See *Using the framework for
other applications* below.

## How the framework is structured

| Component | Module | Role |
|-----------|--------|------|
| Application | `hpc_autotuner.applications.*` | declares tunable parameters, renders a shell command, parses results |
| Optimizers | `hpc_autotuner.optimizers.*` | seven autotuning methods behind one `suggest()` / `observe()` interface |
| Controller | `hpc_autotuner.experiments.common` | sequential parent-job loop: suggest → child Slurm job → parse → record → observe |
| Storage | `hpc_autotuner.storage.filesystem` | `experiment.json` metadata + `evaluations.jsonl` records |
| Plotting | `hpc_autotuner.plotting.core` | field-based analysis and plots (any metric, any optimizer group) |
| CLI | `hpc_autotuner.cli` | `hpc-tune plot` |

### The application interface

An application is a small subclass of
`hpc_autotuner.applications.base.Application`:

* `parameters` — a list of `Parameter` objects: `int` / `float` (with
  `bounds`) or `categorical` (with `choices`). A parameter with a non-`None`
  `fixed_value` is pinned for every evaluation and is **not** part of the
  optimizer's search space.
* `command(configuration) -> str | list[str]` — turn a configuration into the
  shell command the child Slurm job executes.
* `parse_result(output) -> {"metrics": {...}, "objective": <float|None>,
  "success": bool}` — extract the scalar objective (and any other metrics)
  from the application's output.
* `resolve_configuration(configuration)` — optional; merge fixed values and
  derive quantities. (HPL derives the problem size `N` from `memory_fraction`.)

### Optimizers

Seven autotuning methods share one common interface and (by default) use
**optional** dependencies:

| Optimizer | Adapter | Optional dependency | Handles categoricals? |
|-----------|---------|---------------------|------------------------|
| Random Search | `hpc_autotuner.optimizers.random` | (none) | yes |
| Elite Search | `hpc_autotuner.optimizers.elite` | (none) | yes (int/categorical) |
| SMAC3 | `hpc_autotuner.optimizers.smac3` | `smac` | yes |
| Ray Tune | `hpc_autotuner.optimizers.raytune` | `ray[tune]` | yes |
| Hyperopt | `hpc_autotuner.optimizers.hyperopt` | `hyperopt` | yes |
| DEAP | `hpc_autotuner.optimizers.deap` | `deap` | no (numeric only) |
| CMA-ES | `hpc_autotuner.optimizers.cmaes` | `cmaes` | no (numeric only) |

> DEAP and CMA-ES search a normalized continuous space, so they require purely
> numeric tunables. Use Random/SMAC3/Ray Tune/Hyperopt when the parameter space
> contains categorical parameters. **Elite Search** is a dependency-free
> elitist local search over ordered discrete values, migrated from the previous
> standalone `HPC-Tools/HPLtuning` project (it was the `EliteRandomSearch`
> policy used there to tune HPL).

### Controller (parent Slurm job)

Each optimizer experiment runs as a parent Slurm job whose Python process:

1. initializes the optimizer,
2. `optimizer.suggest()` → a configuration,
3. resolves and validates the configuration,
4. renders a child Slurm script from `resources/slurm/job.sh`,
5. submits it with `sbatch`, then waits for the child job,
6. parses the application's output → the objective (GFLOPs or whatever metric),
7. appends an evaluation record to `evaluations.jsonl`,
8. `optimizer.observe(config, result)`,
9. repeats until the evaluation budget is reached.

Evaluations are strictly **sequential** — the optimizer sees the result of
attempt N before attempt N+1 is generated. Attempt numbers run from 1 to the
budget; a failed `sbatch` is recorded (`FAILED_SUBMISSION`) and does **not**
consume an attempt. Experiments are **resumable**: on restart the controller
replays `evaluations.jsonl` into the optimizer and re-runs any interrupted
evaluation.

### Output layout and records

```
outputs/
  slurm/job_<JOBID>.out|err             # child job stdout/stderr
  descriptions.txt                      # human-readable job descriptions
  autotuning/<run_group>/
    experiment.json                     # metadata: seed, space, objective, slurm, ...
    evaluations.jsonl                   # one JSON record per evaluation event
  <run_group>/<run_group>_<JOBID>.log   # child run-group log (application output)
```

Each evaluation record contains the optimizer name, attempt number, the fully
resolved configuration, the Slurm job id, status, success, the objective, and
the application metrics (e.g. `gflops`, `runtime`), plus timing fields
(`queue_time`, `compute_time`) when `sacct` is available.

### Objectives

The objective is a scalar metric plus a direction. An application declares
`objective_metric` and `objective_direction` (defaults: `gflops` / `maximize`).
The controller records the metric and the adapters convert it to the internal
minimization loss; any metric can be used, in either direction.

## Bundled example: the HPL benchmark

The reference experiment tunes HPL on a full Perlmutter CPU node (1 node,
128 tasks) and compares the seven optimizers on identical terms.

* **Problem size.** The tunable problem size is `N`, bounded by the memory
  model so the HPL matrix stays within `[0.80, 0.90]` of the target HPL memory
  of `121920 MB` (~119 GiB, half of the full node target to keep runs shorter).
  Following the HPL FAQ, `memory ≈ N² × 8 bytes`, so the bounds are
  `N ∈ [⌊√(0.80·mem/8)⌋, ⌊√(0.90·mem/8)⌋]` (~110k–117k). The maximum problem
  size at 100% memory is `√(121920 MB / 8) ≈ 123450`. Every evaluation records
  `N` and `target_memory_bytes`.
* **Full HPL.dat parameter space** (mirrors the previous `HPLtuning` work):
  `NB`, `P`, `PMAP`, `PFACT`, `NBMIN`, `NDIV`, `RFACT`, `BCAST`, `DEPTH`,
  `SWAP`, `L1`, `U`, `EQUIL` are tunable. Because HPL requires
  `P*Q == ntasks`, `P` is a categorical choice over the divisors of `ntasks`
  and `Q = ntasks / P` is derived (both are recorded in each evaluation).
  `SWAP_THRESH` and `ALIGN` stay fixed and are recorded.
* **Objective:** maximize GFLOPs.

Configurations:

* `configs/perlmutter_hpl.yaml` — the full 100-attempt benchmark (512 GiB node,
  full tunable set; `slurm.time` is the **child** HPL job limit).
* `configs/perlmutter_smoke.yaml` — a tiny 8 GiB smoke configuration with the
  same tunable set, so each HPL run finishes in seconds instead of hours.

Each optimizer has a thin parent script in `scripts/` that runs
`python -m hpc_autotuner.experiments.<optimizer> --config "$CONFIG"`.

> DEAP and CMA-ES search a normalized continuous space and cannot represent
> categorical parameters (notably `P`), so they are not run against this
> space. Random, SMAC3, Ray Tune, Hyperopt, and Elite Search handle the full
> discrete set.

### Smoke test before the real benchmark

Prove the pipeline works end to end before launching expensive runs:

```bash
scripts/run_smoke_benchmark.sh           # budget=1, then budget=3 (random optimizer)
```

Then verify:

```bash
ls outputs/autotuning/smoke_single/evaluations.jsonl
ls outputs/autotuning/smoke_three/evaluations.jsonl
```

Each record should show `status: COMPLETED`, a parsed `objective` (GFLOPs), and
a fully resolved `configuration` (including the derived `N`).

### Launching the full benchmark

The 100-attempt benchmark is **launched separately**, after the smoke test
passes. One command launches all five discrete-capable optimizers:

```bash
scripts/launch_full_benchmark.sh configs/perlmutter_hpl.yaml   # random, smac3, raytune, hyperopt, elite
```

or launch each individually:

```bash
scripts/launch_benchmark.sh random   configs/perlmutter_hpl.yaml --budget 100 --run-group random
scripts/launch_benchmark.sh smac3    configs/perlmutter_hpl.yaml --budget 100 --run-group smac3
scripts/launch_benchmark.sh raytune  configs/perlmutter_hpl.yaml --budget 100 --run-group raytune
scripts/launch_benchmark.sh hyperopt configs/perlmutter_hpl.yaml --budget 100 --run-group hyperopt
scripts/launch_benchmark.sh elite    configs/perlmutter_hpl.yaml --budget 100 --run-group elite
```

Each produces `outputs/autotuning/<run_group>/experiment.json` and
`evaluations.jsonl`. If a parent job hits the wall limit, resubmit the same
launch and the experiment resumes from its log.

## Using the framework for other applications

Three steps to tune a new workload (compiler flags, runtime knobs, ...):

1. **Subclass `Application`** — declare the tunable parameters, render the
   shell command, and parse the scalar objective from the output.
2. **Register it** with `register_application("<kind>", factory)`.
3. **Point a config** at `application.type: <kind>` and run it through the
   same drivers, scripts, storage, and plotting utility.

The full, tested example below (compiler-flag tuning) ships in
`src/hpc_autotuner/applications/compile_flags.py`:

```python
from hpc_autotuner.applications.base import Application
from hpc_autotuner.core.parameter import Parameter

class CompileFlagsApplication(Application):
    parameters = [
        Parameter("opt_level", "int", bounds=(0, 3)),
        Parameter("arch", "categorical", choices=["native", "x86-64", "avx2"]),
        Parameter("lto", "categorical", choices=[False, True]),
    ]
    objective_metric = "score"
    objective_direction = "maximize"

    def command(self, configuration):
        flags = [f"-O{int(configuration['opt_level'])}",
                 f"-march={configuration['arch']}"]
        if configuration.get("lto"):
            flags.append("-flto")
        cflags = " ".join(flags)
        return (
            f"CFLAGS='{cflags}' make clean >/dev/null 2>&1 || true\n"
            f"CFLAGS='{cflags}' make >/dev/null 2>&1 || true\n"
            "./benchmark\n"
        )

    def parse_result(self, output):
        import re
        m = re.search(r"score\s*[:=]\s*([0-9.eE+-]+)", output, re.IGNORECASE)
        if not m:
            return {"metrics": {}, "objective": None, "success": False}
        value = float(m.group(1))
        return {"metrics": {"score": value}, "objective": value, "success": True}

    @classmethod
    def from_config(cls, config):
        return cls(executable=config.executable or "./benchmark",
                   fixed=config.fixed)
```

Register it (in your experiment entrypoint or at import time):

```python
from hpc_autotuner.experiments.common import register_application
register_application("compile_flags", CompileFlagsApplication.from_config)
```

Then run it with the ordinary drivers:

```bash
# one-off from a login node
python -m hpc_autotuner.experiments.random --config configs/compile_flags_example.yaml --budget 5

# or as a parent Slurm job
scripts/launch_benchmark.sh random configs/compile_flags_example.yaml
```

The bundled `configs/compile_flags_example.yaml` shows the config shape; the
generic parts (controller, optimizers, storage, plotting, CLI) are unchanged —
only the `Application` differs. Note that `compile_flags` uses categorical
parameters, so pick a discrete optimizer (Random/SMAC3/Ray/Hyperopt).

## Installation

The optimizer libraries are **optional** extras. Random Search and Elite Search
need nothing beyond the base package:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                # base framework (Random Search works)
pip install -e '.[smac]'        # SMAC3  (pyrfr builds on Linux)
pip install -e '.[ray]'         # Ray Tune
pip install -e '.[hyperopt]'    # Hyperopt
pip install -e '.[deap]'        # DEAP
pip install -e '.[cmaes]'       # CMA-ES
pip install -e '.[plot]'        # matplotlib for the plotting utility
pip install -e '.[all]'         # everything
```

## Configuration

Machine-specific settings (account, partition, QoS, constraint, executable,
etc.) live in a YAML experiment configuration, never in code. The schema is
generic:

```yaml
experiment:
  name: my-experiment
  optimizer: random      # random | smac3 | raytune | hyperopt | deap | cmaes
  run_group: my_group
  budget: 100            # completed evaluations
  seed: 42               # recorded in experiment.json for reproducibility

objective:
  metric: score          # any metric recorded by the application
  direction: maximize    # maximize | minimize

application:
  type: hpl              # registered application kind (see register_application)
  executable: /path/to/binary
  fixed: {}              # application-specific pinned values

slurm:
  account: null          # machine-specific; set here or on the sbatch command line
  partition: null
  qos: regular           # regular CPU queue on Perlmutter (no partition needed)
  constraint: cpu
  time: "00:30:00"       # child job time limit (the parent should get more)
  exclusive: false
  nodes: 1
  ntasks: 128
  submit_retries: 3
  polling_interval: 15

outputs:
  root: null             # default: <project_root>/outputs/autotuning
```

Application-specific fields may be added under `application` (the HPL example
uses `node_memory_bytes`, `memory_factor`, `memory_fraction_bounds`); the
factory for the registered kind decides which fields it reads.

## Running an experiment

Launch one optimizer's parent Slurm job:

```bash
scripts/launch_benchmark.sh <optimizer> <config.yaml>
```

`<optimizer>` is one of `random`, `smac3`, `raytune`, `hyperopt`, `deap`,
`cmaes`. The launcher reads the account/partition/QoS/constraint/time from the
YAML and passes them to `sbatch`; it exports `AUTOTUNE_CONFIG` (the config
path) and `AUTOTUNE_VENV` (defaults to `$PWD/.venv`) into the parent job. Each
parent script (`scripts/autotune_<optimizer>.slurm`) is a thin wrapper that
runs

```bash
python -m hpc_autotuner.experiments.<optimizer> --config "$CONFIG"
```

Useful driver overrides:

```bash
scripts/launch_benchmark.sh random configs/perlmutter_smoke.yaml --budget 3 --run-group smoke_three
```

Individual drivers can also be run directly from a login node (e.g. a quick
single-evaluation check):

```bash
python -m hpc_autotuner.experiments.random --config configs/perlmutter_smoke.yaml --budget 1
```

## Analysis / plotting

The plotting utility is metric-agnostic: it reads any `evaluations.jsonl`,
discovers all optimizer groups under an input directory, and plots any recorded
field against any other, with an optional aggregate.

```bash
hpc-tune plot --input outputs/autotuning --x attempt --y gflops --aggregate cummax --output cummax_gflops.png
# programmatic equivalent
python -c "from hpc_autotuner.plotting.core import plot_experiments; plot_experiments('outputs/autotuning', y='gflops', aggregate='cummax', output='cummax_gflops.png')"
```

Any recorded field works: `runtime`, `queue_time`, `memory_fraction`, `N`, ...,
for any application. `hpc-tune plot` is run manually — never as part of the
benchmark.

## Tests

Unit tests never touch Slurm — a mock scheduler synthesizes application logs
in-process:

```bash
pytest tests/test_benchmark.py tests/test_experiment.py tests/test_plotting.py tests/test_runner.py tests/test_real_slurm_and_optims.py tests/test_compile_flags.py
pytest tests/test_optimizers.py      # needs the optional optimizer libs installed
```

Covered: parameter bounds and clipping, memory-fraction → problem-size
conversion, HPL.dat generation, HPL result parsing, the generic
`CompileFlagsApplication` (command rendering, parsing, registration), JSON/JSONL
logging, attempt numbering, resume-from-log, optimizer adapters, and plotting
data transforms.

## Repository layout

```
autotuner/
  configs/                     # YAML experiment configurations
  scripts/                     # parent Slurm scripts + launchers
  src/hpc_autotuner/
    applications/              # Application adapters (hpl, compile_flags, ...)
    core/                      # Parameter, ParameterSpace, Evaluation, paths
    experiments/               # sequential controller, config, jobscript, drivers
    optimizers/                # optimizer adapters (incl. Elite Search) + registry
    plotting/                  # generalized analysis/plotting
    resources/slurm/           # child job template
    runner/                    # legacy runner (single-task smoke tool)
    schedulers/ storage/ cli.py
  tests/                       # unit tests (mock scheduler, no Slurm)
```
