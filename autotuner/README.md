# HPC Autotuner

Automated performance tuning for HPC applications on Slurm clusters. The current
benchmark experiment compares **six autotuning methods** on the **HPL** (High
Performance Linpack) benchmark:

| Optimizer  | Driver module                       | Optional dependency |
|------------|-------------------------------------|---------------------|
| Random     | `hpc_autotuner.experiments.random`  | (none)              |
| SMAC3      | `hpc_autotuner.experiments.smac3`   | `smac`              |
| Ray Tune   | `hpc_autotuner.experiments.raytune` | `ray[tune]`         |
| Hyperopt   | `hpc_autotuner.experiments.hyperopt`| `hyperopt`          |
| DEAP       | `hpc_autotuner.experiments.deap`    | `deap`              |
| CMA-ES     | `hpc_autotuner.experiments.cmaes`   | `cmaes`             |

Each optimizer receives exactly **100 completed HPL evaluations** (a sequential
budget), the same HPL executable, the same 1-node / 128-task Slurm allocation,
the same parameter bounds, the same result parser, and the same objective
(**maximize GFLOPs**).

## Experiment design

* **Parent/controller job.** The autotuner runs as a Slurm parent job. Its
  Python process initializes the optimizer, submits one HPL *child* job at a
  time, waits for it, parses GFLOPs, records the evaluation, feeds it back to
  the optimizer (`optimizer.observe(...)`), then asks for the next
  configuration. Evaluations are strictly **sequential** — the optimizer sees
  the result of attempt N before attempt N+1 is generated.
* **Memory model.** The tunable is `memory_fraction ∈ [0.80, 0.96]` of the node's
  ~512 GiB. Following the HPL FAQ, memory use is dominated by the N×N matrix of
  doubles: `memory_bytes ≈ N² × 8 × memory_factor`, so

  ```python
  N = floor(sqrt(memory_fraction * node_memory_bytes / 8))
  ```

  Every recorded configuration includes `memory_fraction`, `target_memory_bytes`,
  `N`, `NB`, `P`, `Q`, and the fixed HPL.dat algorithm parameters.
* **Output layout.** Results follow the team convention plus JSON logs:

  ```
  outputs/
    slurm/job_<JOBID>.out|err          # child job stdout/stderr
    descriptions.txt                   # human-readable job descriptions
    autotuning/<run_group>/
      experiment.json                  # experiment metadata (seed, space, slurm, ...)
      evaluations.jsonl                # one JSON record per evaluation event
    <run_group>/<run_group>_<JOBID>.log   # child HPL run-group log
  ```

* **Attempt numbering & resume.** Attempts start at 1 and end at exactly the
  budget. Only a successful `sbatch` consumes an attempt number; failed
  submissions are recorded (`FAILED_SUBMISSION`) and never counted. If the
  controller is interrupted it resumes from `evaluations.jsonl`, replays history
  into the optimizer, and re-runs any interrupted evaluation.

## Installation

The six optimizer libraries are **optional** extras. Random Search needs nothing
beyond the base package:

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
node memory) live in a YAML experiment configuration, never in code:

* `configs/perlmutter_hpl.yaml` — the full 100-attempt benchmark (512 GiB node,
  `slurm.time` is the **child** HPL job limit).
* `configs/perlmutter_smoke.yaml` — a tiny 8 GiB smoke configuration so HPL runs
  finish in seconds instead of hours.

The parent job should be given a much larger time limit than the children,
because it runs up to 100 sequential HPL jobs.

## Running an optimizer

Launch one optimizer's parent Slurm job:

```bash
scripts/launch_benchmark.sh <optimizer> configs/perlmutter_hpl.yaml
```

`<optimizer>` is one of `random`, `smac3`, `raytune`, `hyperopt`, `deap`,
`cmaes`. The launcher reads the account/partition/QoS/constraint/time from the
YAML and passes them to `sbatch`; it exports `AUTOTUNE_CONFIG` (the config path)
and `AUTOTUNE_VENV` (defaults to `$PWD/.venv`) into the parent job. Each parent
script (e.g. `scripts/autotune_random.slurm`) is a thin wrapper that runs

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

## Perlmutter smoke test (before the real benchmark)

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
a fully-resolved `configuration` (including the derived `N`).

There is also a legacy runner smoke script (`scripts/run_tuning_smoke.py`) that
uses the older 1-task `Runner`; the new benchmark drivers use the sequential
controller in `hpc_autotuner.experiments.common`.

## Launching the full benchmark

The 100-attempt benchmark is **launched separately**, after the smoke test
passes:

```bash
scripts/launch_benchmark.sh random   configs/perlmutter_hpl.yaml
scripts/launch_benchmark.sh smac3    configs/perlmutter_hpl.yaml
scripts/launch_benchmark.sh raytune  configs/perlmutter_hpl.yaml
scripts/launch_benchmark.sh hyperopt configs/perlmutter_hpl.yaml
scripts/launch_benchmark.sh deap     configs/perlmutter_hpl.yaml
scripts/launch_benchmark.sh cmaes    configs/perlmutter_hpl.yaml
```

Each produces `outputs/autotuning/<run_group>/experiment.json` and
`evaluations.jsonl`.

## Analysis / plotting

The plotting utility is generalized beyond GFLOPs: it reads any
`evaluations.jsonl`, discovers all optimizer groups under an input directory,
and plots any recorded field against any other, with an optional aggregate.

```bash
hpc-tune plot --input outputs/autotuning --x attempt --y gflops --aggregate cummax --output cummax_gflops.png
# programmatic equivalent
python -c "from hpc_autotuner.plotting.core import plot_experiments; plot_experiments('outputs/autotuning', y='gflops', aggregate='cummax', output='cummax_gflops.png')"
```

## Tests

Unit tests never touch Slurm — a mock scheduler synthesizes HPL logs in-process:

```bash
pytest tests/test_benchmark.py tests/test_experiment.py tests/test_plotting.py tests/test_runner.py tests/test_real_slurm_and_optims.py
pytest tests/test_optimizers.py      # needs the optional optimizer libs installed
```

Covered: parameter bounds, `memory_fraction -> N` conversion, HPL.dat
generation, HPL result parsing, JSON/JSONL logging, attempt numbering, resume
from log, optimizer adapters, and plotting data transforms.

