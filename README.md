# HPC-Tools

Tools I use for building, running, and tuning applications on HPC clusters
(NERSC Perlmutter, and the student clusters used by UCSC's *Not So Slow Slugs*
HPC competition team).

The main project here is the **[HPC Autotuner](autotuner/)**; the rest are
small Bash helpers for day-to-day cluster work.

## HPC Autotuner — [`autotuner/`](autotuner/)

A generic, Slurm-backed framework for automatically tuning the performance of
HPC applications.

* A parent Slurm job runs an optimizer loop: **suggest** a configuration →
  submit it as a child Slurm job → parse a scalar metric (e.g. GFLOPS) from the
  output → **observe** the result → repeat.
* **Seven optimizers** share one `suggest()` / `observe()` interface: Random
  Search, Elite Search, SMAC3, Ray Tune, Hyperopt, DEAP, and CMA-ES, so they
  can be compared head-to-head on the same evaluation budget.
* **Application-agnostic:** an application is a small class that declares its
  parameters (int / float / categorical), renders a command, and parses its
  result. Bundled examples: HPL (High Performance Linpack), compiler-flag
  tuning, and synthetic test functions.
* Results are stored as `experiment.json` plus append-only
  `evaluations.jsonl`, so experiments can resume after cluster outages, and the
  `hpc-tune plot` CLI compares optimizers.
* Includes a pytest suite and ready-made Perlmutter configs and launch scripts.

See [`autotuner/README.md`](autotuner/README.md) for the full design,
configuration reference, and how to add a new application.

```bash
cd autotuner
python -m venv .venv && source .venv/bin/activate
pip install -e '.[smac,hyperopt]'
python -m hpc_autotuner.experiments.random --config configs/perlmutter_smoke.yaml --budget 5
```

## Shell helpers

Source these into your shell (`source tools.sh`, etc.).

| File | Functions | Purpose |
|---|---|---|
| [`tools.sh`](tools.sh) | `catout`, `caterr` | Print the stdout / stderr of the latest (or a given) Slurm job |
| [`add_build.sh`](add_build.sh) | `add_build`, `remove_build` | Add or remove an install prefix's `bin`, `lib`/`lib64`, and `include` on `PATH`, `LD_LIBRARY_PATH`, `CPATH`, etc. |
| [`basic_build.sh`](basic_build.sh) | `basic_cmake_github`, `basic_cmake_tarball` | Clone (or download) a CMake project, configure, build, and install it in one call |
| [`spack_install.sh`](spack_install.sh) | — | Install [Spack](https://spack.io) v1.0 and add it to `~/.bashrc` |
| [`sample_batch.job`](sample_batch.job) | — | Template Slurm batch script that logs a description for each run |

## Build recipes (work in progress)

`builds/acts.sh` and `spack_builds/acts.sh` are attempts at building the
[ACTS](https://github.com/acts-project/acts) particle-tracking toolkit and its
dependency stack (ROOT, Geant4, DD4hep, Pythia8, HepMC3) from source and with
Spack. They are not yet working end to end.
