# Active Learning Platform for Computer Vision

This project compares active learning strategies for computer vision tasks.

## Goal

Build a reproducible pipeline for:
- dataset preparation
- initial random sampling
- model training
- active learning query strategies
- evaluation
- reporting

## Benchmark strategies

1. Random Sampling
2. Least Confidence
3. Margin Sampling
4. Entropy Sampling
5. MC Dropout Entropy
6. BALD
7. Core-Set / Diversity Sampling
8. BADGE
9. Learning Loss (simplified MSE-based loss prediction)

## Running experiments

Use Python 3.12 or newer. From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_experiment.py --config configs/smoke.yaml --device cpu
```

The smoke configuration runs all nine strategies on small synthetic datasets. It
checks the pipeline without downloading CIFAR-10; its accuracy is not a benchmark
result.

For the real benchmark:

```bash
python scripts/run_experiment.py --config configs/benchmark.yaml --device cuda
```

Use `--device cpu` if CUDA is unavailable. Omit `--strategy` and `--seed` to run all
combinations from the YAML, or override them for a single run:

```bash
python scripts/run_experiment.py --config configs/benchmark.yaml --strategy entropy --seed 1 --device cpu
```

The default protocol uses CIFAR-10, one seed (`1`), an initial pool of 500 images,
eight acquisitions of 500 images, and nine measurements from 500 to 4,500 labeled
images. Every round trains a new model for five epochs, a deliberately limited
compute budget. All nine strategies use the same SimpleCNN backbone with dropout
`0.3`, the same SGD settings, and the same initial split for a given seed. More
seeds can be added to `seeds` before running experiments.

Learning Loss preserves the milestone implementation: a separate module predicts
per-example classification losses using MSE and detached classifier features.
This is a simplified baseline, not a reproduction of the paper's ranking-loss
training procedure. VAAL and the unfinished TypiClust implementation are not part
of this nine-strategy protocol.

Each run creates `reports/benchmark_v1/<strategy>/seed_<seed>/` with `metrics.csv`,
`config.json`, `selected_indices.json`, and `environment.json`. Results are saved
after every completed round. Existing run directories are never overwritten;
choose a new `output_dir` to rerun the same strategy and seed. The same applies to
`reports/smoke/` when rerunning the smoke configuration.

CSV rounds are zero-based. `query_seconds` measures the acquisition that produced
the current round's labeled pool, including feature extraction and selection. It
is zero for the initial pool. Training time excludes model construction and
evaluation. CUDA timings are synchronized.

Reproducibility is targeted within the same software/hardware environment, not
bit-for-bit equality between CPU and CUDA or between PyTorch versions. The
environment file records installed package versions, device, thread count, Git
commit, and working-tree state, excluding the configured output directory.

## Project structure

- `data/` — datasets and splits
- `src/alcv/` — source code
- `configs/` — experiment configs
- `models/` — trained models
- `reports/` — results, figures, tables
- `scripts/` — runnable scripts
