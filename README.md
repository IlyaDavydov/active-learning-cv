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

## Planned strategies

1. Random Sampling
2. Least Confidence
3. Margin Sampling
4. Entropy Sampling
5. BALD / MC Dropout
6. Core-Set / Diversity Sampling

## Project structure

- `data/` — datasets and splits
- `src/alcv/` — source code
- `configs/` — experiment configs
- `models/` — trained models
- `reports/` — results, figures, tables
- `scripts/` — runnable scripts
