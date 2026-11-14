import argparse
from pathlib import Path

import torch
import yaml

from alcv.active_learning.runner import STRATEGIES, load_datasets, resolve_device, run_experiment, validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run reproducible active learning experiments")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/benchmark.yaml")
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        parser.error("The YAML config must contain a mapping")
    if args.strategy is not None:
        config["strategies"] = [args.strategy]
    if args.seed is not None:
        config["seeds"] = [args.seed]
    for name in ("strategies", "seeds"):
        if not isinstance(config.get(name), list) or not config[name]:
            parser.error(f"{name} must be a nonempty list")
        if len(config[name]) != len(set(config[name])):
            parser.error(f"{name} must not contain duplicates")
    for name in ("data_dir", "output_dir"):
        config[name] = str((PROJECT_ROOT / config[name]).resolve())

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = resolve_device(device)
    except ValueError as error:
        parser.error(str(error))

    jobs = [(strategy, seed) for strategy in config["strategies"] for seed in config["seeds"]]
    for strategy, seed in jobs:
        validate_config(config, strategy, seed)
        path = Path(config["output_dir"]) / strategy / f"seed_{seed}"
        if path.exists():
            parser.error(f"Run directory already exists: {path}. Choose a different output_dir or seed.")

    train_dataset, test_dataset = load_datasets(config)
    for strategy, seed in jobs:
        path = run_experiment(config, strategy, seed, device, train_dataset, test_dataset)
        print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
