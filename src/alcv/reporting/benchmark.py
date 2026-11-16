import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROTOCOL_FIELDS = ("dataset", "model", "dropout", "initial_labeled_size", "acquisition_size", "num_rounds", "epochs", "batch_size", "learning_rate", "momentum", "retrain_from_scratch", "deterministic", "num_workers", "num_threads", "mc_passes", "distance_chunk_size", "device")
SUMMARY_FIELDS = ("strategy", "n_seeds", "final_labeled_size", "test_accuracy_mean", "test_accuracy_std", "train_seconds_mean", "query_seconds_mean")
LABELS = {"random": "Random", "least_confidence": "Least Confidence", "margin": "Margin", "entropy": "Entropy", "mc_dropout_entropy": "MC Dropout Entropy", "bald": "BALD", "core_set": "Core-Set", "badge": "BADGE", "learning_loss": "Learning Loss (MSE)"}


def _read_run(path):
    config = json.loads(path.with_name("config.json").read_text(encoding="utf-8"))
    required = {"strategy", "seed", "round", "labeled_size", "unlabeled_size", "test_accuracy", "train_seconds", "query_seconds"}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing columns from the new runner's CSV format")
        rows = list(reader)

    strategy, seed = config["strategy"], config["seed"]
    if path.parent.name != f"seed_{seed}" or path.parent.parent.name != strategy:
        raise ValueError("Directory names do not match config.json")
    if len(rows) != config["num_rounds"]:
        raise ValueError("Incomplete run: the number of rows does not match num_rounds")
    if [int(row["round"]) for row in rows] != list(range(config["num_rounds"])):
        raise ValueError("Rounds must be consecutive and start at zero")
    if any(row["strategy"] != strategy or int(row["seed"]) != seed for row in rows):
        raise ValueError("CSV strategy/seed does not match config.json")

    budgets = [int(row["labeled_size"]) for row in rows]
    expected = [config["initial_labeled_size"] + i * config["acquisition_size"] for i in range(config["num_rounds"])]
    totals = [int(row["labeled_size"]) + int(row["unlabeled_size"]) for row in rows]
    if budgets != expected or len(set(totals)) != 1 or any(int(row["unlabeled_size"]) < 0 for row in rows):
        raise ValueError("Pool sizes do not follow the configured acquisition schedule")

    accuracy = np.array([float(row["test_accuracy"]) for row in rows])
    train_time = np.array([float(row["train_seconds"]) for row in rows])
    query_time = np.array([float(row["query_seconds"]) for row in rows])
    if not all(np.isfinite(values).all() for values in (accuracy, train_time, query_time)):
        raise ValueError("Metrics must be finite")
    if np.any((accuracy < 0) | (accuracy > 1)) or np.any(train_time < 0) or np.any(query_time < 0):
        raise ValueError("Accuracy must be in [0, 1] and timings must be non-negative")
    if query_time[0] != 0:
        raise ValueError("The initial pool must have query_seconds equal to zero")

    protocol = tuple(config[key] for key in PROTOCOL_FIELDS) + (config.get("train_size"), config.get("test_size"))
    return dict(strategy=strategy, seed=seed, budgets=budgets, accuracy=accuracy, train_seconds=float(train_time.sum()), query_seconds=float(query_time.sum()), protocol=protocol, config=config)


def build_report(input_dir, output_dir=None, seed=None):
    input_dir = Path(input_dir).resolve()
    paths = sorted(input_dir.glob("*/seed_*/metrics.csv"))
    if seed is not None:
        paths = [path for path in paths if path.parent.name == f"seed_{seed}"]
    if not paths:
        raise ValueError(f"No runs found under {input_dir}")

    runs = []
    for path in paths:
        try:
            runs.append(_read_run(path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}: {error}") from error
    if any(run["protocol"] != runs[0]["protocol"] for run in runs):
        raise ValueError("Runs use different experiment protocols or devices")

    groups = {}
    for run in runs:
        groups.setdefault(run["strategy"], []).append(run)
    reference_seeds = sorted(run["seed"] for run in next(iter(groups.values())))
    for strategy, group in groups.items():
        seeds = sorted(run["seed"] for run in group)
        if len(seeds) != len(set(seeds)) or seeds != reference_seeds:
            raise ValueError(f"{strategy}: all strategies must use the same unique seeds")

    output_dir = Path(output_dir).resolve() if output_dir else input_dir / ("summary" if seed is None else f"summary_seed_{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    summary = []
    for strategy in sorted(groups, key=lambda name: (name != "random", name)):
        group = groups[strategy]
        values = np.stack([run["accuracy"] for run in group])
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if len(group) > 1 else None
        style = {"color": "black", "linestyle": "--"} if strategy == "random" else {}
        line, = ax.plot(group[0]["budgets"], mean * 100, marker="o", markersize=3, label=LABELS.get(strategy, strategy), **style)
        if std is not None:
            ax.fill_between(group[0]["budgets"], np.maximum(0, mean - std) * 100, np.minimum(1, mean + std) * 100, color=line.get_color(), alpha=0.12)
        summary.append(dict(strategy=strategy, n_seeds=len(group), final_labeled_size=group[0]["budgets"][-1], test_accuracy_mean=float(mean[-1]), test_accuracy_std=float(std[-1]) if std is not None else "", train_seconds_mean=float(np.mean([run["train_seconds"] for run in group])), query_seconds_mean=float(np.mean([run["query_seconds"] for run in group]))))

    config = runs[0]["config"]
    title = "CIFAR-10: active learning comparison" if config["dataset"] == "cifar10" else "Synthetic data: pipeline check"
    if seed is not None:
        title += f" (seed {seed})"
    ax.set(title=title, xlabel="Labeled training images", ylabel="Test accuracy (%)", ylim=(0, 100))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    lines = ["# Benchmark summary", "", f"Dataset: {config['dataset']}. Seeds: {reference_seeds}.", f"Model: {config['model']}. Epochs per round: {config['epochs']}.", ""]
    if config["dataset"] == "synthetic":
        lines.extend(["Synthetic data is used only to check the pipeline; these are not benchmark results.", ""])
    lines.extend(["| Strategy | Seeds | Final labels | Test accuracy | Train (s) | Query (s) |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in summary:
        accuracy = f"{row['test_accuracy_mean'] * 100:.2f}%"
        if row["test_accuracy_std"] != "":
            accuracy += f" +/- {row['test_accuracy_std'] * 100:.2f} pp"
        lines.append(f"| {LABELS.get(row['strategy'], row['strategy'])} | {row['n_seeds']} | {row['final_labeled_size']} | {accuracy} | {row['train_seconds_mean']:.2f} | {row['query_seconds_mean']:.2f} |")
    lines.extend(["", "With multiple seeds, accuracy is mean +/- sample standard deviation across seeds. With one seed, variability is not estimated.", "Training and query times are totals per run, averaged across seeds; compare timings on the same hardware.", "", "![Learning curves](learning_curves.png)", ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir
