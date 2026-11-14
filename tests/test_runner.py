import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from alcv.active_learning import runner
from alcv.evaluation.evaluation import compute_accuracy_and_cross_entropy_loss


torch.set_num_threads(1)


def small_config(output_dir):
    path = Path(__file__).resolve().parents[1] / "configs/smoke.yaml"
    config = yaml.safe_load(path.read_text())
    config["output_dir"] = str(output_dir)
    return config


def test_all_strategies_run_two_rounds(tmp_path):
    config = small_config(tmp_path)
    initial_indices = None

    for strategy in runner.STRATEGIES:
        path = runner.run_experiment(config, strategy, 1, "cpu")
        with (path / "metrics.csv").open() as file:
            rows = list(csv.DictReader(file))
        history = json.loads((path / "selected_indices.json").read_text())

        assert len(rows) == 2
        assert [int(row["labeled_size"]) for row in rows] == [8, 12]
        assert [int(row["unlabeled_size"]) for row in rows] == [16, 12]
        assert float(rows[0]["query_seconds"]) == 0
        assert float(rows[1]["query_seconds"]) > 0
        assert all(0 <= float(row["test_accuracy"]) <= 1 for row in rows)
        assert (path / "config.json").is_file()
        assert (path / "environment.json").is_file()

        if initial_indices is None:
            initial_indices = history["initial_indices"]
        assert history["initial_indices"] == initial_indices
        assert len(history["acquisitions"]) == 1
        selected = history["acquisitions"][0]["selected_indices"]
        assert len(selected) == len(set(selected)) == 4
        assert not set(initial_indices) & set(selected)


def test_query_rng_does_not_change_next_model_initialization(tmp_path, monkeypatch):
    real_build, real_select = runner.build_model, runner.select_samples
    initial_weights = []

    def record_initialization(config, device):
        model = real_build(config, device)
        initial_weights.append(next(model.parameters()).detach().cpu().clone())
        return model

    def noisy_select(*args):
        random.random()
        np.random.random(100)
        torch.rand(100)
        return real_select(*args)

    monkeypatch.setattr(runner, "build_model", record_initialization)
    runner.run_experiment(small_config(tmp_path / "normal"), "random", 1, "cpu")
    monkeypatch.setattr(runner, "select_samples", noisy_select)
    runner.run_experiment(small_config(tmp_path / "noisy"), "random", 1, "cpu")

    assert torch.equal(initial_weights[0], initial_weights[2])
    assert torch.equal(initial_weights[1], initial_weights[3])


def test_existing_run_is_not_overwritten(tmp_path):
    config = small_config(tmp_path)
    runner.run_experiment(config, "random", 1, "cpu")
    with pytest.raises(FileExistsError):
        runner.run_experiment(config, "random", 1, "cpu")


def test_evaluation_rejects_empty_loader():
    dataset = TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError, match="empty"):
        compute_accuracy_and_cross_entropy_loss(torch.nn.Linear(2, 2), DataLoader(dataset), "cpu")
