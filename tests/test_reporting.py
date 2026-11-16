import csv
import json
from pathlib import Path

import pytest
import yaml

from alcv.reporting.benchmark import build_report


def write_run(root, strategy, seed=1, final_accuracy=0.5):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/smoke.yaml").read_text())
    config.update(strategy=strategy, seed=seed, device="cpu")
    path = root / strategy / f"seed_{seed}"
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    rows = []
    for round_idx, accuracy in enumerate((0.25, final_accuracy)):
        size = 8 + round_idx * 4
        rows.append(dict(strategy=strategy, seed=seed, round=round_idx, labeled_size=size, unlabeled_size=24-size, test_accuracy=accuracy, train_seconds=1, query_seconds=round_idx * 0.1))
    with (path / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_report_averages_seeds_and_writes_files(tmp_path):
    for strategy in ("random", "entropy"):
        write_run(tmp_path, strategy, 1, 0.5)
        write_run(tmp_path, strategy, 2, 0.7)
    output = build_report(tmp_path)
    with (output / "summary.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert all(float(row["test_accuracy_mean"]) == pytest.approx(0.6) for row in rows)
    assert all(float(row["test_accuracy_std"]) == pytest.approx(0.1414213562) for row in rows)
    assert (output / "learning_curves.png").stat().st_size > 0
    assert (output / "summary.md").is_file()


def test_report_rejects_incomplete_run(tmp_path):
    path = write_run(tmp_path, "random") / "metrics.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="Incomplete"):
        build_report(tmp_path)


def test_report_can_select_one_seed(tmp_path):
    for seed, accuracy in ((1, 0.5), (2, 0.6), (3, 0.7)):
        write_run(tmp_path, "random", seed, accuracy)
    for seed, accuracy in ((1, 0.5), (2, 0.6), (3, 0.7)):
        output = build_report(tmp_path, seed=seed)
        assert output == tmp_path / f"summary_seed_{seed}"
        with (output / "summary.csv").open() as file:
            row, = list(csv.DictReader(file))
        assert int(row["n_seeds"]) == 1
        assert float(row["test_accuracy_mean"]) == pytest.approx(accuracy)
        assert row["test_accuracy_std"] == ""
        assert (output / "learning_curves.png").stat().st_size > 0
    with pytest.raises(ValueError, match="No runs found"):
        build_report(tmp_path, seed=4)


def test_report_rejects_mixed_protocols(tmp_path):
    write_run(tmp_path, "random")
    path = write_run(tmp_path, "entropy") / "config.json"
    config = json.loads(path.read_text())
    config["epochs"] = 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="different experiment protocols"):
        build_report(tmp_path)
