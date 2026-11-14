import csv
import importlib.metadata
import platform
import subprocess
import time
from operator import index as integer_index
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from alcv.active_learning.BALD import BALD_sampling
from alcv.active_learning.badge_sampling import badge_sampling
from alcv.active_learning.core_set_sampling import core_set_sampling
from alcv.active_learning.entropy_sampling import entropy_sampling
from alcv.active_learning.learning_loss import learning_loss
from alcv.active_learning.least_confidence import DatasetWithIndices, least_confidence
from alcv.active_learning.margin_sampling import margin_sampling
from alcv.active_learning.mc_dropout_entropy_sampling import mc_dropout_entropy_sampling
from alcv.active_learning.random_sampling import expand_labeled_pool, random_sample_indices
from alcv.evaluation.evaluation import compute_accuracy_and_cross_entropy_loss
from alcv.models.loss_prediction_module import LossPredictionModule
from alcv.models.simple_cnn import SimpleCNN
from alcv.models.simple_cnn_with_dropout import SimpleCNNWithDropout
from alcv.training.train import train_learning_loss, train_model
from alcv.utils.io import save_json
from alcv.utils.seed import seed_everything, seed_worker
from alcv.utils.update_indices import update_labeled_and_unlabeled_indices


STRATEGIES = ("random", "least_confidence", "margin", "entropy", "mc_dropout_entropy", "bald", "core_set", "badge", "learning_loss")
METRIC_FIELDS = ("strategy", "seed", "round", "labeled_size", "unlabeled_size", "train_accuracy", "train_loss", "test_accuracy", "test_loss", "train_seconds", "query_seconds")


def validate_config(config, strategy, seed):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if config["dataset"] not in ("cifar10", "synthetic"):
        raise ValueError("dataset must be cifar10 or synthetic")
    if config["model"] not in ("simple_cnn", "simple_cnn_dropout"):
        raise ValueError("Unknown model")
    if config["retrain_from_scratch"] is not True:
        raise ValueError("This runner requires retrain_from_scratch: true")
    if type(config["deterministic"]) is not bool:
        raise ValueError("deterministic must be true or false")

    for name in ("initial_labeled_size", "acquisition_size", "num_rounds", "epochs", "batch_size", "num_threads", "mc_passes", "distance_chunk_size"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if not 0 <= config["dropout"] < 1:
        raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive)")
    if not 0 < config["learning_rate"] < float("inf"):
        raise ValueError("learning_rate must be positive and finite")
    if not 0 <= config["momentum"] < 1:
        raise ValueError("momentum must be between 0 (inclusive) and 1 (exclusive)")
    if strategy in ("mc_dropout_entropy", "bald"):
        if config["model"] != "simple_cnn_dropout" or config["dropout"] == 0:
            raise ValueError(f"{strategy} requires a model with nonzero dropout")
        if config["mc_passes"] < 2:
            raise ValueError("MC dropout strategies require at least two mc_passes")
    if config["dataset"] == "synthetic":
        for name in ("train_size", "test_size"):
            if type(config[name]) is not int or config[name] < 1:
                raise ValueError(f"{name} must be a positive integer")


def load_datasets(config):
    if config["dataset"] == "cifar10":
        from alcv.data.cifar10 import load_cifar10
        return load_cifar10(config["data_dir"])

    # Fixed synthetic data, independent of the experiment's training/query seed.
    generator = torch.Generator().manual_seed(0)
    datasets = []
    for size in (config["train_size"], config["test_size"]):
        images = torch.rand(size, 3, 32, 32, generator=generator)
        labels = torch.randint(10, (size,), generator=generator)
        datasets.append(TensorDataset(images, labels))
    return tuple(datasets)


def build_model(config, device):
    if config["model"] == "simple_cnn":
        return SimpleCNN().to(device)
    return SimpleCNNWithDropout(config["dropout"]).to(device)


def resolve_device(device):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This runner supports CPU and CUDA devices")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            raise ValueError("The requested CUDA device does not exist")
    return device


def make_loader(dataset, config, seed, shuffle=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle, num_workers=config["num_workers"], generator=generator, worker_init_fn=seed_worker)


def select_samples(strategy, model, loss_model, labeled_loader, unlabeled_loader, labeled, unlabeled, config, query_seed, device):
    size = config["acquisition_size"]
    if strategy == "random":
        _, _, selected = expand_labeled_pool(labeled, unlabeled, size, seed=query_seed)
        return selected

    # Dropout uses PyTorch's RNG rather than accepting a Generator argument.
    cuda_devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(query_seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(query_seed)

        if strategy == "core_set":
            return core_set_sampling(model, labeled_loader, unlabeled_loader, size, distance_chunk_size=config["distance_chunk_size"])
        if strategy == "badge":
            return badge_sampling(model, unlabeled_loader, size, seed=query_seed)
        if strategy == "mc_dropout_entropy":
            return mc_dropout_entropy_sampling(model, unlabeled_loader, size, T=config["mc_passes"])
        if strategy == "bald":
            return BALD_sampling(model, unlabeled_loader, size, T=config["mc_passes"])
        if strategy == "learning_loss":
            return learning_loss(model, loss_model, unlabeled_loader, size)
        methods = {"least_confidence": least_confidence, "margin": margin_sampling, "entropy": entropy_sampling}
        return methods[strategy](model, unlabeled_loader, size)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _save_json(data, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_json(data, temporary)
    temporary.replace(path)


def _save_metrics(rows, path):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _environment(device, output_dir):
    root = Path(__file__).resolve().parents[3]
    git_commit, git_status = None, None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        arguments = ["git", "status", "--porcelain", "--untracked-files=normal", "--", "."]
        if output_dir.is_relative_to(root):
            arguments.append(f":(exclude){output_dir.relative_to(root).as_posix()}")
        git_status = subprocess.check_output(arguments, cwd=root, text=True, stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass

    packages = {}
    for name in ("torch", "torchvision", "numpy", "PyYAML", "matplotlib", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    return {
        "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "cuda_version": torch.version.cuda, "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "git_commit": git_commit, "git_dirty": bool(git_status) if git_status is not None else None,
        "git_status": git_status, "git_status_excluded_output_dir": str(output_dir),
    }


def run_experiment(config, strategy, seed, device="cpu", train_dataset=None, test_dataset=None):
    validate_config(config, strategy, seed)
    device = resolve_device(device)
    torch.set_num_threads(config["num_threads"])

    if (train_dataset is None) != (test_dataset is None):
        raise ValueError("Provide both train_dataset and test_dataset, or neither")
    if train_dataset is None:
        train_dataset, test_dataset = load_datasets(config)
    needed = config["initial_labeled_size"] + (config["num_rounds"] - 1) * config["acquisition_size"]
    if needed > len(train_dataset):
        raise ValueError("The requested rounds require more samples than the training dataset contains")
    if len(test_dataset) == 0:
        raise ValueError("The test dataset must not be empty")

    output_dir = Path(config["output_dir"]).expanduser().resolve()
    run_dir = output_dir / strategy / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    actual_config = dict(config)
    actual_config.pop("strategies", None)
    actual_config.pop("seeds", None)
    actual_config.update(strategy=strategy, seed=seed, device=str(device), output_dir=str(output_dir))
    actual_config["data_dir"] = str(Path(config["data_dir"]).expanduser().resolve())
    actual_config["query_seconds_semantics"] = "Incoming acquisition; zero for the initial pool"
    if strategy == "learning_loss":
        actual_config["loss_prediction_objective"] = "mse_detached_features"
    _save_json(actual_config, run_dir / "config.json")
    seed_everything(seed, config["deterministic"])
    _save_json(_environment(device, output_dir), run_dir / "environment.json")

    labeled, unlabeled = random_sample_indices(len(train_dataset), config["initial_labeled_size"], seed)
    labeled, unlabeled = labeled.tolist(), unlabeled.tolist()
    history = {"initial_indices": labeled.copy(), "acquisitions": []}
    _save_json(history, run_dir / "selected_indices.json")
    indexed_train = DatasetWithIndices(train_dataset)
    test_loader = make_loader(test_dataset, config, seed)
    rows, query_seconds = [], 0.0

    for round_idx in range(config["num_rounds"]):
        train_seed = seed + round_idx
        train_loader = make_loader(Subset(train_dataset, labeled), config, train_seed, shuffle=True)
        train_eval_loader = make_loader(Subset(train_dataset, labeled), config, train_seed)

        # Acquisition RNG use cannot affect the next model's initialization.
        seed_everything(train_seed, config["deterministic"])
        model = build_model(config, device)
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
        loss_model, loss_optimizer = None, None
        if strategy == "learning_loss":
            loss_model = LossPredictionModule().to(device)
            loss_optimizer = torch.optim.SGD(loss_model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])

        # The extra loss head must not shift the classifier's dropout RNG stream.
        seed_everything(train_seed, config["deterministic"])
        print(f"{strategy} seed={seed}: round {round_idx + 1}/{config['num_rounds']}, labeled={len(labeled)}, training", flush=True)
        _synchronize(device)
        started = time.perf_counter()
        if loss_model is None:
            train_model(model, train_loader, optimizer, config["epochs"], device)
        else:
            train_learning_loss(model, loss_model, train_loader, optimizer, loss_optimizer, config["epochs"], device)
        _synchronize(device)
        train_seconds = time.perf_counter() - started

        train_accuracy, train_loss = compute_accuracy_and_cross_entropy_loss(model, train_eval_loader, device)
        test_accuracy, test_loss = compute_accuracy_and_cross_entropy_loss(model, test_loader, device)
        rows.append(dict(strategy=strategy, seed=seed, round=round_idx, labeled_size=len(labeled), unlabeled_size=len(unlabeled), train_accuracy=train_accuracy, train_loss=train_loss, test_accuracy=test_accuracy, test_loss=test_loss, train_seconds=train_seconds, query_seconds=query_seconds))
        _save_metrics(rows, run_dir / "metrics.csv")
        print(f"  test_accuracy={test_accuracy:.4f}, train_seconds={train_seconds:.2f}, query_seconds={query_seconds:.2f}", flush=True)
        if round_idx == config["num_rounds"] - 1:
            break

        labeled_loader = make_loader(Subset(indexed_train, labeled), config, train_seed)
        unlabeled_loader = make_loader(Subset(indexed_train, unlabeled), config, train_seed)
        query_seed = seed + round_idx
        print(f"  acquiring {config['acquisition_size']} samples from {len(unlabeled)} candidates", flush=True)
        _synchronize(device)
        started = time.perf_counter()
        selected = select_samples(strategy, model, loss_model, labeled_loader, unlabeled_loader, labeled, unlabeled, config, query_seed, device)
        _synchronize(device)
        query_seconds = time.perf_counter() - started

        selected = [integer_index(index) for index in selected]
        if len(selected) != config["acquisition_size"]:
            raise ValueError("The strategy returned the wrong number of indices")
        labeled, unlabeled = update_labeled_and_unlabeled_indices(labeled, unlabeled, selected)
        if set(labeled) & set(unlabeled) or len(labeled) + len(unlabeled) != len(train_dataset):
            raise RuntimeError("Pool invariants were violated")
        history["acquisitions"].append({"after_round": round_idx, "query_seed": query_seed, "query_seconds": query_seconds, "selected_indices": selected})
        _save_json(history, run_dir / "selected_indices.json")

    return run_dir
