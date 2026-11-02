"""Train one experiment across seeds. Resumes from the last epoch if a run was interrupted.

python -m ml.train --experiment shipped --seeds 0 1 2 3 4
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml import data
from ml.gravity import FEATURES
from ml.metrics import summarize
from ml.model import ActivityLSTM
from ml.rotation import random_rotations, rotate_windows, small_rotations

RUNS = Path("runs")
CPU_THREADS = 4


@dataclass(frozen=True)
class Config:
    channels: tuple[str, ...] = data.DEVICE_CHANNELS
    split: str = "subject"
    rotate: bool = False
    jitter_degrees: float = 0.0
    features: str = ""
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 10
    grad_clip: float = 1.0


SHIPPED = Config(dropout=0.5, jitter_degrees=30.0)

EXPERIMENTS = {
    # The first run, before tuning. Its test scores were seen, so it is reported but never picked.
    "baseline": Config(),
    "shipped": SHIPPED,
    "all9": replace(SHIPPED, channels=data.ALL_CHANNELS),
    "leaky_random_split": replace(SHIPPED, split="random"),
    "rotation_augmented": replace(SHIPPED, jitter_degrees=0.0, rotate=True),
    # Pose-independent channels, compared on validation subjects only. A rotation of the phone
    # cancels in these, so rotation jitter has nothing left to do.
    "gravity_tune_aligned4": Config(dropout=0.5, features="aligned4"),
    "gravity_tune_aligned3": Config(dropout=0.5, features="aligned3"),
    "gravity_tune_log": Config(dropout=0.5, features="aligned4_log"),
    "gravity_tune_log_dropout3": Config(dropout=0.3, features="aligned4_log"),
    "gravity_tune_dynamic3": Config(dropout=0.5, features="dynamic3"),
    "gravity_tune_dynamic3_log": Config(dropout=0.5, features="dynamic3_log"),
    # The eligible winner on validation, and the ineligible one kept to show what the leak is worth.
    "gravity_final": Config(dropout=0.5, features="dynamic3"),
    "gravity_leaky_comparison": Config(dropout=0.5, features="aligned3"),
    # Tuning candidates, compared on validation subjects only.
    "tune_dropout5": Config(dropout=0.5),
    "tune_slow": Config(dropout=0.5, lr=3e-4),
    "tune_jitter15": Config(dropout=0.5, jitter_degrees=15.0),
    "tune_jitter30": SHIPPED,
    "tune_hidden32": Config(hidden=32, jitter_degrees=15.0),
    "tune_1layer": Config(layers=1, dropout=0.0, jitter_degrees=15.0),
}


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_splits(config: Config, seed: int) -> tuple[data.Split, data.Split, data.Split]:
    train_part = data.load_part("train", config.channels)
    test_part = data.load_part("test", config.channels)
    if config.split == "random":
        return data.random_window_split(train_part, test_part, seed)
    fit, validation = data.subject_split(train_part)
    return fit, validation, test_part


def predict(model: nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = [model(torch.from_numpy(b)) for b in np.array_split(x, len(x) // batch_size + 1)]
    return torch.cat(logits).argmax(dim=-1).numpy()


def train_seed(name: str, config: Config, seed: int, score_test: bool = True) -> dict:
    run_dir = RUNS / name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit, validation, test = load_splits(config, seed)
    if config.features:
        convert = FEATURES[config.features]
        fit, validation, test = (
            data.Split(convert(split.x), split.y, split.subject)
            for split in (fit, validation, test)
        )
    n_channels = fit.x.shape[-1]
    mean, std = data.channel_stats(fit.x)

    def scaled(x: np.ndarray) -> np.ndarray:
        return ((x - mean) / std).astype(np.float32)

    x_validation, x_test = scaled(validation.x), scaled(test.x)
    model = ActivityLSTM(n_channels, config.hidden, config.layers, config.dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.max_epochs)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch, best_f1, best_epoch, best_state, history = 0, -1.0, -1, None, []
    latest_path = run_dir / "latest.pt"
    if latest_path.exists():
        state = torch.load(latest_path, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        start_epoch, best_f1, best_epoch = state["epoch"] + 1, state["best_f1"], state["best_epoch"]
        best_state, history = state["best_state"], state["history"]

    started = time.time()
    for epoch in range(start_epoch, config.max_epochs):
        if epoch - best_epoch > config.patience and best_epoch >= 0:
            break
        model.train()
        order = rng.permutation(len(fit))
        total_loss = 0.0
        for first in range(0, len(order), config.batch_size):
            batch = order[first : first + config.batch_size]
            x = fit.x[batch]
            if config.rotate:
                x = rotate_windows(x, random_rotations(len(batch), rng))
            elif config.jitter_degrees:
                x = rotate_windows(x, small_rotations(len(batch), config.jitter_degrees, rng))
            logits = model(torch.from_numpy(scaled(x)))
            loss = loss_fn(logits, torch.from_numpy(fit.y[batch]))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()

        f1 = summarize(validation.y, predict(model, x_validation), validation.subject)["macro_f1"]
        history.append({"epoch": epoch, "loss": total_loss / len(fit), "val_macro_f1": f1})
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "epoch": epoch,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "history": history,
            },
            latest_path,
        )
        print(
            f"{name} seed={seed} epoch={epoch} loss={total_loss / len(fit):.4f} "
            f"val_macro_f1={f1:.4f} elapsed={time.time() - started:.0f}s",
            flush=True,
        )

    model.load_state_dict(best_state)
    atomic_save(
        {
            "model": best_state,
            "config": asdict(config),
            "mean": mean,
            "std": std,
            "seed": seed,
            "n_channels": n_channels,
        },
        run_dir / "best.pt",
    )
    result = {
        "experiment": name,
        "seed": seed,
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "train_seconds": time.time() - started,
        "history": history,
    }
    if score_test:
        result["test"] = summarize(test.y, predict(model, x_test), test.subject)
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, result_path)
    latest_path.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="leave the test subjects unscored, as every tuning run must",
    )
    args = parser.parse_args()
    torch.set_num_threads(CPU_THREADS)
    for seed in args.seeds:
        config = EXPERIMENTS[args.experiment]
        result = train_seed(args.experiment, config, seed, score_test=not args.validation_only)
        print(
            f"DONE {args.experiment} seed={seed} best_epoch={result['best_epoch']} "
            f"val_macro_f1={result['val_macro_f1']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
