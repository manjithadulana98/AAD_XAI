"""Training entry-point for AAD models.

Usage examples::

    # Synthetic smoke-test (no dataset required)
    python -m aad_xai.train --synthetic --epochs 2

    # Train CNN on synthetic data, 3 seeds, window 1s
    python -m aad_xai.train --synthetic --model cnn --seeds 3 --window 1.0

    # Train TRF baseline on synthetic data
    python -m aad_xai.train --synthetic --model trf

    # Train ST-GCN on synthetic data
    python -m aad_xai.train --synthetic --model stgcn --epochs 5

    # Load config from JSON
    python -m aad_xai.train --config runs/config.json
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import RunConfig, PreprocessConfig
from .data.base import BaseDataset, Trial
from .data.splits import Record, Split, subject_independent_split, assert_no_leakage
from .data.windowing import assert_no_cross_split_overlap
from .data.torch_dataset import WindowedEEGDataset, WindowedEEGAudioDataset
from .data.synthetic_dataset import SyntheticDataset
from .models.aadnet import AADNet
from .models.aadnet_external import ExternalAADNet
from .models.stgcn import STGCN
from .models.trf_baseline import TRFDecoder
from .utils.seed import seed_everything
from .utils.metrics import accuracy
from .utils.logging import save_json, get_run_dir, log_run_metadata


# ======================================================================== #
#  Dataset factory
# ======================================================================== #

def _load_dataset(cfg: RunConfig) -> BaseDataset:
    """Return the appropriate BaseDataset subclass."""
    if cfg.dataset == "synthetic":
        return SyntheticDataset(seed=cfg.split.seed)
    elif cfg.dataset == "kul":
        from .data.kul_dataset import KULeuvenDataset
        # TRF needs envelopes; deep models only use EEG
        load_audio = cfg.train.model in {"trf", "aadnet_ext"}
        return KULeuvenDataset(
            root=cfg.dataset_root,
            preprocess=cfg.preprocess,
            load_audio=load_audio,
        )
    elif cfg.dataset == "dtu":
        from .data.dtu_dataset import DTUDataset
        return DTUDataset(root=cfg.dataset_root)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


# ======================================================================== #
#  Data pipeline: trials → split → windowed datasets
# ======================================================================== #

def _prepare_data(
    dataset: BaseDataset,
    cfg: RunConfig,
    window_s: float,
) -> tuple[list[Trial], Split, Dataset, Dataset, Dataset]:
    """Load trials, split by subject, build windowed datasets, assert no leakage."""
    trials = list(dataset.trials())
    if len(trials) == 0:
        raise RuntimeError("Dataset returned 0 trials — check your data loader.")

    subject_ids = sorted({t.subject_id for t in trials})
    split = subject_independent_split(
        subject_ids,
        train_frac=cfg.split.train_frac,
        val_frac=cfg.split.val_frac,
        seed=cfg.split.seed,
    )

    # Build leakage records and assert
    records = [Record(subject_id=t.subject_id, trial_id=t.trial_id, group_id=t.group_id) for t in trials]
    assert_no_leakage(records, split)

    # Partition trials by split
    train_set, val_set, test_set = set(split.train), set(split.val), set(split.test)
    trials_train = [t for t in trials if t.subject_id in train_set]
    trials_val = [t for t in trials if t.subject_id in val_set]
    trials_test = [t for t in trials if t.subject_id in test_set]

    overlap_s = cfg.window.overlap_s

    ds_cls = WindowedEEGAudioDataset if cfg.train.model == "aadnet_ext" else WindowedEEGDataset
    ds_train = ds_cls(trials_train, window_s=window_s, overlap_s=overlap_s)
    ds_val = ds_cls(trials_val, window_s=window_s, overlap_s=0.0)
    ds_test = ds_cls(trials_test, window_s=window_s, overlap_s=0.0)

    # Defence-in-depth: no window overlap across splits
    assert_no_cross_split_overlap(
        ds_train.window_indices, ds_val.window_indices, ds_test.window_indices
    )

    print(f"  Subjects  train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    print(f"  Windows   train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")

    return trials, split, ds_train, ds_val, ds_test


# ======================================================================== #
#  TRF baseline training
# ======================================================================== #

def train_trf(
    trials: list[Trial],
    split: Split,
    window_s: float,
    run_dir: Path,
    max_train_seconds: float = 1200.0,
) -> dict:
    """Train and evaluate the TRF (ridge) baseline.

    For each test trial, reconstruct attended & unattended envelopes and
    decide based on correlation margin.

    Parameters
    ----------
    max_train_seconds : float
        Approximate cap on total training audio/EEG seconds used for fitting (memory guard).
    """
    from .models.trf_baseline import _safe_corrcoef

    train_subjects = set(split.train)
    test_subjects = set(split.test)

    train_trials = [t for t in trials if t.subject_id in train_subjects]
    test_trials = [t for t in trials if t.subject_id in test_subjects]

    # Filter to trials with audio
    train_trials = [t for t in train_trials if t.audio_a is not None and t.audio_b is not None]
    test_trials = [t for t in test_trials if t.audio_a is not None and t.audio_b is not None]

    if not train_trials:
        print("  [WARN] No training trials with audio envelopes — skipping TRF.")
        return {"test_accuracy": 0.0, "n_test_trials": 0}

    if max_train_seconds <= 0:
        raise ValueError("max_train_seconds must be > 0")

    # Fit one TRF decoder on training data using attended envelope per trial.
    # Candidate ordering is fixed as audio_a=left, audio_b=right.
    decoder = TRFDecoder(tmin_s=0.0, tmax_s=0.4, alpha=100.0)

    # Memory-safe: sample short segments across training trials up to max_train_seconds.
    rng = np.random.default_rng(42)
    sfreq = float(train_trials[0].sfreq)
    seg_len = int(round(float(window_s) * sfreq))
    if seg_len <= 0:
        return {"test_accuracy": 0.0, "n_test_trials": 0}

    max_samples = int(round(float(max_train_seconds) * sfreq))
    n_segments = max(1, int(np.ceil(max_samples / seg_len)))

    eeg_segs: list[np.ndarray] = []
    env_segs: list[np.ndarray] = []
    attempts = 0
    while len(eeg_segs) < n_segments and attempts < n_segments * 20:
        attempts += 1
        t = train_trials[int(rng.integers(0, len(train_trials)))]
        n = int(t.eeg.shape[1])
        if n < seg_len:
            continue
        start = int(rng.integers(0, n - seg_len + 1))
        stop = start + seg_len
        eeg_segs.append(t.eeg[:, start:stop])
        env = (t.audio_a if int(t.label) == 0 else t.audio_b)
        env_segs.append(np.asarray(env[start:stop], dtype=np.float32))

    if not eeg_segs:
        return {"test_accuracy": 0.0, "n_test_trials": 0}

    eeg_cat = np.concatenate(eeg_segs, axis=1)
    env_cat = np.concatenate(env_segs, axis=0)
    decoder.fit(eeg_cat, env_cat, sfreq)

    # Evaluate on test set: predict attended candidate among (audio_a, audio_b)
    correct, total = 0, 0
    for t in test_trials:
        pred_env = decoder.predict(t.eeg)
        corr_a = _safe_corrcoef(pred_env, t.audio_a)
        corr_b = _safe_corrcoef(pred_env, t.audio_b)
        pred_label = 0 if corr_a > corr_b else 1
        if pred_label == int(t.label):
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    result = {"test_accuracy": acc, "n_test_trials": total}
    save_json(run_dir / "results.json", result)
    print(f"  TRF test accuracy: {acc:.4f} ({total} trials, {correct} correct)")
    return result


# ======================================================================== #
#  Deep model training loop
# ======================================================================== #

def _build_model(model_name: str, n_channels: int, device: torch.device, window_samples: int) -> nn.Module:
    if model_name == "cnn":
        return AADNet(n_channels=n_channels).to(device)
    elif model_name == "stgcn":
        return STGCN(n_channels=n_channels).to(device)
    elif model_name == "aadnet_ext":
        return ExternalAADNet(n_channels=n_channels, window_samples=window_samples).to(device)
    else:
        raise ValueError(f"Unknown deep model: {model_name}")


def _eval_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, model_name: str):
    """Run one evaluation pass; return (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if model_name == "aadnet_ext":
                xb, envb, yb = batch
                xb = xb.to(device)
                envb = envb.to(device)
                yb = torch.as_tensor(yb, dtype=torch.long, device=device)
                logits = model(xb, envb)
            else:
                xb, yb = batch
                xb = xb.to(device)
                yb = torch.as_tensor(yb, dtype=torch.long, device=device)
                logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_deep(
    cfg: RunConfig,
    ds_train: Dataset,
    ds_val: Dataset,
    ds_test: Dataset,
    seed: int,
    run_dir: Path,
) -> dict:
    """Full training loop with early stopping for a deep model (CNN / ST-GCN)."""
    seed_everything(seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    n_channels = ds_train[0][0].shape[0]
    window_samples = ds_train[0][0].shape[1]
    model = _build_model(cfg.train.model, n_channels, device, window_samples)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    dl_train = DataLoader(ds_train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, cfg.train.epochs + 1):
        # -- Train --
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        pbar = tqdm(dl_train, desc=f"  [seed={seed}] epoch {epoch}/{cfg.train.epochs}", leave=False)
        for batch in pbar:
            if cfg.train.model == "aadnet_ext":
                xb, envb, yb = batch
                xb = xb.to(device)
                envb = envb.to(device)
                yb = torch.as_tensor(yb, dtype=torch.long, device=device)
            else:
                xb, yb = batch
                xb = xb.to(device)
                envb = None
                yb = torch.as_tensor(yb, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, envb) if cfg.train.model == "aadnet_ext" else model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
            running_correct += (logits.argmax(1) == yb).sum().item()
            running_total += xb.size(0)
            pbar.set_postfix(loss=running_loss / max(running_total, 1), acc=running_correct / max(running_total, 1))

        # -- Validate --
        val_loss, val_acc = _eval_epoch(model, dl_val, loss_fn, device, cfg.train.model)
        print(f"    epoch {epoch}: val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Save checkpoint
    ckpt_path = run_dir / "best_model.pt"
    torch.save({"state_dict": best_state or model.state_dict(), "config": asdict(cfg)}, ckpt_path)

    # -- Test --
    test_loss, test_acc = _eval_epoch(model, dl_test, loss_fn, device, cfg.train.model)
    result = {"test_accuracy": test_acc, "test_loss": test_loss, "best_val_loss": best_val_loss}
    save_json(run_dir / "results.json", result)
    print(f"    [OK] test_acc={test_acc:.4f}  (checkpoint saved to {ckpt_path})")
    return result


# ======================================================================== #
#  CLI entry point
# ======================================================================== #

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train AAD models with leakage-safe evaluation.")
    ap.add_argument("--config", type=str, default=None, help="Path to a RunConfig JSON file.")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic dataset (smoke-test).")
    ap.add_argument("--dataset", type=str, default=None, choices=["synthetic", "kul", "dtu"],
                    help="Dataset name (overrides --synthetic).")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Path to dataset root (e.g. data/KULeuven).")
    ap.add_argument("--model", type=str, default="cnn", choices=["trf", "cnn", "stgcn", "aadnet_ext"])
    ap.add_argument("--window", type=float, default=1.0, help="Decision window length in seconds.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3, help="Number of random seeds.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str, default="runs")
    ap.add_argument("--no-preprocess", action="store_true",
                    help="Skip EEG preprocessing (use raw data as-is).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    # Build or load config
    if args.config:
        cfg = RunConfig.from_json(args.config)
    else:
        from .config import TrainConfig, SplitConfig, PreprocessConfig, WindowConfig

        # Determine dataset
        if args.dataset:
            ds_name = args.dataset
        elif args.synthetic:
            ds_name = "synthetic"
        elif args.data_dir:
            ds_name = "kul"  # default real dataset
        else:
            ds_name = "kul"

        # Determine dataset root
        ds_root = args.data_dir or ("data/KULeuven" if ds_name == "kul" else "data_raw")

        preprocess_cfg = PreprocessConfig() if not args.no_preprocess else PreprocessConfig(
            sfreq_out=128, bandpass_hz=(0.5, 8.0), reref=None,  # minimal: keep native rate
        )

        cfg = RunConfig(
            preprocess=preprocess_cfg,
            train=TrainConfig(
                model=args.model, epochs=args.epochs,
                num_seeds=args.seeds, device=args.device,
            ),
            dataset=ds_name,
            dataset_root=ds_root,
            output_dir=args.output,
        )

    print(f"====================================================")
    print(f"  AAD-XAI Training  |  model={cfg.train.model}  dataset={cfg.dataset}")
    print("====================================================")

    # Save config for reproducibility
    cfg.to_json(Path(cfg.output_dir) / "config.json")

    dataset = _load_dataset(cfg)
    window_s = args.window if not args.config else cfg.window.lengths_s[0]

    print(f"\n> Window length: {window_s}s")
    trials, split, ds_train, ds_val, ds_test = _prepare_data(dataset, cfg, window_s)

    if cfg.train.model == "trf":
        for seed_idx in range(cfg.train.num_seeds):
            seed = cfg.split.seed + seed_idx
            seed_everything(seed)
            run_dir = get_run_dir(cfg.output_dir, "trf", seed, window_s)
            log_run_metadata(run_dir, split=asdict(split), window_s=window_s, seed=seed)
            print(f"\n> TRF seed={seed}")
            train_trf(trials, split, window_s, run_dir)
    else:
        for seed_idx in range(cfg.train.num_seeds):
            seed = cfg.split.seed + seed_idx
            run_dir = get_run_dir(cfg.output_dir, cfg.train.model, seed, window_s)
            log_run_metadata(
                run_dir,
                split={"train": split.train, "val": split.val, "test": split.test},
                window_s=window_s, seed=seed, config=asdict(cfg),
            )
            print(f"\n> Training {cfg.train.model.upper()} seed={seed}")
            train_deep(cfg, ds_train, ds_val, ds_test, seed, run_dir)

    print("\n[OK] All training runs completed.")


if __name__ == "__main__":
    main()

