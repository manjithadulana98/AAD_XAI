"""Evaluation script: accuracy vs decision-window length + confidence intervals.

Usage examples::

    # Evaluate across all window lengths on synthetic data (trains on-the-fly)
    python -m aad_xai.evaluate --synthetic --model cnn --seeds 3 --epochs 5

    # Evaluate pre-trained checkpoints (provide output dir from a training run)
    python -m aad_xai.evaluate --runs-dir runs --model cnn

    # Run XAI (IG) on a trained checkpoint
    python -m aad_xai.evaluate --synthetic --model cnn --xai ig --seeds 1 --epochs 5
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import RunConfig, TrainConfig
from .data.base import Trial
from .data.splits import Record, Split, subject_independent_split, assert_no_leakage
from .data.windowing import assert_no_cross_split_overlap
from .data.torch_dataset import WindowedEEGDataset
from .data.synthetic_dataset import SyntheticDataset
from .models.aadnet import AADNet
from .models.stgcn import STGCN
from .models.trf_baseline import TRFDecoder
from .utils.seed import seed_everything
from .utils.metrics import accuracy, bootstrap_ci
from .utils.logging import save_json, get_run_dir
from .xai.integrated_gradients import ig_attribution
from .xai.faithfulness import deletion_curve, insertion_curve


# ======================================================================== #
#  Helpers
# ======================================================================== #

def _load_dataset(cfg: RunConfig):
    if cfg.dataset == "synthetic":
        return SyntheticDataset(seed=cfg.split.seed)
    elif cfg.dataset == "kul":
        from .data.kul_dataset import KULeuvenDataset
        load_audio = cfg.train.model == "trf"
        return KULeuvenDataset(
            root=cfg.dataset_root,
            preprocess=cfg.preprocess,
            load_audio=load_audio,
        )
    elif cfg.dataset == "dtu":
        from .data.dtu_dataset import DTUDataset
        return DTUDataset(root=cfg.dataset_root)
    raise ValueError(cfg.dataset)


def _build_model(name: str, n_channels: int) -> nn.Module:
    if name == "cnn":
        return AADNet(n_channels=n_channels)
    elif name == "stgcn":
        return STGCN(n_channels=n_channels)
    raise ValueError(name)


def _split_trials(trials, split):
    s_tr, s_v, s_te = set(split.train), set(split.val), set(split.test)
    return (
        [t for t in trials if t.subject_id in s_tr],
        [t for t in trials if t.subject_id in s_v],
        [t for t in trials if t.subject_id in s_te],
    )


# ======================================================================== #
#  TRF evaluation
# ======================================================================== #

def _eval_trf_window(trials, split, window_s, seed) -> float:
    """Train and evaluate TRF baseline for a single window / seed."""
    from .train import train_trf
    run_dir = get_run_dir("eval_tmp", "trf", seed, window_s)
    res = train_trf(trials, split, window_s, run_dir)
    return res["test_accuracy"]


# ======================================================================== #
#  Deep model evaluation
# ======================================================================== #

def _eval_deep_window(cfg, trials, split, window_s, seed) -> float:
    """Train-and-evaluate a deep model for one (window, seed) pair."""
    from .train import train_deep, _prepare_data, _load_dataset
    dataset = _load_dataset(cfg)
    _, _, ds_tr, ds_v, ds_te = _prepare_data(dataset, cfg, window_s)
    run_dir = get_run_dir("eval_tmp", cfg.train.model, seed, window_s)
    res = train_deep(cfg, ds_tr, ds_v, ds_te, seed, run_dir)
    return res["test_accuracy"]


def _eval_deep_from_checkpoint(ckpt_path: Path, ds_test, device) -> float:
    """Load a checkpoint and compute test accuracy."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    cfg_raw = ckpt.get("config", {})
    model_name = cfg_raw.get("train", {}).get("model", "cnn")
    n_channels = ds_test[0][0].shape[0]
    model = _build_model(model_name, n_channels).to(device)
    model.load_state_dict(state)
    model.eval()

    dl = DataLoader(ds_test, batch_size=64, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = torch.as_tensor(yb, dtype=torch.long, device=device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return correct / max(total, 1)


# ======================================================================== #
#  XAI evaluation
# ======================================================================== #

def _run_xai(cfg, trials, split, window_s, seed, device):
    """Run Integrated Gradients + deletion/insertion on the trained model."""
    dataset = _load_dataset(cfg)
    from .train import _prepare_data
    _, _, ds_tr, ds_v, ds_te = _prepare_data(dataset, cfg, window_s)

    # Quick train
    from .train import train_deep
    run_dir = get_run_dir("eval_tmp", cfg.train.model, seed, window_s)
    train_deep(cfg, ds_tr, ds_v, ds_te, seed, run_dir)

    # Load best model
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    n_ch = ds_te[0][0].shape[0]
    model = _build_model(cfg.train.model, n_ch).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Take a small batch from test set
    dl = DataLoader(ds_te, batch_size=16, shuffle=False)
    xb, yb = next(iter(dl))
    xb = xb.to(device).requires_grad_(True)
    yb = torch.as_tensor(yb, dtype=torch.long, device=device)

    # IG attribution
    attr = ig_attribution(model, xb, target=1, steps=32)
    print(f"  IG attribution shape: {attr.shape}  mean |attr|: {attr.abs().mean().item():.6f}")

    # Deletion & insertion curves
    del_curve = deletion_curve(model, xb, yb, attr, steps=10)
    ins_curve = insertion_curve(model, xb, yb, attr, steps=10)
    print(f"  Deletion curve: {[f'{c:.3f}' for c in del_curve]}")
    print(f"  Insertion curve: {[f'{c:.3f}' for c in ins_curve]}")

    xai_results = {
        "ig_mean_abs": float(attr.abs().mean().item()),
        "deletion_curve": del_curve,
        "insertion_curve": ins_curve,
    }
    save_json(run_dir / "xai_results.json", xai_results)
    print(f"  [OK] XAI results saved to {run_dir / 'xai_results.json'}")
    return xai_results


# ======================================================================== #
#  Plotting
# ======================================================================== #

def _plot_accuracy_vs_window(
    window_lengths: Sequence[float],
    results: dict[float, tuple[float, float, float]],
    model_name: str,
    out_path: Path,
) -> None:
    """Plot accuracy ± CI vs window length."""
    wl = sorted(results.keys())
    means = [results[w][0] for w in wl]
    los = [results[w][1] for w in wl]
    his = [results[w][2] for w in wl]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        wl, means,
        yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
        marker="o", capsize=4, linewidth=2,
    )
    ax.set_xlabel("Decision window (s)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"{model_name.upper()} - Accuracy vs Window Length (95% CI)")
    ax.set_xscale("log")
    ax.set_xticks(wl)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="chance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] saved to {out_path}")


# ======================================================================== #
#  Main entry-point
# ======================================================================== #

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate AAD models across window lengths.")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--dataset", type=str, default=None, choices=["synthetic", "kul", "dtu"],
                    help="Dataset name (overrides --synthetic).")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Path to dataset root (e.g. data/KULeuven).")
    ap.add_argument("--model", type=str, default="cnn", choices=["trf", "cnn", "stgcn"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str, default="eval_results")
    ap.add_argument("--runs-dir", type=str, default=None,
                    help="If set, load checkpoints from this dir instead of training.")
    ap.add_argument("--xai", type=str, default=None, choices=["ig"],
                    help="Run XAI evaluation (IG) instead of accuracy sweep.")
    ap.add_argument("--windows", type=str, default="1,5,10,60",
                    help="Comma-separated window lengths in seconds.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    window_lengths = [float(w) for w in args.windows.split(",")]

    # Determine dataset
    if args.dataset:
        ds_name = args.dataset
    elif args.synthetic:
        ds_name = "synthetic"
    elif args.data_dir:
        ds_name = "kul"
    else:
        ds_name = "kul"

    ds_root = args.data_dir or ("data/KULeuven" if ds_name == "kul" else "data_raw")

    cfg = RunConfig(
        train=TrainConfig(
            model=args.model, epochs=args.epochs,
            num_seeds=args.seeds, device=args.device,
        ),
        dataset=ds_name,
        dataset_root=ds_root,
        output_dir=args.output,
    )

    print("====================================================")
    print(f"  AAD-XAI Evaluation  |  model={args.model}  dataset={cfg.dataset}")
    print("====================================================")

    dataset = _load_dataset(cfg)
    trials = list(dataset.trials())
    subject_ids = sorted({t.subject_id for t in trials})
    split = subject_independent_split(subject_ids, cfg.split.train_frac, cfg.split.val_frac, cfg.split.seed)

    records = [Record(subject_id=t.subject_id, trial_id=t.trial_id, group_id=t.group_id) for t in trials]
    assert_no_leakage(records, split)

    # --- XAI mode ---
    if args.xai:
        print(f"\n> Running XAI ({args.xai}) with window={window_lengths[0]}s")
        _run_xai(cfg, trials, split, window_lengths[0], cfg.split.seed, device)
        return

    # --- Accuracy vs window length sweep ---
    all_results: dict[float, tuple[float, float, float]] = {}
    raw_accs: dict[float, list[float]] = {}

    for wl in window_lengths:
        print(f"\n> Window {wl}s")
        accs = []
        for si in range(args.seeds):
            seed = cfg.split.seed + si
            if args.model == "trf":
                acc = _eval_trf_window(trials, split, wl, seed)
            else:
                acc = _eval_deep_window(cfg, trials, split, wl, seed)
            accs.append(acc)

        mean, lo, hi = bootstrap_ci(np.array(accs))
        all_results[wl] = (mean, lo, hi)
        raw_accs[wl] = accs
        print(f"  -> mean={mean:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "accuracy_vs_window.json", {
        "model": args.model,
        "window_lengths": window_lengths,
        "results": {str(k): {"mean": v[0], "ci_low": v[1], "ci_high": v[2]} for k, v in all_results.items()},
        "raw_accuracies": {str(k): v for k, v in raw_accs.items()},
    })

    _plot_accuracy_vs_window(window_lengths, all_results, args.model, out_dir / "accuracy_vs_window.png")
    print("\n[OK] Evaluation complete.")


if __name__ == "__main__":
    main()
