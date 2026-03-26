"""Run all 4 CV strategies x 3 models on the KULeuven dataset.

Usage::

    # Full run (all strategies, all models, 2s window, 15 epochs)
    python -m aad_xai.run_experiments --data-dir data/KULeuven

    # Single strategy
    python -m aad_xai.run_experiments --data-dir data/KULeuven --cv within_subject --model cnn

    # Quick test
    python -m aad_xai.run_experiments --data-dir data/KULeuven --cv loso --model cnn --epochs 5 --max-folds 2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import PreprocessConfig
from .data.base import Trial
from .data.kul_dataset import KULeuvenDataset
from .data.cv_splits import (
    CV_STRATEGIES,
    _content_group,
)
from .data.torch_dataset import WindowedEEGDataset, WindowedEEGAudioDataset
from .models.aadnet import AADNet
from .models.aadnet_external import ExternalAADNet
from .models.stgcn import STGCN
from .models.trf_baseline import TRFDecoder, _safe_corrcoef, lag_matrix
from .utils.seed import seed_everything
from .utils.logging import save_json
from .utils.metrics import binary_confusion_matrix


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


# ======================================================================== #
#  Model helpers
# ======================================================================== #

def _build_model(name: str, n_ch: int, window_samples: int, device: torch.device) -> nn.Module:
    if name == "cnn":
        return AADNet(n_channels=n_ch).to(device)
    elif name == "stgcn":
        return STGCN(n_channels=n_ch).to(device)
    elif name == "aadnet_ext":
        return ExternalAADNet(n_channels=n_ch, window_samples=window_samples).to(device)
    raise ValueError(name)


def _eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> tuple[float, list[int], list[int], list[float]]:
    model.eval()
    correct, total = 0, 0
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    p1_all: list[float] = []
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
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            y_true_all.extend(yb.cpu().tolist())
            y_pred_all.extend(preds.cpu().tolist())
            p1_all.extend(probs[:, 1].cpu().tolist())
    return correct / max(total, 1), y_true_all, y_pred_all, p1_all


# ======================================================================== #
#  Training routines
# ======================================================================== #

def _train_deep_fold(
    model_name: str,
    ds_train: Dataset,
    ds_val: Dataset,
    ds_test: Dataset,
    seed: int,
    epochs: int,
    patience: int,
    device: torch.device,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
) -> dict:
    """Train a deep model on one CV fold; return results dict."""
    seed_everything(seed)
    n_ch = ds_train[0][0].shape[0]
    window_samples = ds_train[0][0].shape[1]
    model = _build_model(model_name, n_ch, window_samples, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    dl_tr = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_loss, patience_cnt, best_state = float("inf"), 0, None

    for ep in range(1, epochs + 1):
        model.train()
        for batch in dl_tr:
            if model_name == "aadnet_ext":
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
            logits = model(xb, envb) if model_name == "aadnet_ext" else model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        vloss, vtot = 0.0, 0
        with torch.no_grad():
            for batch in dl_val:
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
                vloss += loss_fn(logits, yb).item() * xb.size(0)
                vtot += xb.size(0)
        vloss /= max(vtot, 1)

        if vloss < best_val_loss:
            best_val_loss = vloss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    test_acc, y_true, y_pred, p1 = _eval_model(model, dl_te, device, model_name)
    return {
        "test_accuracy": test_acc,
        "best_val_loss": best_val_loss,
        "epochs_run": ep,
        "y_true": y_true,
        "y_pred": y_pred,
        "p_class1": p1,
    }


def _train_trf_fold(
    train_trials: list[Trial],
    val_trials: list[Trial],
    test_trials: list[Trial],
    *,
    input_window_s: float,
    decision_window_s: float,
    overlap_s: float = 0.0,
    max_train_seconds: float = 1200.0,
    tune_alpha: bool = False,
    alpha_grid: Optional[np.ndarray] = None,
    tmin_s: float = 0.0,
    tmax_s: float = 0.5,
    alpha_metric: str = "corr",
) -> dict:
    """Train and evaluate TRF on one CV fold."""
    tr = [t for t in train_trials if t.audio_a is not None and t.audio_b is not None]
    va = [t for t in val_trials if t.audio_a is not None and t.audio_b is not None]
    te = [t for t in test_trials if t.audio_a is not None and t.audio_b is not None]
    if not tr or not te:
        return {"test_accuracy": 0.0, "n_test": 0}

    if max_train_seconds <= 0:
        raise ValueError("max_train_seconds must be > 0")

    if alpha_metric not in {"corr", "aad_acc"}:
        raise ValueError("alpha_metric must be one of {'corr','aad_acc'}")

    decoder = TRFDecoder(tmin_s=float(tmin_s), tmax_s=float(tmax_s), alpha=100.0)

    # --- Memory-safe training: sample many short segments across training trials ---
    rng = np.random.default_rng(42)
    sfreq = float(tr[0].sfreq)
    seg_len = int(round(float(input_window_s) * sfreq))
    if seg_len <= 0:
        return {"test_accuracy": 0.0, "n_test": 0}

    max_samples = int(round(float(max_train_seconds) * sfreq))

    # Conservative memory guard: cap the lag-matrix rows so X doesn't explode.
    # X has shape (n_times, n_channels * n_lags) in float32.
    n_ch = int(tr[0].eeg.shape[0])
    n_lags = int(round(decoder.tmax_s * sfreq)) - int(round(decoder.tmin_s * sfreq)) + 1
    n_features = max(1, n_ch * max(1, n_lags))
    max_x_mb = 256.0
    max_rows_by_mem = int((max_x_mb * 1024 * 1024) // (n_features * 4))
    if max_rows_by_mem > 0:
        max_samples = min(max_samples, max_rows_by_mem)

    n_segments = max(1, int(np.ceil(max_samples / seg_len)))

    eeg_segs: list[np.ndarray] = []
    env_segs: list[np.ndarray] = []
    seg_groups: list[str] = []

    # Prefer long trials; retry a bounded number of times to find valid segments.
    attempts = 0
    while len(eeg_segs) < n_segments and attempts < n_segments * 20:
        attempts += 1
        t = tr[int(rng.integers(0, len(tr)))]
        n = int(t.eeg.shape[1])
        if n < seg_len:
            continue
        start = int(rng.integers(0, n - seg_len + 1))
        stop = start + seg_len
        eeg_segs.append(t.eeg[:, start:stop])
        env = (t.audio_a if int(t.label) == 0 else t.audio_b)
        env_segs.append(np.asarray(env[start:stop], dtype=np.float32))
        seg_groups.append(str(t.trial_id))

    if not eeg_segs:
        return {"test_accuracy": 0.0, "n_test": 0}

    eeg_cat = np.concatenate(eeg_segs, axis=1)
    env_cat = np.concatenate(env_segs, axis=0)

    def _aad_decoding_accuracy(
        dec: TRFDecoder,
        trials_: list[Trial],
        *,
        win_s: float,
        step_s: float,
    ) -> float:
        win = int(round(float(win_s) * float(sfreq)))
        step = int(round(float(step_s) * float(sfreq)))
        if win <= 0 or step <= 0:
            return 0.0
        correct_ = 0
        total_ = 0
        for tt in trials_:
            if tt.audio_a is None or tt.audio_b is None:
                continue
            n_ = int(tt.eeg.shape[-1])
            true_ = int(tt.label)
            for st in range(0, max(0, n_ - win + 1), step):
                sp = st + win
                pred_env_ = dec.predict(tt.eeg[:, st:sp])
                ca = _safe_corrcoef(pred_env_, tt.audio_a[st:sp])
                cb = _safe_corrcoef(pred_env_, tt.audio_b[st:sp])
                pred_ = 0 if ca > cb else 1
                correct_ += int(pred_ == true_)
                total_ += 1
        return float(correct_ / max(total_, 1))

    # Optional: tune ridge alpha.
    # - alpha_metric='corr' uses group CV on the sampled training segments (no val split required)
    # - alpha_metric='aad_acc' uses the fold's validation trials
    if tune_alpha:
        if alpha_grid is None:
            # Wider than typical but still manageable.
            alpha_grid = np.logspace(-7, 7, 15, dtype=np.float64)

        if alpha_metric == "corr":
            # Select alpha via RidgeCV with GroupKFold, scoring by Pearson correlation.
            lags = np.arange(
                int(round(decoder.tmin_s * sfreq)),
                int(round(decoder.tmax_s * sfreq)) + 1,
            )
            decoder.lags_ = lags
            decoder.sfreq_ = float(sfreq)

            X_parts: list[np.ndarray] = []
            y_parts: list[np.ndarray] = []
            g_parts: list[np.ndarray] = []
            for eeg_seg, env_seg, gid in zip(eeg_segs, env_segs, seg_groups):
                Xs = lag_matrix(np.asarray(eeg_seg, dtype=np.float32), lags)
                ys = np.asarray(env_seg, dtype=np.float32)[: Xs.shape[0]]
                X_parts.append(Xs)
                y_parts.append(ys)
                g_parts.append(np.full((Xs.shape[0],), gid, dtype=object))

            x_tr = np.concatenate(X_parts, axis=0)
            y_tr = np.concatenate(y_parts, axis=0)
            groups = np.concatenate(g_parts, axis=0)

            decoder._X_mean, decoder._X_std = decoder._z_inplace(x_tr)
            decoder._y_mean = float(y_tr.mean())
            decoder._y_std = float(y_tr.std()) or 1.0
            y_tr_sc = (y_tr - decoder._y_mean) / decoder._y_std

            unique_groups = np.unique(groups)
            n_splits = int(min(5, len(unique_groups)))
            if n_splits < 2:
                decoder.fit(eeg_cat, env_cat, sfreq)
            else:
                cv_gen = GroupKFold(n_splits=n_splits).split(x_tr, y_tr_sc, groups=groups)

                def _pearson_scorer(est, X, y):
                    return _safe_corrcoef(est.predict(X), y)

                ridgecv = RidgeCV(
                    alphas=np.asarray(alpha_grid, dtype=np.float64).reshape(-1),
                    fit_intercept=True,
                    scoring=_pearson_scorer,
                    cv=cv_gen,
                    gcv_mode=None,
                    store_cv_results=False,
                )
                ridgecv.fit(x_tr, y_tr_sc)
                best_alpha = float(ridgecv.alpha_)

                decoder.model = Ridge(alpha=best_alpha, fit_intercept=True, solver="lsqr", copy_X=False)
                decoder.alpha = best_alpha
                decoder.model.fit(x_tr, y_tr_sc)
                decoder.best_alpha_ = best_alpha
        else:
            # Literature-style: pick alpha that maximises *decoding accuracy* on the fold's validation trials.
            # Reuse a single training design matrix for all alphas for efficiency.
            if not va:
                decoder.fit(eeg_cat, env_cat, sfreq)
            else:
                lags = np.arange(
                    int(round(decoder.tmin_s * sfreq)),
                    int(round(decoder.tmax_s * sfreq)) + 1,
                )
                decoder.lags_ = lags
                decoder.sfreq_ = float(sfreq)

                x_tr = lag_matrix(eeg_cat, lags)
                y_tr = env_cat[: x_tr.shape[0]]

                decoder._X_mean, decoder._X_std = decoder._z_inplace(x_tr)
                decoder._y_mean = float(y_tr.mean())
                decoder._y_std = float(y_tr.std()) or 1.0
                y_tr_sc = (y_tr - decoder._y_mean) / decoder._y_std

                # Determine validation scoring windows consistent with how we will evaluate.
                if float(decision_window_s) > float(input_window_s):
                    score_win_s = float(decision_window_s)
                    score_step_s = max(0.5, score_win_s / 5.0)
                else:
                    score_win_s = float(input_window_s)
                    score_step_s = max(1e-9, float(input_window_s) - float(overlap_s))

                best_alpha = float(alpha_grid[0])
                best_score = -np.inf
                best_model = None
                for a in np.asarray(alpha_grid, dtype=np.float64).reshape(-1):
                    mdl = Ridge(alpha=float(a), fit_intercept=True, solver="lsqr", copy_X=False)
                    mdl.fit(x_tr, y_tr_sc)
                    decoder.model = mdl
                    decoder.alpha = float(a)
                    score = _aad_decoding_accuracy(decoder, va, win_s=score_win_s, step_s=score_step_s)
                    if score > best_score:
                        best_score = score
                        best_alpha = float(a)
                        best_model = mdl

                assert best_model is not None
                decoder.model = best_model
                decoder.alpha = best_alpha
                decoder.best_alpha_ = best_alpha
    else:
        decoder.fit(eeg_cat, env_cat, sfreq)

    if input_window_s <= 0 or decision_window_s <= 0:
        raise ValueError("input_window_s and decision_window_s must be > 0")
    if decision_window_s < input_window_s:
        raise ValueError("decision_window_s must be >= input_window_s for TRF aggregation")
    if overlap_s < 0 or overlap_s >= input_window_s:
        raise ValueError("overlap_s must satisfy 0 <= overlap_s < input_window_s")

    base_window_rows: list[dict] = []
    for t in te:
        sfreq = float(t.sfreq)
        base_win = int(round(float(input_window_s) * sfreq))
        step = int(round((float(input_window_s) - float(overlap_s)) * sfreq))
        if base_win <= 0 or step <= 0:
            continue

        n = int(t.eeg.shape[-1])
        true_label = int(t.label)

        for start in range(0, max(0, n - base_win + 1), step):
            stop = start + base_win
            pred_env = decoder.predict(t.eeg[:, start:stop])
            corr_a = _safe_corrcoef(pred_env, t.audio_a[start:stop])
            corr_b = _safe_corrcoef(pred_env, t.audio_b[start:stop])
            # Standard AAD decoding decision: corr(attended) > corr(unattended)
            # Avoid abs(): strong *negative* correlation to unattended should not count as evidence.
            pred_label = 0 if corr_a > corr_b else 1
            p_class1 = 1.0 if corr_b >= corr_a else 0.0
            base_window_rows.append(
                {
                    "subject_id": t.subject_id,
                    "trial_id": t.trial_id,
                    "start": int(start),
                    "stop": int(stop),
                    "y_true": true_label,
                    "y_pred": int(pred_label),
                    "ref_outcome": _label_to_ab(true_label),
                    "pred_outcome": _label_to_ab(int(pred_label)),
                    "p_class1": float(p_class1),
                }
            )

    if not base_window_rows:
        return {"test_accuracy": 0.0, "n_test": 0, "n_trials": len(te)}

    # Aggregate into decision windows if requested.
    decision_window_rows: list[dict]
    if float(decision_window_s) > float(input_window_s):
        # Paper-style for linear decoders: evaluate correlation on the full decision window
        # (no need to vote/aggregate smaller windows).
        decision_window_rows = []
        for t in te:
            sfreq = float(t.sfreq)
            decision_win = int(round(float(decision_window_s) * sfreq))
            step0 = int(round(0.5 * sfreq))
            decision_step = max(step0, max(1, decision_win // 5))
            if decision_win <= 0:
                continue
            n = int(t.eeg.shape[-1])
            true_label = int(t.label)

            # Sliding decision windows as used in many AAD baselines (e.g., step=max(0.5s, L/5)).
            for start in range(0, max(0, n - decision_win + 1), decision_step):
                stop = start + decision_win
                pred_env = decoder.predict(t.eeg[:, start:stop])
                corr_a = _safe_corrcoef(pred_env, t.audio_a[start:stop])
                corr_b = _safe_corrcoef(pred_env, t.audio_b[start:stop])
                pred_label = 0 if corr_a > corr_b else 1
                p_class1 = 1.0 if corr_b >= corr_a else 0.0
                decision_window_rows.append(
                    {
                        "subject_id": t.subject_id,
                        "trial_id": t.trial_id,
                        "start": int(start),
                        "stop": int(stop),
                        "y_true": true_label,
                        "y_pred": int(pred_label),
                        "ref_outcome": _label_to_ab(true_label),
                        "pred_outcome": _label_to_ab(int(pred_label)),
                        "p_class1": float(p_class1),
                    }
                )

        # Fallback to aggregation if, for some reason, we produced no decision windows.
        if not decision_window_rows:
            sfreq = float(te[0].sfreq)
            decision_samples = int(round(float(decision_window_s) * sfreq))
            decision_window_rows = _aggregate_decision_windows_from_base_windows(
                base_window_rows,
                decision_window_samples=decision_samples,
            )

        base_acc = float(
            np.mean([int(int(r["y_true"]) == int(r["y_pred"])) for r in base_window_rows])
        )
    else:
        decision_window_rows = base_window_rows
        base_acc = float(
            np.mean([int(int(r["y_true"]) == int(r["y_pred"])) for r in decision_window_rows])
        )

    y_true = [int(r["y_true"]) for r in decision_window_rows]
    y_pred = [int(r["y_pred"]) for r in decision_window_rows]
    p1 = [float(r["p_class1"]) for r in decision_window_rows]
    test_acc = float(np.mean([int(a == b) for a, b in zip(y_true, y_pred)])) if y_true else 0.0

    trial_rows = _aggregate_trial_rows_from_windows(decision_window_rows)

    return {
        "test_accuracy": test_acc,
        "base_window_accuracy": base_acc if float(decision_window_s) > float(input_window_s) else None,
        "n_test": len(decision_window_rows),
        "n_trials": len(te),
        "trf_best_alpha": float(decoder.best_alpha_) if decoder.best_alpha_ is not None else None,
        "y_true": y_true,
        "y_pred": y_pred,
        "p_class1": p1,
        "base_window_rows": base_window_rows if float(decision_window_s) > float(input_window_s) else None,
        "window_rows": decision_window_rows,
        "trial_rows": trial_rows,
    }


def _label_to_ab(label: int) -> str:
    return "A" if int(label) == 0 else "B"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_prediction_artifacts(
    *,
    output_dir: Path,
    cv_name: str,
    model_name: str,
    fold_id: str,
    window_s: float,
    y_true: list[int],
    y_pred: list[int],
    p_class1: list[float],
    base_window_rows: Optional[list[dict]] = None,
    window_rows: Optional[list[dict]] = None,
    trial_rows: Optional[list[dict]] = None,
    console_preview: int = 5,
) -> dict:
    fold_slug = fold_id.replace("/", "_")
    window_tag = str(window_s).replace(".", "p")
    base_dir = output_dir / "predictions" / f"{cv_name}_{model_name}_w{window_tag}s"
    base_dir.mkdir(parents=True, exist_ok=True)

    confusion = binary_confusion_matrix(np.asarray(y_true), np.asarray(y_pred))
    confusion["label_mapping"] = {"0": "A(left)", "1": "B(right)"}
    conf_path = base_dir / f"{fold_slug}_confusion.json"
    conf_path.write_text(json.dumps(confusion, indent=2), encoding="utf-8")

    window_path = None
    if window_rows:
        window_path = base_dir / f"{fold_slug}_window_predictions.csv"
        _write_csv(
            window_path,
            [
                "subject_id",
                "trial_id",
                "start",
                "stop",
                "y_true",
                "y_pred",
                "ref_outcome",
                "pred_outcome",
                "p_class1",
            ],
            window_rows,
        )

    base_window_path = None
    if base_window_rows:
        base_window_path = base_dir / f"{fold_slug}_base_window_predictions.csv"
        _write_csv(
            base_window_path,
            [
                "subject_id",
                "trial_id",
                "start",
                "stop",
                "y_true",
                "y_pred",
                "ref_outcome",
                "pred_outcome",
                "p_class1",
            ],
            base_window_rows,
        )

    trial_path = None
    if trial_rows:
        trial_path = base_dir / f"{fold_slug}_trial_predictions.csv"
        _write_csv(
            trial_path,
            [
                "subject_id",
                "trial_id",
                "n_windows",
                "y_true",
                "y_pred",
                "ref_outcome",
                "pred_outcome",
                "mean_p_class1",
            ],
            trial_rows,
        )

    if y_true and y_pred:
        print(" | sample pred/ref:", end="")
        n = min(console_preview, len(y_true))
        for i in range(n):
            print(f" [{_label_to_ab(y_pred[i])}/{_label_to_ab(y_true[i])}]", end="")
        print()

    return {
        "confusion_json": str(conf_path),
        "base_window_csv": str(base_window_path) if base_window_path else None,
        "window_csv": str(window_path) if window_path else None,
        "trial_csv": str(trial_path) if trial_path else None,
    }


def _aggregate_trial_rows_from_windows(window_rows: list[dict]) -> list[dict]:
    by_trial: dict[tuple[str, str], list[dict]] = {}
    for row in window_rows:
        key = (row["subject_id"], row["trial_id"])
        by_trial.setdefault(key, []).append(row)

    trial_rows: list[dict] = []
    for (subject_id, trial_id), rows in by_trial.items():
        y_true = int(rows[0]["y_true"])
        pred_ones = sum(int(r["y_pred"]) for r in rows)
        pred_zeros = len(rows) - pred_ones
        y_pred = 1 if pred_ones >= pred_zeros else 0
        mean_p1 = float(np.mean([float(r["p_class1"]) for r in rows]))
        trial_rows.append(
            {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "n_windows": len(rows),
                "y_true": y_true,
                "y_pred": y_pred,
                "ref_outcome": _label_to_ab(y_true),
                "pred_outcome": _label_to_ab(y_pred),
                "mean_p_class1": mean_p1,
            }
        )

    trial_rows.sort(key=lambda r: (r["subject_id"], r["trial_id"]))
    return trial_rows


def _aggregate_decision_windows_from_base_windows(
    base_window_rows: list[dict],
    *,
    decision_window_samples: int,
) -> list[dict]:
    """Aggregate smaller (base) windows into larger decision windows.

    Input rows must contain integer sample indices `start`/`stop` relative to the trial.
    Decision windows are grouped by `block_start = floor(start / decision_window_samples) * decision_window_samples`.
    """
    by_block: dict[tuple[str, str, int], list[dict]] = {}
    for row in base_window_rows:
        start = int(row["start"])
        block_start = (start // decision_window_samples) * decision_window_samples
        key = (row["subject_id"], row["trial_id"], block_start)
        by_block.setdefault(key, []).append(row)

    decision_rows: list[dict] = []
    for (subject_id, trial_id, block_start), rows in by_block.items():
        rows = sorted(rows, key=lambda r: int(r["start"]))
        y_true = int(rows[0]["y_true"])
        mean_p1 = float(np.mean([float(r["p_class1"]) for r in rows]))
        y_pred = 1 if mean_p1 >= 0.5 else 0
        decision_rows.append(
            {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "start": int(block_start),
                "stop": int(max(int(r["stop"]) for r in rows)),
                "y_true": y_true,
                "y_pred": y_pred,
                "ref_outcome": _label_to_ab(y_true),
                "pred_outcome": _label_to_ab(y_pred),
                "p_class1": mean_p1,
            }
        )

    decision_rows.sort(key=lambda r: (r["subject_id"], r["trial_id"], int(r["start"])))
    return decision_rows


def _assert_subject_and_story_disjoint(
    train_trials: list[Trial],
    val_trials: list[Trial],
    test_trials: list[Trial],
) -> None:
    tr_subj = {t.subject_id for t in train_trials}
    va_subj = {t.subject_id for t in val_trials}
    te_subj = {t.subject_id for t in test_trials}
    if not (tr_subj.isdisjoint(va_subj) and tr_subj.isdisjoint(te_subj) and va_subj.isdisjoint(te_subj)):
        raise AssertionError("Subject overlap across train/val/test in strict protocol")

    tr_group = {_content_group(t) for t in train_trials}
    va_group = {_content_group(t) for t in val_trials}
    te_group = {_content_group(t) for t in test_trials}
    if not (tr_group.isdisjoint(va_group) and tr_group.isdisjoint(te_group) and va_group.isdisjoint(te_group)):
        raise AssertionError("Story/stimulus overlap across train/val/test in strict protocol")


# ======================================================================== #
#  Main experiment loop
# ======================================================================== #

def _save_summary(results: list[dict], cv_name: str, model_name: str,
                   window_s: float, out_file: Path, config: dict, config_fingerprint: str) -> None:
    """Write current results to JSON (called after every fold)."""
    accs = [r["test_accuracy"] for r in results]
    summary = {
        "cv_strategy": cv_name,
        "model": model_name,
        "window_s": window_s,
        "n_folds": len(results),
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "config": config,
        "config_fingerprint": config_fingerprint,
        "per_fold": results,
    }
    out_file.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def run_experiment(
    trials: list[Trial],
    cv_name: str,
    model_name: str,
    window_s: float,
    epochs: int,
    patience: int,
    device: torch.device,
    seed: int = 42,
    lr: float = 1e-3,
    batch_size: int = 64,
    overlap_s: float = 0.0,
    weight_decay: float = 1e-4,
    max_folds: Optional[int] = None,
    output_dir: Path = Path("results"),
    write_artifacts: bool = True,
    train_window_s: Optional[float] = None,
    trf_tune_alpha: bool = False,
    trf_alpha_metric: str = "corr",
    trf_alphas: Optional[str] = None,
    trf_max_train_seconds: float = 1200.0,
    trf_tmin_s: float = 0.0,
    trf_tmax_s: float = 0.5,
) -> list[dict]:
    """Run all folds for one (cv_strategy, model) combination."""
    if train_window_s is None:
        train_window_s = window_s
    if train_window_s <= 0 or window_s <= 0:
        raise ValueError("window_s and train_window_s must be > 0")
    if window_s < train_window_s:
        raise ValueError(
            "For this runner, decision window must be >= train_window_s. "
            "Use a smaller train_window_s if you want to evaluate shorter decision windows."
        )

    cv_fn = CV_STRATEGIES[cv_name]
    folds = list(cv_fn(trials, seed=seed))
    if max_folds is not None:
        folds = folds[:max_folds]

    window_tag = str(window_s).replace(".", "p")
    out_file = output_dir / f"{cv_name}_{model_name}_w{window_tag}s.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    run_config = {
        "cv": cv_name,
        "model": model_name,
        "window_s": window_s,
        "train_window_s": float(train_window_s),
        "overlap_s": overlap_s,
        "epochs": epochs,
        "patience": patience,
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "max_folds": max_folds,
        "trf_tune_alpha": bool(trf_tune_alpha),
        "trf_alpha_metric": str(trf_alpha_metric),
        "trf_alphas": str(trf_alphas) if trf_alphas is not None else None,
        "trf_max_train_seconds": float(trf_max_train_seconds),
        "trf_tmin_s": float(trf_tmin_s),
        "trf_tmax_s": float(trf_tmax_s),
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    # ---- Resume support: load previously completed folds ---- #
    results = []
    done_ids: set[str] = set()
    if out_file.exists():
        try:
            prev = json.loads(out_file.read_text(encoding="utf-8"))
            prev_fp = prev.get("config_fingerprint")
            if prev_fp == config_fingerprint:
                for r in prev.get("per_fold", []):
                    results.append(r)
                    done_ids.add(r["fold_id"])
                if done_ids:
                    print(f"  (resumed: {len(done_ids)} folds already done)")
            else:
                print("  (existing results config mismatch: starting fresh for this run)")
        except Exception:
            pass

    for fi, fold in enumerate(folds):
        if fold.fold_id in done_ids:
            print(f"  [{fi+1}/{len(folds)}] {fold.fold_id}  (cached)")
            continue

        print(f"  [{fi+1}/{len(folds)}] {fold.fold_id}", end="  ", flush=True)

        train_trials = [trials[i] for i in fold.train_idx]
        val_trials = [trials[i] for i in fold.val_idx]
        test_trials = [trials[i] for i in fold.test_idx]

        if cv_name == "strict_si_story":
            _assert_subject_and_story_disjoint(train_trials, val_trials, test_trials)

        sfreq = float(test_trials[0].sfreq) if test_trials else 64.0
        decision_window_samples = int(round(float(window_s) * sfreq))

        if model_name == "trf":
            alpha_grid = None
            if trf_alphas is not None:
                alpha_grid = np.asarray(_parse_csv_floats(trf_alphas), dtype=np.float64)
            res = _train_trf_fold(
                train_trials,
                val_trials,
                test_trials,
                input_window_s=float(train_window_s),
                decision_window_s=float(window_s),
                overlap_s=float(overlap_s),
                max_train_seconds=float(trf_max_train_seconds),
                tune_alpha=bool(trf_tune_alpha),
                alpha_grid=alpha_grid,
                tmin_s=float(trf_tmin_s),
                tmax_s=float(trf_tmax_s),
                alpha_metric=str(trf_alpha_metric),
            )
            artifact_paths = {}
            if write_artifacts:
                artifact_paths = _write_prediction_artifacts(
                    output_dir=output_dir,
                    cv_name=cv_name,
                    model_name=model_name,
                    fold_id=fold.fold_id,
                    window_s=window_s,
                    y_true=res.get("y_true", []),
                    y_pred=res.get("y_pred", []),
                    p_class1=res.get("p_class1", []),
                    base_window_rows=res.get("base_window_rows"),
                    window_rows=res.get("window_rows"),
                    trial_rows=res.get("trial_rows", []),
                )
        else:
            ds_cls = WindowedEEGAudioDataset if model_name == "aadnet_ext" else WindowedEEGDataset
            ds_tr = ds_cls(train_trials, window_s=float(train_window_s), overlap_s=overlap_s)
            ds_val = ds_cls(val_trials, window_s=float(train_window_s), overlap_s=overlap_s)
            ds_te = ds_cls(test_trials, window_s=float(train_window_s), overlap_s=overlap_s)

            if len(ds_tr) == 0 or len(ds_val) == 0 or len(ds_te) == 0:
                print(f"skip (empty dataset: tr={len(ds_tr)} val={len(ds_val)} te={len(ds_te)})")
                continue

            res = _train_deep_fold(
                model_name, ds_tr, ds_val, ds_te,
                seed=seed,
                epochs=epochs,
                patience=patience,
                device=device,
                lr=lr,
                batch_size=batch_size,
                weight_decay=weight_decay,
            )

            y_true = [int(v) for v in res.pop("y_true", [])]
            y_pred = [int(v) for v in res.pop("y_pred", [])]
            p_class1 = [float(v) for v in res.pop("p_class1", [])]
            base_window_rows: list[dict] = []
            for i, win in enumerate(ds_te.window_indices):
                if i >= len(y_true):
                    break
                row = {
                    "subject_id": win.subject_id,
                    "trial_id": win.trial_id,
                    "start": int(win.start),
                    "stop": int(win.stop),
                    "y_true": int(y_true[i]),
                    "y_pred": int(y_pred[i]),
                    "ref_outcome": _label_to_ab(int(y_true[i])),
                    "pred_outcome": _label_to_ab(int(y_pred[i])),
                    "p_class1": float(p_class1[i]),
                }
                base_window_rows.append(row)

            # If decision window > train window, aggregate base-window predictions into larger decisions.
            if float(window_s) > float(train_window_s):
                window_rows = _aggregate_decision_windows_from_base_windows(
                    base_window_rows,
                    decision_window_samples=decision_window_samples,
                )
                y_true = [int(r["y_true"]) for r in window_rows]
                y_pred = [int(r["y_pred"]) for r in window_rows]
                p_class1 = [float(r["p_class1"]) for r in window_rows]
                res["base_window_accuracy"] = float(res.get("test_accuracy", 0.0))
                res["test_accuracy"] = float(np.mean([int(a == b) for a, b in zip(y_true, y_pred)])) if y_true else 0.0
            else:
                window_rows = base_window_rows

            trial_rows = _aggregate_trial_rows_from_windows(window_rows)
            if trial_rows:
                trial_y_true = np.asarray([r["y_true"] for r in trial_rows], dtype=int)
                trial_y_pred = np.asarray([r["y_pred"] for r in trial_rows], dtype=int)
                res["trial_accuracy"] = float((trial_y_true == trial_y_pred).mean())
            artifact_paths = {}
            if write_artifacts:
                artifact_paths = _write_prediction_artifacts(
                    output_dir=output_dir,
                    cv_name=cv_name,
                    model_name=model_name,
                    fold_id=fold.fold_id,
                    window_s=window_s,
                    y_true=y_true,
                    y_pred=y_pred,
                    p_class1=p_class1,
                    base_window_rows=base_window_rows if float(window_s) > float(train_window_s) else None,
                    window_rows=window_rows,
                    trial_rows=trial_rows,
                )

        res["fold_id"] = fold.fold_id
        res["meta"] = fold.meta
        res["artifacts"] = artifact_paths
        for key in ["y_true", "y_pred", "p_class1", "trial_rows", "window_rows", "base_window_rows"]:
            if key in res:
                del res[key]
        results.append(res)
        print(f"acc={res['test_accuracy']:.4f}")

        # ---- Incremental save after each fold ---- #
        _save_summary(results, cv_name, model_name, window_s, out_file, run_config, config_fingerprint)

    # Final save
    _save_summary(results, cv_name, model_name, window_s, out_file, run_config, config_fingerprint)

    return results


# ======================================================================== #
#  CLI
# ======================================================================== #

def main() -> None:
    ap = argparse.ArgumentParser(description="Run KULeuven AAD experiments with 4 CV strategies.")
    ap.add_argument("--data-dir", type=str, default="data/KULeuven")
    ap.add_argument(
        "--protocol",
        type=str,
        default="default",
        choices=["default", "vandecappelle2021_linear"],
        help=(
            "Protocol preset. 'vandecappelle2021_linear' matches the eLife 2021 "
            "linear stimulus reconstruction baseline (20 Hz, 1-9 Hz, powerlaw-subband envelope, "
            "exclude experiment-3 repeats, leave-one-story+speaker-out CV)."
        ),
    )
    ap.add_argument(
        "--cv",
        type=str,
        default="all",
        choices=["all"] + sorted(list(CV_STRATEGIES.keys())),
    )
    ap.add_argument("--model", type=str, default="all", choices=["all", "trf", "cnn", "stgcn", "aadnet_ext"])
    ap.add_argument("--window", type=float, default=2.0, help="Decision window (seconds).")
    ap.add_argument(
        "--train-window",
        type=float,
        default=None,
        help="Model input / training window (seconds). If < --window, predictions are aggregated into the larger decision window.",
    )
    ap.add_argument("--overlap", type=float, default=0.0, help="Window overlap (seconds).")
    ap.add_argument(
        "--sfreq-out",
        type=int,
        default=None,
        help="Override EEG/envelope sampling rate after preprocessing (Hz).",
    )
    ap.add_argument(
        "--bandpass",
        type=str,
        default=None,
        help="Override preprocessing bandpass as 'low,high' (Hz), e.g. '1,9'.",
    )
    ap.add_argument(
        "--envelope",
        type=str,
        default=None,
        choices=["hilbert", "powerlaw_subbands"],
        help="Override envelope extraction method (KUL only).",
    )
    ap.add_argument(
        "--include-experiments",
        type=str,
        default=None,
        help="Comma-separated experiment IDs to include (e.g. '1,2'). If omitted, includes all.",
    )
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str, default="results")
    ap.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Disable writing per-fold prediction artifacts (CSV/JSON) to save disk space.",
    )
    ap.add_argument("--max-folds", type=int, default=None,
                    help="Cap on number of folds (for quick testing).")
    ap.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs to include (e.g., S1,S2). If omitted, uses all subjects.",
    )

    # TRF-specific knobs (to match common linear AAD baselines)
    ap.add_argument("--trf-tune-alpha", action="store_true", help="Tune TRF ridge alpha on the fold validation split.")
    ap.add_argument(
        "--trf-alpha-metric",
        type=str,
        default="corr",
        choices=["corr", "aad_acc"],
        help="Metric for choosing alpha when --trf-tune-alpha is set: 'corr' (envelope corr) or 'aad_acc' (validation decoding accuracy).",
    )
    ap.add_argument(
        "--trf-alphas",
        type=str,
        default=None,
        help="Optional comma-separated alpha grid (e.g., '1e-2,1e-1,1,10,100'). If omitted, uses logspace grid.",
    )
    ap.add_argument(
        "--trf-max-train-seconds",
        type=float,
        default=1200.0,
        help="Cap training data used per fold (seconds) for TRF training segment sampling.",
    )
    ap.add_argument("--trf-tmin", type=float, default=0.0, help="TRF minimum lag (seconds).")
    ap.add_argument("--trf-tmax", type=float, default=0.5, help="TRF maximum lag (seconds).")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Apply protocol preset overrides.
    if args.protocol == "vandecappelle2021_linear":
        if args.cv == "all":
            args.cv = "within_subject_story_speaker_out"
        if args.model == "all":
            args.model = "trf"

        if args.sfreq_out is None:
            args.sfreq_out = 20
        if args.bandpass is None:
            args.bandpass = "1,9"
        if args.envelope is None:
            args.envelope = "powerlaw_subbands"
        if args.include_experiments is None:
            args.include_experiments = "1,2"

        # TRF lag window used in the eLife linear baseline.
        if float(args.trf_tmax) == 0.5:
            args.trf_tmax = 0.25

    cv_names = list(CV_STRATEGIES.keys()) if args.cv == "all" else [args.cv]
    model_names = ["trf", "cnn", "stgcn", "aadnet_ext"] if args.model == "all" else [args.model]

    # -- Load data once --
    print("====================================================")
    print("  KULeuven AAD Experiments")
    print("====================================================")
    print(f"  Loading data from {args.data_dir} ...")

    # Load with audio for TRF, and separately without if needed
    need_audio = any(m in {"trf", "aadnet_ext"} for m in model_names)

    # Preprocess overrides
    preprocess = PreprocessConfig()
    if args.sfreq_out is not None:
        preprocess = PreprocessConfig(
            sfreq_out=int(args.sfreq_out),
            bandpass_hz=preprocess.bandpass_hz,
            reref=preprocess.reref,
        )
    if args.bandpass is not None:
        vals = _parse_csv_floats(args.bandpass)
        if len(vals) != 2:
            raise ValueError("--bandpass must be 'low,high'")
        preprocess = PreprocessConfig(
            sfreq_out=preprocess.sfreq_out,
            bandpass_hz=(float(vals[0]), float(vals[1])),
            reref=preprocess.reref,
        )

    envelope_method = args.envelope
    if envelope_method is None:
        envelope_method = "hilbert"

    include_experiments = None
    if args.include_experiments:
        include_experiments = [int(x.strip()) for x in args.include_experiments.split(",") if x.strip()]

    ds = KULeuvenDataset(
        root=args.data_dir,
        preprocess=preprocess,
        load_audio=need_audio,
        envelope_method=envelope_method,
        include_experiments=include_experiments,
    )
    t0 = time.time()
    trials = list(ds.trials())
    load_time = time.time() - t0

    if args.subjects:
        keep = {s.strip() for s in args.subjects.split(",") if s.strip()}
        trials = [t for t in trials if str(t.subject_id) in keep]
    print(f"  {len(trials)} trials loaded in {load_time:.1f}s")
    print(f"  Subjects: {sorted(set(t.subject_id for t in trials))}")
    print(f"  Labels: {sum(t.label for t in trials)} right / {len(trials) - sum(t.label for t in trials)} left")
    print()

    output_dir = Path(args.output)
    all_summaries = []

    for cv_name in cv_names:
        for model_name in model_names:
            header = f"{cv_name} x {model_name.upper()}"
            print(f"--- {header} ---")
            t0 = time.time()

            results = run_experiment(
                trials=trials,
                cv_name=cv_name,
                model_name=model_name,
                window_s=args.window,
                train_window_s=args.train_window,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                seed=args.seed,
                lr=args.lr,
                batch_size=args.batch_size,
                overlap_s=args.overlap,
                weight_decay=args.weight_decay,
                max_folds=args.max_folds,
                output_dir=output_dir,
                write_artifacts=not bool(args.no_artifacts),
                trf_tune_alpha=bool(args.trf_tune_alpha),
                trf_alpha_metric=str(args.trf_alpha_metric),
                trf_alphas=args.trf_alphas,
                trf_max_train_seconds=float(args.trf_max_train_seconds),
                trf_tmin_s=float(args.trf_tmin),
                trf_tmax_s=float(args.trf_tmax),
            )

            accs = [r["test_accuracy"] for r in results]
            mean_acc = float(np.mean(accs)) if accs else 0.0
            std_acc = float(np.std(accs)) if accs else 0.0
            elapsed = time.time() - t0
            print(f"  => {header}: {mean_acc:.4f} +/- {std_acc:.4f}  ({len(results)} folds, {elapsed:.0f}s)\n")

            all_summaries.append({
                "cv": cv_name,
                "model": model_name,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "n_folds": len(results),
                "time_s": elapsed,
            })

    # -- Summary table --
    print("\n====================================================")
    print("  RESULTS SUMMARY")
    print("====================================================")
    print(f"  {'CV Strategy':<20s} {'Model':<8s} {'Acc (mean +/- std)':<25s} {'Folds':<6s} {'Time'}")
    print(f"  {'-'*20} {'-'*8} {'-'*25} {'-'*6} {'-'*8}")
    for s in all_summaries:
        acc_str = f"{s['mean_acc']:.4f} +/- {s['std_acc']:.4f}"
        time_str = f"{s['time_s']:.0f}s"
        print(f"  {s['cv']:<20s} {s['model'].upper():<8s} {acc_str:<25s} {s['n_folds']:<6d} {time_str}")
    print()

    # Save master summary
    save_json(output_dir / "summary.json", all_summaries)
    print(f"  Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
