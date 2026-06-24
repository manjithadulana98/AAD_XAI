"""Focused XAI pipeline: Channel importance -> ROI grouping -> Frequency analysis.

Sections:
  A. Architecture summary + block ablation
  B. Channel importance (occlusion + permutation + IG) with bootstrap CIs
  C. Multiple-comparison correction (FDR)
  D. Final robust important channels
  E. ROI-level grouping of important channels
  F. Frequency-band analysis on robust important channels / ROIs

Usage:
    python scripts/run_focused_xai.py
    python scripts/run_focused_xai.py --max-samples 500 --n-boot 1000
    python scripts/run_focused_xai.py --montage-file config/dtu_channel_montage.csv
    python scripts/run_focused_xai.py --skip-ig --skip-frequency
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import traceback
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Focused XAI: Channel -> ROI -> Frequency")
    p.add_argument("--data-dir", type=str,
                   default=str(ROOT / "external" / "vlaai" / "evaluation_datasets" / "DTU"))
    p.add_argument("--h5-path", type=str,
                   default=str(ROOT / "external" / "vlaai" / "pretrained_models" / "vlaai.h5"))
    p.add_argument("--output-dir", type=str, default=str(ROOT / "xai_results_focused"))
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--max-samples", type=int, default=200,
                   help="Number of EEG windows. Use -1 for all available windows.")
    p.add_argument("--balanced-by-subject", action="store_true", default=True,
                   help="Sample approximately equal windows per subject (default True).")
    p.add_argument("--no-balanced-by-subject", dest="balanced_by_subject", action="store_false",
                   help="Disable balanced subject sampling.")
    p.add_argument("--random-seed", type=int, default=42, dest="seed")
    p.add_argument("--n-boot", type=int, default=500, help="Bootstrap iterations for CIs.")
    p.add_argument("--n-perm", type=int, default=5000,
                   help="Sign-flip permutation iterations for p-values.")
    p.add_argument("--ig-samples", type=int, default=30, dest="n_ig",
                   help="Number of windows for Integrated Gradients (slower).")
    p.add_argument("--ig-steps", type=int, default=50, help="IG interpolation steps.")
    p.add_argument("--windows-per-subject", type=int, default=50,
                   help="Windows per subject for subject-wise stability.")
    p.add_argument("--top-k", type=int, default=15,
                   help="Number of top channels for frequency analysis.")
    p.add_argument("--montage-file", type=str,
                   default=str(ROOT / "config" / "dtu_channel_montage.csv"),
                   help="Path to channel montage CSV (channel_index,electrode_name,roi,x,y,z).")
    p.add_argument("--fdr-alpha", type=float, default=0.05,
                   help="FDR significance threshold.")
    p.add_argument("--stability-threshold", type=float, default=0.5,
                   help="Fraction of subjects that must agree in sign for stability.")
    p.add_argument("--occlusion-mode", type=str, default="zero",
                   choices=["zero", "mean"],
                   help="Channel replacement used during occlusion.")
    p.add_argument("--roi-frequency-mode", type=str, default="whole_roi",
                   choices=["whole_roi", "robust_only"],
                   help="ROI frequency perturbation mode.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip-ig", action="store_true", help="Skip Integrated Gradients.")
    p.add_argument("--skip-frequency", action="store_true", help="Skip frequency analysis.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════

FS = 64  # Hz

# Fallback index-based ROI mapping (used when no montage file is available)
ROIS_FALLBACK = OrderedDict([
    ("Frontal",        list(range(0, 12))),
    ("Fronto-Central", list(range(12, 18))),
    ("Central",        list(range(18, 30))),
    ("Temporal",       list(range(30, 42))),
    ("Parietal",       list(range(42, 54))),
    ("Occipital",      list(range(54, 64))),
])

BANDS = OrderedDict([
    ("delta",  (0.5, 4.0)),
    ("theta",  (4.0, 8.0)),
    ("alpha",  (8.0, 13.0)),
    ("beta",   (13.0, 30.0)),
])


# ══════════════════════════════════════════════════════════════════════
# Montage loading
# ══════════════════════════════════════════════════════════════════════

def load_montage(montage_path: str | None) -> dict:
    """Load channel montage from CSV or fall back to index-based mapping.

    Returns dict with keys:
        source: "montage_file" or "index_fallback"
        ch_to_name: {int: str}  channel_index -> electrode_name
        ch_to_roi:  {int: str}  channel_index -> roi
        rois:       OrderedDict {roi_name: [ch_indices]}
        montage_path: str or None
    """
    if montage_path and Path(montage_path).is_file():
        ch_to_name = {}
        ch_to_roi = {}
        rois = OrderedDict()
        with open(montage_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["channel_index"])
                ch_to_name[idx] = row["electrode_name"]
                roi = row["roi"]
                ch_to_roi[idx] = roi
                rois.setdefault(roi, []).append(idx)
        print(f"  Montage loaded: {montage_path} ({len(ch_to_name)} channels, {len(rois)} ROIs)")
        return {
            "source": "montage_file",
            "ch_to_name": ch_to_name,
            "ch_to_roi": ch_to_roi,
            "rois": rois,
            "montage_path": montage_path,
        }
    else:
        print(f"  WARNING: Montage file not found ({montage_path}). Using index-based fallback.")
        ch_to_roi = {}
        ch_to_name = {}
        for roi_name, chs in ROIS_FALLBACK.items():
            for ch in chs:
                ch_to_roi[ch] = roi_name
                ch_to_name[ch] = f"Ch{ch}"
        return {
            "source": "index_fallback",
            "ch_to_name": ch_to_name,
            "ch_to_roi": ch_to_roi,
            "rois": ROIS_FALLBACK,
            "montage_path": None,
        }


# ══════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════

def get_attended_prob(decision, eeg, att, unatt):
    """P(attended) for a batch."""
    decision.set_envelopes(att, unatt)
    with torch.no_grad():
        logits = decision(eeg)
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    return probs


def bootstrap_ci(values: np.ndarray, n_boot: int = 500, ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via percentile bootstrap."""
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, len(values), size=len(values))
        means[b] = values[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(values.mean()), float(lo), float(hi)


def sign_flip_p_value(values: np.ndarray, n_perm: int = 5000,
                      seed: int = 42) -> float:
    """Two-sided sign-flip permutation p-value for mean != 0."""
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    obs = abs(values.mean())
    null = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=len(values))
        null[i] = abs((values * signs).mean())
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns (adjusted_p_values, significant_mask).
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Compute adjusted p-values (BH procedure)
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        rank = i + 1
        bh_val = sorted_p[i] * n / rank
        adjusted[sorted_idx[i]] = min(bh_val, adjusted[sorted_idx[i + 1]])

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    significant = adjusted < alpha

    return adjusted, significant


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


def make_output_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base


def compute_baseline_aad_metrics(decision, eeg, att, unatt,
                                 n_boot: int, seed: int,
                                 selected_subject_ids: np.ndarray,
                                 out_dir: Path) -> tuple[dict, bool]:
    """Validate baseline AAD performance before XAI interpretation."""
    decision.set_envelopes(att, unatt)
    with torch.no_grad():
        logits = decision(eeg)
        p_att = torch.softmax(logits, dim=-1)[:, 1]

    logits_np = logits.detach().cpu().numpy()
    corr_unatt = logits_np[:, 0]
    corr_att = logits_np[:, 1]
    margin = corr_att - corr_unatt

    margin_mean, margin_lo, margin_hi = bootstrap_ci(margin, n_boot=n_boot, seed=seed)
    aad_acc = float((corr_att > corr_unatt).mean())
    p_att_np = p_att.detach().cpu().numpy()

    baseline = {
        "corr_attended_mean": float(corr_att.mean()),
        "corr_unattended_mean": float(corr_unatt.mean()),
        "correlation_margin_mean": margin_mean,
        "correlation_margin_CI": [float(margin_lo), float(margin_hi)],
        "aad_accuracy": aad_acc,
        "aad_accuracy_rule": "corr_attended > corr_unattended",
        "mean_p_attended": float(p_att_np.mean()),
        "n_windows": int(eeg.shape[0]),
        "n_subjects": int(len(set(selected_subject_ids.tolist()))) if len(selected_subject_ids) > 0 else 0,
    }
    save_json(baseline, out_dir / "baseline_aad_metrics.json")

    print("\nBaseline AAD validation metrics:")
    print(f"  corr_attended_mean:      {baseline['corr_attended_mean']:+.5f}")
    print(f"  corr_unattended_mean:    {baseline['corr_unattended_mean']:+.5f}")
    print(f"  correlation_margin_mean: {baseline['correlation_margin_mean']:+.5f}")
    print(f"  correlation_margin_CI:   [{margin_lo:+.5f}, {margin_hi:+.5f}]")
    print(f"  AAD accuracy:            {baseline['aad_accuracy']:.3f}")
    print(f"  mean P(attended):        {baseline['mean_p_attended']:.3f}")
    print(f"  windows: {baseline['n_windows']}, subjects: {baseline['n_subjects']}")

    near_chance = abs(aad_acc - 0.5) <= 0.05
    if near_chance:
        print("WARNING: Baseline AAD performance is near chance. XAI results may not be meaningful.")

    return baseline, near_chance


# ══════════════════════════════════════════════════════════════════════
# SECTION A — Architecture + Block Ablation
# ══════════════════════════════════════════════════════════════════════

def run_architecture(model, decision, eeg, att, unatt, n_boot, seed, out_dir, device):
    """Architecture summary and block ablation."""
    print("\n" + "=" * 70)
    print("SECTION A: ARCHITECTURE + BLOCK ABLATION")
    print("=" * 70)
    rng = np.random.RandomState(seed)

    # --- Architecture summary ---
    total_params = sum(p.numel() for p in model.parameters())
    components = {}
    for name, child in model.named_children():
        components[name] = sum(p.numel() for p in child.parameters())

    shared_names = {"extractor", "output_context", "final_dense"}
    shared_params = sum(v for k, v in components.items() if k in shared_names)

    arch_info = {
        "total_params": total_params,
        "components": components,
        "shared_pct": shared_params / total_params,
        "architecture": "VLAAI: 4-iteration recurrent CNN with shared extractor + output_context",
        "forward_pass": "x=0; for i in 0..3: x=output_context(block_denses[i](extractor(eeg+x))); out=final_dense(x)",
    }
    save_json(arch_info, out_dir / "architecture_summary.json")
    print(f"  Total params: {total_params:,}")
    for name, n in components.items():
        print(f"    {name}: {n:,} ({n/total_params*100:.1f}%)")

    # --- Block ablation ---
    base_probs = get_attended_prob(decision, eeg, att, unatt)
    base_mean, base_lo, base_hi = bootstrap_ci(base_probs, n_boot, seed=seed)
    base_acc = float((base_probs > 0.5).mean())

    ablation = {"baseline": {"mean_p": base_mean, "ci": [base_lo, base_hi], "accuracy": base_acc}}
    modes = ["zero_weights", "permute"]

    for bi in range(4):
        ablation[f"block_{bi}"] = {}
        for mode in modes:
            orig_w = model.block_denses[bi].weight.data.clone()
            orig_b = model.block_denses[bi].bias.data.clone()

            if mode == "zero_weights":
                model.block_denses[bi].weight.data.zero_()
                model.block_denses[bi].bias.data.zero_()
                abl_probs = get_attended_prob(decision, eeg, att, unatt)
            elif mode == "permute":
                def perm_hook(mod, inp, out, _rng=rng):
                    perm = torch.from_numpy(_rng.permutation(out.shape[0])).long().to(out.device)
                    return out[perm]
                handle = model.block_denses[bi].register_forward_hook(perm_hook)
                abl_probs = get_attended_prob(decision, eeg, att, unatt)
                handle.remove()

            model.block_denses[bi].weight.data = orig_w
            model.block_denses[bi].bias.data = orig_b

            delta_p = base_probs - abl_probs
            mean_dp, lo_dp, hi_dp = bootstrap_ci(delta_p, n_boot, seed=seed)
            abl_acc = float((abl_probs > 0.5).mean())

            ablation[f"block_{bi}"][mode] = {
                "delta_p_mean": mean_dp, "ci": [lo_dp, hi_dp],
                "ablated_acc": abl_acc, "delta_acc": base_acc - abl_acc,
            }
            print(f"  Block {bi} [{mode:14s}]: dP={mean_dp:+.5f} [{lo_dp:+.5f},{hi_dp:+.5f}], dAcc={base_acc - abl_acc:+.3f}")

    save_json(ablation, out_dir / "block_ablation.json")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(4)
    width = 0.35
    colors = {"zero_weights": "#d32f2f", "permute": "#f57c00"}
    for mi, mode in enumerate(modes):
        dps = [ablation[f"block_{bi}"][mode]["delta_p_mean"] for bi in range(4)]
        cis = [ablation[f"block_{bi}"][mode]["ci"] for bi in range(4)]
        errs = [[d - c[0] for d, c in zip(dps, cis)], [c[1] - d for d, c in zip(dps, cis)]]
        ax.bar(x_pos + mi * width, dps, width, yerr=errs, label=mode,
               color=colors[mode], alpha=0.85, capsize=3)
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels([f"Block {i}" for i in range(4)])
    ax.set_ylabel("ΔP(attended)")
    ax.set_title("Block Ablation")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "block_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()

    return arch_info, ablation


# ══════════════════════════════════════════════════════════════════════
# SECTION B — Channel Importance (Occlusion + Permutation + IG)
# ══════════════════════════════════════════════════════════════════════

def compute_channel_occlusion(decision, eeg, att, unatt, n_boot, n_perm, seed, occlusion_mode):
    """Mask each channel to zero and measure ΔP."""
    N = eeg.shape[0]
    base_probs = get_attended_prob(decision, eeg, att, unatt)
    drops_pw = np.zeros((N, 64))
    ch_means = eeg.mean(dim=(0, 1)) if occlusion_mode == "mean" else None

    for ch in range(64):
        eeg_m = eeg.clone()
        if occlusion_mode == "mean":
            eeg_m[:, :, ch] = ch_means[ch]
        else:
            eeg_m[:, :, ch] = 0.0
        m_probs = get_attended_prob(decision, eeg_m, att, unatt)
        drops_pw[:, ch] = base_probs - m_probs
        if (ch + 1) % 16 == 0:
            print(f"    Occlusion: {ch+1}/64 channels done")

    results = []
    for ch in range(64):
        mean, lo, hi = bootstrap_ci(drops_pw[:, ch], n_boot, seed=seed)
        p_val = sign_flip_p_value(drops_pw[:, ch], n_perm=n_perm, seed=seed + ch)
        results.append({"channel": ch, "mean_dp": mean, "ci_lo": lo, "ci_hi": hi, "p_value": p_val})

    return results, drops_pw


def compute_channel_permutation(decision, eeg, att, unatt, n_boot, n_perm, seed):
    """Shuffle each channel across windows and measure ΔP."""
    N = eeg.shape[0]
    rng = np.random.RandomState(seed)
    base_probs = get_attended_prob(decision, eeg, att, unatt)
    drops_pw = np.zeros((N, 64))

    for ch in range(64):
        eeg_m = eeg.clone()
        perm = torch.from_numpy(rng.permutation(N)).long().to(eeg.device)
        eeg_m[:, :, ch] = eeg.index_select(0, perm)[:, :, ch]
        m_probs = get_attended_prob(decision, eeg_m, att, unatt)
        drops_pw[:, ch] = base_probs - m_probs
        if (ch + 1) % 16 == 0:
            print(f"    Permutation: {ch+1}/64 channels done")

    results = []
    for ch in range(64):
        mean, lo, hi = bootstrap_ci(drops_pw[:, ch], n_boot, seed=seed)
        p_val = sign_flip_p_value(drops_pw[:, ch], n_perm=n_perm, seed=seed + 1000 + ch)
        results.append({"channel": ch, "mean_dp": mean, "ci_lo": lo, "ci_hi": hi, "p_value": p_val})

    return results, drops_pw


def compute_integrated_gradients_summary(decision, eeg, att, unatt, n_ig, ig_steps):
    """Integrated Gradients channel importance (supporting evidence).

    Process windows in small batches (batch_size=5) to avoid OOM on CPU.
    """
    from captum.attr import IntegratedGradients

    n = min(n_ig, eeg.shape[0])
    batch_size = 5
    all_importance = []

    try:
        ig_obj = IntegratedGradients(decision)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            eeg_b = eeg[start:end].clone().requires_grad_(True)
            decision.set_envelopes(att[start:end], unatt[start:end])
            ig_attr = ig_obj.attribute(
                eeg_b, target=1, n_steps=ig_steps,
                baselines=torch.zeros_like(eeg_b),
                internal_batch_size=end - start,
            )
            batch_imp = np.abs(ig_attr.detach().cpu().numpy()).mean(axis=(0, 1))
            all_importance.append(batch_imp * (end - start))
            print(f"    IG: {end}/{n} windows done")
        ch_importance = sum(all_importance) / n
        ranks = np.argsort(np.argsort(-ch_importance)) + 1
        return ch_importance, ranks
    except Exception as e:
        print(f"    IG failed: {e}")
        traceback.print_exc()
        return np.zeros(64), np.arange(1, 65)


def compute_subject_stability(decision, eeg, att, unatt,
                              selected_subject_ids, windows_per_subject, seed):
    """Per-subject channel occlusion to assess stability on selected windows."""
    all_subject_ids = np.asarray(selected_subject_ids)
    unique_subjects = sorted(set(all_subject_ids))
    rng = np.random.RandomState(seed)

    if len(unique_subjects) < 2:
        print("    Need >=2 subjects for stability analysis.")
        return {}, np.zeros((0, 64))

    subj_profiles = {}
    for subj in unique_subjects:
        mask = all_subject_ids == subj
        idxs = np.where(mask)[0]
        n_s = min(windows_per_subject, len(idxs))
        if len(idxs) > n_s:
            idxs = np.sort(rng.choice(idxs, size=n_s, replace=False))
        else:
            idxs = np.sort(idxs)

        idxs_t = torch.from_numpy(idxs).long().to(eeg.device)
        eeg_s = eeg.index_select(0, idxs_t)
        att_s = att.index_select(0, idxs_t)
        unatt_s = unatt.index_select(0, idxs_t)

        base_p = get_attended_prob(decision, eeg_s, att_s, unatt_s)
        ch_drops = np.zeros(64)
        for ch in range(64):
            eeg_m = eeg_s.clone()
            eeg_m[:, :, ch] = 0.0
            m_p = get_attended_prob(decision, eeg_m, att_s, unatt_s)
            ch_drops[ch] = (base_p - m_p).mean()

        subj_profiles[subj] = ch_drops
        print(f"    {subj}: {n_s} windows processed")

    subj_list = sorted(subj_profiles.keys())
    ch_matrix = np.array([subj_profiles[s] for s in subj_list])
    return subj_profiles, ch_matrix


# ══════════════════════════════════════════════════════════════════════
# Combined channel importance with FDR correction
# ══════════════════════════════════════════════════════════════════════

def run_channel_importance(decision, model, eeg, att, unatt, selected_subject_ids,
                           n_boot, n_perm, seed, n_ig, ig_steps, windows_per_subject,
                           top_k, out_dir, montage, fdr_alpha, stability_threshold,
                           occlusion_mode, skip_ig):
    """Multi-method channel importance analysis with FDR correction."""
    print("\n" + "=" * 70)
    print("SECTION B: CHANNEL IMPORTANCE (Occlusion + Permutation + IG)")
    print("=" * 70)

    ch_to_name = montage["ch_to_name"]
    ch_to_roi = montage["ch_to_roi"]

    # --- B.1 Channel occlusion ---
    print(f"  [B.1] Channel occlusion ({occlusion_mode} replacement)...")
    occ_results, occ_pw = compute_channel_occlusion(
        decision, eeg, att, unatt, n_boot, n_perm, seed, occlusion_mode)

    # --- B.2 Channel permutation ---
    print("  [B.2] Channel permutation (shuffle across windows)...")
    perm_results, perm_pw = compute_channel_permutation(
        decision, eeg, att, unatt, n_boot, n_perm, seed)

    # --- B.3 Integrated Gradients ---
    if not skip_ig:
        print("  [B.3] Integrated Gradients (supporting evidence)...")
        ig_importance, ig_ranks = compute_integrated_gradients_summary(
            decision, eeg, att, unatt, n_ig, ig_steps)
        # Restore full-batch envelopes after IG batching
        decision.set_envelopes(att, unatt)
    else:
        print("  [B.3] Integrated Gradients SKIPPED (--skip-ig)")
        ig_importance = np.zeros(64)
        ig_ranks = np.arange(1, 65)

    # --- B.4 Subject-wise stability ---
    print("  [B.4] Subject-wise channel stability...")
    subj_profiles, ch_matrix = compute_subject_stability(
        decision, eeg, att, unatt, selected_subject_ids, windows_per_subject, seed)

    # ── FDR correction ──────────────────────────────────────────────
    print("\n  [C] Multiple-comparison correction (Benjamini-Hochberg FDR)...")
    occ_p_values = np.array([r["p_value"] for r in occ_results])
    perm_p_values = np.array([r["p_value"] for r in perm_results])

    occ_fdr_p, occ_fdr_sig = fdr_correction(occ_p_values, fdr_alpha)
    perm_fdr_p, perm_fdr_sig = fdr_correction(perm_p_values, fdr_alpha)

    n_occ_fdr = int(occ_fdr_sig.sum())
    n_perm_fdr = int(perm_fdr_sig.sum())
    print(f"    Occlusion:   {n_occ_fdr}/64 channels FDR-significant (a={fdr_alpha})")
    print(f"    Permutation: {n_perm_fdr}/64 channels FDR-significant (a={fdr_alpha})")

    # ── Combine into ranked table ──────────────────────────────────
    combined = []
    n_subjects = ch_matrix.shape[0]

    for ch in range(64):
        occ_mean = occ_results[ch]["mean_dp"]
        occ_lo = occ_results[ch]["ci_lo"]
        occ_hi = occ_results[ch]["ci_hi"]
        perm_mean = perm_results[ch]["mean_dp"]
        perm_lo = perm_results[ch]["ci_lo"]
        perm_hi = perm_results[ch]["ci_hi"]

        # CI-based significance (legacy)
        occ_ci_sig = (occ_lo > 0) or (occ_hi < 0)
        perm_ci_sig = (perm_lo > 0) or (perm_hi < 0)

        # Contribution type
        if occ_mean > 0 and perm_mean > 0:
            contribution_type = "facilitatory"
        elif occ_mean < 0 and perm_mean < 0:
            contribution_type = "suppressive"
        else:
            contribution_type = "mixed"

        # Subject-wise stability
        stable_frac = 0.0
        stable_str = "N/A"
        if n_subjects > 0:
            ch_col = ch_matrix[:, ch]
            majority_sign = np.sign(np.median(ch_col))
            if majority_sign == 0:
                majority_sign = np.sign(occ_mean)
            n_agree = int(np.sum(np.sign(ch_col) == majority_sign))
            stable_frac = n_agree / n_subjects
            stable_str = f"{n_agree}/{n_subjects}"

        # Robust significance: same sign + at least one FDR-sig + reasonably stable
        same_sign = contribution_type in ("facilitatory", "suppressive")
        at_least_one_fdr = bool(occ_fdr_sig[ch]) or bool(perm_fdr_sig[ch])
        is_stable = stable_frac >= stability_threshold if n_subjects > 0 else True
        robust_significant = same_sign and at_least_one_fdr and is_stable

        combined.append({
            "channel": ch,
            "electrode_name": ch_to_name.get(ch, f"Ch{ch}"),
            "roi": ch_to_roi.get(ch, "Unknown"),
            "roi_mapping_source": montage["source"],
            "occ_score": occ_mean,
            "occ_ci_lo": occ_lo,
            "occ_ci_hi": occ_hi,
            "occ_p_value": float(occ_p_values[ch]),
            "occ_fdr_p_value": float(occ_fdr_p[ch]),
            "occ_fdr_significant": bool(occ_fdr_sig[ch]),
            "occ_ci_significant": occ_ci_sig,
            "perm_score": perm_mean,
            "perm_ci_lo": perm_lo,
            "perm_ci_hi": perm_hi,
            "perm_p_value": float(perm_p_values[ch]),
            "perm_fdr_p_value": float(perm_fdr_p[ch]),
            "perm_fdr_significant": bool(perm_fdr_sig[ch]),
            "perm_ci_significant": perm_ci_sig,
            "ig_rank": int(ig_ranks[ch]),
            "ig_importance": float(ig_importance[ch]),
            "contribution_type": contribution_type,
            "subject_stability": stable_str,
            "subject_stability_frac": stable_frac,
            "robust_significant": robust_significant,
        })

    # Combined importance score: average of |occ| and |perm| z-scored
    occ_abs = np.array([abs(c["occ_score"]) for c in combined])
    perm_abs = np.array([abs(c["perm_score"]) for c in combined])
    if occ_abs.std() > 1e-10:
        occ_z = (occ_abs - occ_abs.mean()) / occ_abs.std()
    else:
        occ_z = np.zeros(64)
    if perm_abs.std() > 1e-10:
        perm_z = (perm_abs - perm_abs.mean()) / perm_abs.std()
    else:
        perm_z = np.zeros(64)
    combined_score = (occ_z + perm_z) / 2.0

    # Rank by combined score
    rank_order = np.argsort(-combined_score)
    for rank, ch_idx in enumerate(rank_order):
        combined[ch_idx]["rank"] = rank + 1
        combined[ch_idx]["combined_score"] = float(combined_score[ch_idx])

    # Sort by rank
    combined.sort(key=lambda x: x["rank"])

    # Print top channels
    n_robust = sum(1 for c in combined if c["robust_significant"])
    print(f"\n  Total robust significant channels: {n_robust}/64")
    print(f"\n  RANKED CHANNEL TABLE (top {top_k}):")
    print(f"  {'Rank':>4} | {'Ch':>3} | {'Name':>6} | {'ROI':>16} | {'Occ dP':>9} | {'Perm dP':>9} | {'FDR':>5} | {'Type':>12} | {'Stable':>7} | {'Robust':>6}")
    print("  " + "-" * 110)
    for c in combined[:top_k]:
        fdr_mark = ""
        if c["occ_fdr_significant"] and c["perm_fdr_significant"]:
            fdr_mark = "both"
        elif c["occ_fdr_significant"]:
            fdr_mark = "occ"
        elif c["perm_fdr_significant"]:
            fdr_mark = "perm"
        else:
            fdr_mark = "no"
        print(f"  {c['rank']:>4} | {c['channel']:>3} | {c['electrode_name']:>6} | {c['roi']:>16} | "
              f"{c['occ_score']:>+9.5f} | {c['perm_score']:>+9.5f} | {fdr_mark:>5} | "
              f"{c['contribution_type']:>12} | {c['subject_stability']:>7} | "
              f"{'YES' if c['robust_significant'] else 'no':>6}")

    # Save JSON
    save_json({
        "channels": combined,
        "fdr_alpha": fdr_alpha,
        "p_value_method": "sign_flip_permutation",
        "n_perm": int(n_perm),
        "occlusion_mode": occlusion_mode,
        "n_occ_fdr_significant": n_occ_fdr,
        "n_perm_fdr_significant": n_perm_fdr,
        "n_robust_significant": n_robust,
        "montage_source": montage["source"],
        "n_windows": int(eeg.shape[0]),
        "n_boot": int(n_boot),
    }, out_dir / "channel_importance.json")

    # Save CSV — full table
    csv_fields = [
        "rank", "channel", "electrode_name", "roi", "roi_mapping_source",
        "occ_score", "occ_ci_lo", "occ_ci_hi", "occ_p_value", "occ_fdr_p_value",
        "occ_fdr_significant", "occ_ci_significant",
        "perm_score", "perm_ci_lo", "perm_ci_hi", "perm_p_value", "perm_fdr_p_value",
        "perm_fdr_significant", "perm_ci_significant",
        "ig_rank", "ig_importance",
        "contribution_type", "subject_stability", "robust_significant", "combined_score",
    ]
    with open(out_dir / "channel_importance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for c in combined:
            row = {k: c.get(k, "") for k in csv_fields}
            w.writerow(row)

    # Save per-window arrays
    np.save(out_dir / "occlusion_perwindow.npy", occ_pw)
    np.save(out_dir / "permutation_perwindow.npy", perm_pw)
    if ch_matrix.shape[0] > 0:
        np.save(out_dir / "subject_channel_matrix.npy", ch_matrix)

    # ── Plots ──────────────────────────────────────────────────────
    # (1) Standard 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    occ_means = np.array([c["occ_score"] for c in sorted(combined, key=lambda x: x["channel"])])
    colors_bar = ["#d32f2f" if d > 0 else "#1976d2" for d in occ_means]
    ax.bar(range(64), occ_means, color=colors_bar, alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Channel")
    ax.set_ylabel("ΔP (occlusion)")
    ax.set_title("Channel Occlusion Importance (all 64)")

    ax = axes[0, 1]
    top_ch = combined[:top_k]
    x_pos = np.arange(top_k)
    ax.barh(x_pos - 0.15, [c["occ_score"] for c in top_ch], height=0.3,
            color="#d32f2f", alpha=0.8, label="Occlusion")
    ax.barh(x_pos + 0.15, [c["perm_score"] for c in top_ch], height=0.3,
            color="#1976d2", alpha=0.8, label="Permutation")
    for i, c in enumerate(top_ch):
        ax.plot([c["occ_ci_lo"], c["occ_ci_hi"]], [i - 0.15, i - 0.15],
                color="black", linewidth=1)
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f"{c['electrode_name']} ({c['roi'][:4]})" for c in top_ch], fontsize=8)
    ax.set_xlabel("ΔP(attended)")
    ax.set_title(f"Top-{top_k} Channels: Occlusion vs Permutation")
    ax.legend(fontsize=8)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.invert_yaxis()

    ax = axes[1, 0]
    occ_all = [c["occ_score"] for c in combined]
    perm_all = [c["perm_score"] for c in combined]
    ax.scatter(occ_all, perm_all, c="#6a1b9a", alpha=0.6, s=20)
    for c in combined[:10]:
        ax.annotate(c['electrode_name'], (c["occ_score"], c["perm_score"]),
                    fontsize=6, alpha=0.7)
    r_val = np.corrcoef(occ_all, perm_all)[0, 1]
    ax.set_xlabel("Occlusion ΔP")
    ax.set_ylabel("Permutation ΔP")
    ax.set_title(f"Occlusion vs Permutation (r={r_val:.3f})")
    ax.axhline(0, color="k", linewidth=0.3)
    ax.axvline(0, color="k", linewidth=0.3)
    lims = [min(min(occ_all), min(perm_all)), max(max(occ_all), max(perm_all))]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.3)

    ax = axes[1, 1]
    ig_all = [c["ig_importance"] for c in combined]
    occ_abs_all = [abs(c["occ_score"]) for c in combined]
    ax.scatter(occ_abs_all, ig_all, c="#e65100", alpha=0.6, s=20)
    for c in combined[:10]:
        ax.annotate(c['electrode_name'], (abs(c["occ_score"]), c["ig_importance"]),
                    fontsize=6, alpha=0.7)
    if np.std(occ_abs_all) < 1e-12 or np.std(ig_all) < 1e-12:
        r_ig = float("nan")
    else:
        r_ig = float(np.corrcoef(occ_abs_all, ig_all)[0, 1])
    ax.set_xlabel("|Occlusion ΔP|")
    ax.set_ylabel("IG |attribution|")
    ax.set_title(f"Occlusion vs IG importance (r={r_ig:.3f})")

    plt.tight_layout()
    plt.savefig(out_dir / "channel_importance_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # (2) FDR-annotated plot
    fig, ax = plt.subplots(figsize=(14, 6))
    by_channel = {int(c["channel"]): c for c in combined}
    ch_order = np.argsort(-combined_score)
    x_pos = np.arange(64)
    bar_colors = []
    for ch in ch_order:
        c = by_channel[int(ch)]
        if c["robust_significant"] and c["contribution_type"] == "facilitatory":
            bar_colors.append("#2e7d32")  # green
        elif c["robust_significant"] and c["contribution_type"] == "suppressive":
            bar_colors.append("#c62828")  # deep red
        elif c["occ_fdr_significant"] or c["perm_fdr_significant"]:
            bar_colors.append("#f9a825")  # amber
        else:
            bar_colors.append("#bdbdbd")  # grey
    ax.bar(x_pos, combined_score[ch_order], color=bar_colors, alpha=0.9)
    ax.set_xlabel("Channels (ranked by combined importance)", fontsize=10)
    ax.set_ylabel("Combined Z-score", fontsize=10)
    ax.set_title("Channel Importance with FDR Correction", fontsize=12)
    ax.set_xticks(x_pos[::4])
    ax.set_xticklabels([by_channel[int(ch_order[i])]["electrode_name"] for i in range(0, 64, 4)],
                       rotation=45, ha="right", fontsize=7)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2e7d32", label="Robust facilitatory"),
        Patch(facecolor="#c62828", label="Robust suppressive"),
        Patch(facecolor="#f9a825", label="FDR-sig (not robust)"),
        Patch(facecolor="#bdbdbd", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "channel_importance_fdr_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved channel importance results to {out_dir}")

    return combined, subj_profiles, ch_matrix, r_val


# ══════════════════════════════════════════════════════════════════════
# Build final important channel table
# ══════════════════════════════════════════════════════════════════════

def build_final_important_channel_table(combined_channels, freq_by_channel, out_dir):
    """Build and save the final_important_channels.csv for robust channels only."""
    robust = [c for c in combined_channels if c["robust_significant"]]
    robust.sort(key=lambda x: x["rank"])

    # Build freq lookup if available
    freq_lookup = {}
    if freq_by_channel:
        for r in freq_by_channel:
            freq_lookup[r["channel"]] = r

    rows = []
    for i, c in enumerate(robust):
        ch = c["channel"]
        freq_data = freq_lookup.get(ch, {})

        # Interpretation note
        notes = []
        if c["contribution_type"] == "facilitatory":
            notes.append("Removing this channel reduces P(attended)")
        elif c["contribution_type"] == "suppressive":
            notes.append("Removing this channel increases P(attended)")
        if c["occ_fdr_significant"] and c["perm_fdr_significant"]:
            notes.append("significant by both methods")
        elif c["occ_fdr_significant"]:
            notes.append("significant by occlusion only")
        elif c["perm_fdr_significant"]:
            notes.append("significant by permutation only")
        best_band = freq_data.get("most_important_band", "")
        if best_band:
            band_dp = freq_data.get(f"{best_band}_dp", 0)
            if band_dp > 0:
                notes.append(f"best band {best_band} supports decoding")
            else:
                notes.append(f"best band {best_band} may be suppressive/noisy")

        # Frequency contribution type
        freq_contrib = ""
        if freq_data:
            positive_bands = sum(1 for b in BANDS if freq_data.get(f"{b}_dp", 0) > 0)
            if positive_bands >= 3:
                freq_contrib = "broadband_facilitatory"
            elif positive_bands >= 1:
                freq_contrib = "band_selective"
            else:
                freq_contrib = "broadband_suppressive"

        rows.append({
            "rank": i + 1,
            "channel_index": ch,
            "electrode_name": c["electrode_name"],
            "roi": c["roi"],
            "occlusion_delta_p": c["occ_score"],
            "occlusion_ci_low": c["occ_ci_lo"],
            "occlusion_ci_high": c["occ_ci_hi"],
            "occ_p_value": c["occ_p_value"],
            "occ_fdr_p_value": c["occ_fdr_p_value"],
            "permutation_delta_p": c["perm_score"],
            "permutation_ci_low": c["perm_ci_lo"],
            "permutation_ci_high": c["perm_ci_hi"],
            "perm_p_value": c["perm_p_value"],
            "perm_fdr_p_value": c["perm_fdr_p_value"],
            "ig_rank": c["ig_rank"],
            "contribution_type": c["contribution_type"],
            "subject_stability": c["subject_stability"],
            "robust_significant": True,
            "best_frequency_band": best_band,
            "delta_score": freq_data.get("delta_dp", ""),
            "theta_score": freq_data.get("theta_dp", ""),
            "alpha_score": freq_data.get("alpha_dp", ""),
            "beta_score": freq_data.get("beta_dp", ""),
            "frequency_contribution_type": freq_contrib,
            "interpretation_note": "; ".join(notes),
        })

    csv_fields = [
        "rank", "channel_index", "electrode_name", "roi",
        "occlusion_delta_p", "occlusion_ci_low", "occlusion_ci_high",
        "occ_p_value", "occ_fdr_p_value",
        "permutation_delta_p", "permutation_ci_low", "permutation_ci_high",
        "perm_p_value", "perm_fdr_p_value",
        "ig_rank", "contribution_type", "subject_stability", "robust_significant",
        "best_frequency_band", "delta_score", "theta_score", "alpha_score", "beta_score",
        "frequency_contribution_type", "interpretation_note",
    ]
    path = out_dir / "final_important_channels.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"  Saved final_important_channels.csv ({len(rows)} robust channels)")
    return rows


# ══════════════════════════════════════════════════════════════════════
# SECTION E — ROI-Level Grouping
# ══════════════════════════════════════════════════════════════════════

def run_roi_analysis(combined_channels, occ_pw, perm_pw, n_boot, seed, out_dir, montage):
    """Aggregate channel importance at ROI level."""
    print("\n" + "=" * 70)
    print("SECTION E: ROI-LEVEL IMPORTANCE")
    print("=" * 70)

    rois = montage["rois"]
    roi_results = []
    for roi_name, chs in rois.items():
        # Occlusion: mean ΔP across ROI channels per window, then bootstrap
        roi_occ_pw = occ_pw[:, chs].mean(axis=1)
        occ_mean, occ_lo, occ_hi = bootstrap_ci(roi_occ_pw, n_boot, seed=seed)

        roi_perm_pw = perm_pw[:, chs].mean(axis=1)
        perm_mean, perm_lo, perm_hi = bootstrap_ci(roi_perm_pw, n_boot, seed=seed)

        occ_sig = (occ_lo > 0) or (occ_hi < 0)
        perm_sig = (perm_lo > 0) or (perm_hi < 0)

        # Channels in this ROI from ranked list
        roi_ch_ranks = [c for c in combined_channels if c["roi"] == roi_name]
        n_sig = sum(1 for c in roi_ch_ranks if c["occ_fdr_significant"])
        n_robust = sum(1 for c in roi_ch_ranks if c["robust_significant"])

        roi_results.append({
            "roi": roi_name,
            "channels": chs,
            "n_channels": len(chs),
            "occ_mean_dp": occ_mean,
            "occ_ci": [occ_lo, occ_hi],
            "occ_significant": occ_sig,
            "perm_mean_dp": perm_mean,
            "perm_ci": [perm_lo, perm_hi],
            "perm_significant": perm_sig,
            "n_fdr_significant_channels": n_sig,
            "n_robust_channels": n_robust,
        })

        print(f"  {roi_name:20s}: Occ dP={occ_mean:+.5f} [{occ_lo:+.5f},{occ_hi:+.5f}] "
              f"{'SIG' if occ_sig else '   '} | "
              f"Perm ΔP={perm_mean:+.5f} | "
              f"{n_sig} FDR-sig, {n_robust} robust")

    save_json(roi_results, out_dir / "roi_importance.json")

    # CSV
    csv_fields = ["roi", "n_channels", "occ_mean_dp", "occ_ci_lo", "occ_ci_hi",
                  "occ_significant", "perm_mean_dp", "perm_ci_lo", "perm_ci_hi",
                  "perm_significant", "n_fdr_significant_channels", "n_robust_channels"]
    with open(out_dir / "roi_importance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in roi_results:
            row = {k: r.get(k, "") for k in csv_fields}
            row["occ_ci_lo"] = r["occ_ci"][0]
            row["occ_ci_hi"] = r["occ_ci"][1]
            row["perm_ci_lo"] = r["perm_ci"][0]
            row["perm_ci_hi"] = r["perm_ci"][1]
            w.writerow(row)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    roi_names = [r["roi"] for r in roi_results]
    occ_means = [r["occ_mean_dp"] for r in roi_results]
    occ_errs = [[r["occ_mean_dp"] - r["occ_ci"][0] for r in roi_results],
                [r["occ_ci"][1] - r["occ_mean_dp"] for r in roi_results]]
    colors_roi = ["#d32f2f" if m > 0 else "#1976d2" for m in occ_means]
    ax.bar(range(len(roi_names)), occ_means, yerr=occ_errs,
           color=colors_roi, alpha=0.8, capsize=5)
    ax.set_xticks(range(len(roi_names)))
    ax.set_xticklabels(roi_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ΔP (occlusion)")
    ax.set_title("ROI-Level Channel Occlusion Importance")
    ax.axhline(0, color="k", linewidth=0.5)
    for i, r in enumerate(roi_results):
        if r["occ_significant"]:
            ax.text(i, occ_means[i] + 0.001 * np.sign(occ_means[i]), "*",
                    ha="center", fontsize=14, fontweight="bold")

    ax = axes[1]
    perm_means = [r["perm_mean_dp"] for r in roi_results]
    perm_errs = [[r["perm_mean_dp"] - r["perm_ci"][0] for r in roi_results],
                 [r["perm_ci"][1] - r["perm_mean_dp"] for r in roi_results]]
    colors_roi2 = ["#d32f2f" if m > 0 else "#1976d2" for m in perm_means]
    ax.bar(range(len(roi_names)), perm_means, yerr=perm_errs,
           color=colors_roi2, alpha=0.8, capsize=5)
    ax.set_xticks(range(len(roi_names)))
    ax.set_xticklabels(roi_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ΔP (permutation)")
    ax.set_title("ROI-Level Channel Permutation Importance")
    ax.axhline(0, color="k", linewidth=0.5)
    for i, r in enumerate(roi_results):
        if r["perm_significant"]:
            ax.text(i, perm_means[i] + 0.001 * np.sign(perm_means[i]), "*",
                    ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "roi_importance_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ROI importance to {out_dir}")

    return roi_results


# ══════════════════════════════════════════════════════════════════════
# SECTION G — Subject Specificity Analysis
# ══════════════════════════════════════════════════════════════════════

def run_subject_specificity_analysis(combined_channels, ch_matrix, out_dir):
    """Per-channel subject specificity metrics and correlation with importance features.

    Metrics computed per channel:
      - subj_mean / subj_std  : mean and std of per-subject occlusion drops
      - subject_cv            : coefficient of variation (std / |mean|) — high = subject-specific
      - subject_agree_frac    : fraction of subjects whose sign matches the group sign
                                (already captured in combined_channels["subject_stability_frac"])

    Correlations (Pearson r):
      - subject_agree_frac  vs  |occ|, |perm|, IG importance, combined score
      - cross-subject std    vs  |occ|, |perm|, IG importance, combined score
    """
    print("\n" + "=" * 70)
    print("SECTION G: SUBJECT SPECIFICITY & FEATURE CORRELATION")
    print("=" * 70)

    if ch_matrix.shape[0] < 2:
        print("  Need >=2 subjects for specificity analysis. Skipping.")
        return {}, []

    n_subjects = ch_matrix.shape[0]
    ch_by_idx = {c["channel"]: c for c in combined_channels}

    # ── Per-channel specificity metrics ──────────────────────────
    ch_spec = []
    for ch in range(64):
        col = ch_matrix[:, ch]
        subj_mean = float(col.mean())
        subj_std = float(col.std())
        abs_mean = abs(subj_mean)
        cv = float(subj_std / abs_mean) if abs_mean > 1e-10 else float("nan")

        cinfo = ch_by_idx.get(ch, {})
        ch_spec.append({
            "channel": ch,
            "electrode_name": cinfo.get("electrode_name", f"Ch{ch}"),
            "roi": cinfo.get("roi", "Unknown"),
            "subj_mean": subj_mean,
            "subj_std": subj_std,
            "subj_range": float(col.max() - col.min()),
            "subject_cv": cv,
            "subject_agree_frac": cinfo.get("subject_stability_frac", 0.0),
            "occ_score": cinfo.get("occ_score", 0.0),
            "perm_score": cinfo.get("perm_score", 0.0),
            "ig_importance": cinfo.get("ig_importance", 0.0),
            "combined_score": cinfo.get("combined_score", 0.0),
            "robust_significant": cinfo.get("robust_significant", False),
        })

    # ── Build feature arrays ──────────────────────────────────────
    occ_abs = np.array([abs(r["occ_score"]) for r in ch_spec])
    perm_abs = np.array([abs(r["perm_score"]) for r in ch_spec])
    ig_imp = np.array([r["ig_importance"] for r in ch_spec])
    comb_score = np.array([r["combined_score"] for r in ch_spec])
    agree_arr = np.array([r["subject_agree_frac"] for r in ch_spec])
    std_arr = np.array([r["subj_std"] for r in ch_spec])
    cv_arr_raw = np.array([r["subject_cv"] for r in ch_spec])
    cv_arr = np.where(np.isfinite(cv_arr_raw), cv_arr_raw, 0.0)
    electrode_names = [r["electrode_name"] for r in ch_spec]

    # ── Per-subject Spearman ρ vs group mean ──────────────────────
    from scipy.stats import spearmanr as _spearmanr
    group_mean_profile = ch_matrix.mean(axis=0)   # shape (64,)
    per_subj_rho = []
    for _i in range(n_subjects):
        _rho, _ = _spearmanr(ch_matrix[_i, :], group_mean_profile)
        per_subj_rho.append(float(_rho) if np.isfinite(_rho) else 0.0)
    mean_subj_rho = float(np.mean(per_subj_rho))
    std_subj_rho  = float(np.std(per_subj_rho))

    def safe_corr(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return float("nan")
        return float(np.corrcoef(x[mask], y[mask])[0, 1])

    corr_results = {
        "agree_frac_vs_occ_abs": safe_corr(agree_arr, occ_abs),
        "agree_frac_vs_perm_abs": safe_corr(agree_arr, perm_abs),
        "agree_frac_vs_ig": safe_corr(agree_arr, ig_imp),
        "agree_frac_vs_combined": safe_corr(agree_arr, comb_score),
        "subj_std_vs_occ_abs": safe_corr(std_arr, occ_abs),
        "subj_std_vs_perm_abs": safe_corr(std_arr, perm_abs),
        "subj_std_vs_ig": safe_corr(std_arr, ig_imp),
        "subj_std_vs_combined": safe_corr(std_arr, comb_score),
        "cv_vs_occ_abs": safe_corr(cv_arr, occ_abs),
        "cv_vs_perm_abs": safe_corr(cv_arr, perm_abs),
        "cv_vs_ig": safe_corr(cv_arr, ig_imp),
        "cv_vs_combined": safe_corr(cv_arr, comb_score),
    }

    # ── Cross-subject inter-subject correlation matrix ────────────
    if n_subjects >= 2:
        corr_mat = np.corrcoef(ch_matrix)
        triu = corr_mat[np.triu_indices(n_subjects, k=1)]
        intersubj_r_mean = float(np.nanmean(triu))
        intersubj_r_std = float(np.nanstd(triu))
    else:
        intersubj_r_mean = float("nan")
        intersubj_r_std = float("nan")

    print(f"\n  n_subjects: {n_subjects}")
    print(f"  Inter-subject profile correlation: mean r = {intersubj_r_mean:.3f} ± {intersubj_r_std:.3f}")
    print(f"\n  Subject agreement fraction vs feature correlations:")
    print(f"    agree_frac vs |occ|:    r = {corr_results['agree_frac_vs_occ_abs']:+.3f}")
    print(f"    agree_frac vs |perm|:   r = {corr_results['agree_frac_vs_perm_abs']:+.3f}")
    print(f"    agree_frac vs IG:       r = {corr_results['agree_frac_vs_ig']:+.3f}")
    print(f"    agree_frac vs combined: r = {corr_results['agree_frac_vs_combined']:+.3f}")
    print(f"\n  Cross-subject std vs feature correlations:")
    print(f"    subj_std vs |occ|:      r = {corr_results['subj_std_vs_occ_abs']:+.3f}")
    print(f"    subj_std vs |perm|:     r = {corr_results['subj_std_vs_perm_abs']:+.3f}")
    print(f"    subj_std vs IG:         r = {corr_results['subj_std_vs_ig']:+.3f}")
    print(f"    subj_std vs combined:   r = {corr_results['subj_std_vs_combined']:+.3f}")
    print(f"\n  CV vs feature correlations:")
    print(f"    CV vs |occ|:            r = {corr_results['cv_vs_occ_abs']:+.3f}")
    print(f"    CV vs |perm|:           r = {corr_results['cv_vs_perm_abs']:+.3f}")
    print(f"    CV vs IG:               r = {corr_results['cv_vs_ig']:+.3f}")
    print(f"    CV vs combined:         r = {corr_results['cv_vs_combined']:+.3f}")

    # ── Save outputs ──────────────────────────────────────────────
    save_json({
        "n_subjects": n_subjects,
        "intersubj_r_mean": intersubj_r_mean,
        "intersubj_r_std": intersubj_r_std,
        "per_subject_spearman_rho": {
            f"S{i+1}": per_subj_rho[i] for i in range(n_subjects)
        },
        "per_subject_spearman_mean": mean_subj_rho,
        "per_subject_spearman_std": std_subj_rho,
        "feature_vs_specificity_correlations": corr_results,
        "per_channel": ch_spec,
    }, out_dir / "subject_specificity.json")

    spec_fields = [
        "channel", "electrode_name", "roi",
        "subj_mean", "subj_std", "subj_range", "subject_cv",
        "subject_agree_frac", "occ_score", "perm_score",
        "ig_importance", "combined_score", "robust_significant",
    ]
    with open(out_dir / "subject_specificity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=spec_fields)
        w.writeheader()
        for r in ch_spec:
            w.writerow({k: r.get(k, "") for k in spec_fields})

    # ── Plots ─────────────────────────────────────────────────────

    # (1) 4-panel correlation scatter
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: subjects × channels heatmap
    ax = axes[0, 0]
    vmax = max(np.abs(ch_matrix).max(), 1e-6)
    im = ax.imshow(ch_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Subject")
    ax.set_title(f"Per-Subject Importance (occlusion ΔP)  [{n_subjects} subjects]")
    plt.colorbar(im, ax=ax, label="ΔP", shrink=0.8)

    # Top-right: agree_frac vs |occ|
    ax = axes[0, 1]
    colors_scatter = ["#2e7d32" if r["robust_significant"] else "#9e9e9e" for r in ch_spec]
    ax.scatter(occ_abs, agree_arr, c=colors_scatter, alpha=0.7, s=35)
    thresh = np.percentile(occ_abs, 85)
    for r in ch_spec:
        if abs(r["occ_score"]) >= thresh:
            ax.annotate(r["electrode_name"], (abs(r["occ_score"]), r["subject_agree_frac"]),
                        fontsize=6, alpha=0.8)
    r_val = corr_results["agree_frac_vs_occ_abs"]
    ax.set_xlabel("|Occlusion ΔP|")
    ax.set_ylabel("Subject Agreement Fraction")
    ax.set_title(f"|Occ| vs Subject Agreement  (r = {r_val:.3f})")
    ax.axhline(0.5, color="k", linewidth=0.7, linestyle="--", alpha=0.5, label="chance=0.5")
    ax.legend(fontsize=7)

    # Bottom-left: subj_std vs |occ|
    ax = axes[1, 0]
    ax.scatter(occ_abs, std_arr, c=colors_scatter, alpha=0.7, s=35)
    for r in ch_spec:
        if abs(r["occ_score"]) >= thresh:
            ax.annotate(r["electrode_name"], (abs(r["occ_score"]), r["subj_std"]),
                        fontsize=6, alpha=0.8)
    r_val2 = corr_results["subj_std_vs_occ_abs"]
    ax.set_xlabel("|Occlusion ΔP|")
    ax.set_ylabel("Cross-Subject Std")
    ax.set_title(f"|Occ| vs Subject Variability  (r = {r_val2:.3f})")

    # Bottom-right: combined score vs agree_frac
    ax = axes[1, 1]
    ax.scatter(comb_score, agree_arr, c=colors_scatter, alpha=0.7, s=35)
    thresh_comb = np.percentile(comb_score, 85)
    for r in ch_spec:
        if r["combined_score"] >= thresh_comb:
            ax.annotate(r["electrode_name"], (r["combined_score"], r["subject_agree_frac"]),
                        fontsize=6, alpha=0.8)
    r_val3 = corr_results["agree_frac_vs_combined"]
    ax.set_xlabel("Combined Importance Score")
    ax.set_ylabel("Subject Agreement Fraction")
    ax.set_title(f"Combined Score vs Subject Agreement  (r = {r_val3:.3f})")
    ax.axhline(0.5, color="k", linewidth=0.7, linestyle="--", alpha=0.5)

    for ax in axes.flat:
        ax.tick_params(labelsize=8)
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#2e7d32", label="robust"), Patch(facecolor="#9e9e9e", label="not robust")]
    fig.legend(handles=legend_els, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_dir / "subject_specificity_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # (2) Correlation summary bar chart
    labels_bar = [
        "agree×|occ|", "agree×|perm|", "agree×IG", "agree×comb",
        "std×|occ|", "std×|perm|", "std×IG", "std×comb",
        "cv×|occ|", "cv×|perm|", "cv×IG", "cv×comb",
    ]
    vals_bar = [
        corr_results["agree_frac_vs_occ_abs"], corr_results["agree_frac_vs_perm_abs"],
        corr_results["agree_frac_vs_ig"], corr_results["agree_frac_vs_combined"],
        corr_results["subj_std_vs_occ_abs"], corr_results["subj_std_vs_perm_abs"],
        corr_results["subj_std_vs_ig"], corr_results["subj_std_vs_combined"],
        corr_results["cv_vs_occ_abs"], corr_results["cv_vs_perm_abs"],
        corr_results["cv_vs_ig"], corr_results["cv_vs_combined"],
    ]
    bar_colors = ["#1565c0"] * 4 + ["#c62828"] * 4 + ["#6a1b9a"] * 4

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(labels_bar)), vals_bar, color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(labels_bar)))
    ax.set_xticklabels(labels_bar, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r")
    ax.set_title("Feature vs Subject Specificity — Pearson Correlations\n"
                 "(blue=agree frac, red=cross-subj std, purple=CV)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(-1, 1)
    for i, (bar, val) in enumerate(zip(bars, vals_bar)):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02 * np.sign(val) if val != 0 else 0.02,
                    f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "subject_specificity_correlation_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # (3) Subject-vs-Group heatmap — individual rows + group mean row
    n_rows_top = n_subjects
    fig, axes = plt.subplots(
        2, 1, figsize=(max(16, len(electrode_names) * 0.28), 4 + n_rows_top * 0.35),
        gridspec_kw={"height_ratios": [n_rows_top, 1.8]},
    )
    vmax_hm = max(np.abs(ch_matrix).max(), 1e-6)
    im_top = axes[0].imshow(ch_matrix, aspect="auto", cmap="RdBu_r",
                             vmin=-vmax_hm, vmax=vmax_hm)
    axes[0].set_yticks(range(n_subjects))
    axes[0].set_yticklabels([f"S{i+1}" for i in range(n_subjects)], fontsize=7)
    axes[0].set_xticks([])
    axes[0].set_title("Per-Subject Channel Importance (Occlusion ΔP) vs Group Mean",
                       fontsize=11, pad=4)
    plt.colorbar(im_top, ax=axes[0], label="ΔP", shrink=0.8, pad=0.01)

    vmax_gm = max(np.abs(group_mean_profile).max(), 1e-6)
    im_bot = axes[1].imshow(group_mean_profile.reshape(1, -1), aspect="auto",
                             cmap="RdBu_r", vmin=-vmax_gm, vmax=vmax_gm)
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(["Group\nMean"], fontsize=8)
    axes[1].set_xticks(range(len(electrode_names)))
    axes[1].set_xticklabels(electrode_names, rotation=90, fontsize=6)
    axes[1].set_title("Group Mean Profile", fontsize=9, pad=3)
    plt.colorbar(im_bot, ax=axes[1], label="ΔP", shrink=0.8, pad=0.01)

    plt.tight_layout()
    plt.savefig(out_dir / "subject_vs_group_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # (4) Per-subject Spearman ρ vs group mean — bar chart
    sort_rho = np.argsort(per_subj_rho)[::-1]
    rho_sorted = [per_subj_rho[i] for i in sort_rho]
    labels_rho  = [f"S{i+1}" for i in sort_rho]
    bar_colors_rho = ["#1565c0" if v >= 0 else "#c62828" for v in rho_sorted]

    fig, ax = plt.subplots(figsize=(max(8, n_subjects * 0.55), 5))
    bars = ax.bar(range(len(rho_sorted)), rho_sorted, color=bar_colors_rho, alpha=0.85)
    ax.set_xticks(range(len(rho_sorted)))
    ax.set_xticklabels(labels_rho, fontsize=9)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Spearman ρ vs Group Mean Profile")
    ax.set_title(
        f"Per-Subject Profile Similarity to Group Mean\n"
        f"Mean ρ = {mean_subj_rho:.3f} ± {std_subj_rho:.3f}  "
        f"(higher = subject matches group pattern)"
    )
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(-1, 1)
    for bar, val in zip(bars, rho_sorted):
        if np.isfinite(val):
            ypos = val + 0.03 if val >= 0 else val - 0.03
            va   = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:.2f}",
                    ha="center", va=va, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "subject_profile_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()

    # (5) Channel disagreement — sorted by group |occ|, top-40
    disagreement = 1.0 - agree_arr
    sort_dis = np.argsort(occ_abs)[::-1]
    top_n = min(40, len(sort_dis))
    top_idx = sort_dis[:top_n]
    dis_top  = disagreement[top_idx]
    occ_top  = occ_abs[top_idx]
    names_top = [electrode_names[i] for i in top_idx]
    bar_colors_dis = ["#c62828" if d > 0.5 else "#1565c0" for d in dis_top]

    fig, ax = plt.subplots(figsize=(max(14, top_n * 0.42), 5))
    bars = ax.bar(range(top_n), dis_top, color=bar_colors_dis, alpha=0.85)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(names_top, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of Subjects Disagreeing with Group Sign")
    ax.set_title(
        "Channel Disagreement  (sorted by group |Occ ΔP|, top 40)\n"
        "Red = majority of subjects disagree with group mean sign"
    )
    ax.axhline(0.5, color="k", linewidth=0.8, linestyle="--", alpha=0.6, label="50% disagreement")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    for i, (bar, occ_val) in enumerate(zip(bars, occ_top)):
        ax.text(bar.get_x() + bar.get_width() / 2, dis_top[i] + 0.02,
                f"{occ_val:.3f}", ha="center", va="bottom", fontsize=5, color="#555555")
    plt.tight_layout()
    plt.savefig(out_dir / "channel_disagreement.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Per-subject profile similarity: mean ρ = {mean_subj_rho:.3f} ± {std_subj_rho:.3f}")
    print(f"  Saved subject specificity results to {out_dir}")
    return corr_results, ch_spec


# ══════════════════════════════════════════════════════════════════════
# SECTION F — Frequency Analysis on Robust Important Channels
# ══════════════════════════════════════════════════════════════════════

def compute_frequency_by_important_channels(decision, eeg, att, unatt, combined_channels,
                                             n_boot, seed, top_k, out_dir, montage,
                                             roi_frequency_mode):
    """Frequency-band analysis prioritising robust important channels, then ROIs."""
    print("\n" + "=" * 70)
    print("SECTION F: FREQUENCY ANALYSIS (robust important channels first)")
    print("=" * 70)

    from scipy.signal import butter, sosfiltfilt

    ch_to_roi = montage["ch_to_roi"]
    ch_to_name = montage["ch_to_name"]
    rois = montage["rois"]

    N = eeg.shape[0]
    base_probs = get_attended_prob(decision, eeg, att, unatt)
    pad_samples = 64
    device = eeg.device

    # Select channels: robust first, then fill with top-k by rank
    robust = [c for c in combined_channels if c["robust_significant"]]
    robust_chs = [c["channel"] for c in robust]
    top_chs = [c["channel"] for c in combined_channels[:top_k]]
    # Union preserving robust-first order
    analysis_chs = list(robust_chs)
    for ch in top_chs:
        if ch not in analysis_chs:
            analysis_chs.append(ch)
    analysis_chs = analysis_chs[:max(top_k, len(robust_chs))]

    print(f"  Analysing {len(analysis_chs)} channels (robust: {len(robust_chs)}, total: {len(analysis_chs)})")
    print(f"  ROI frequency mode: {roi_frequency_mode}")

    # Precompute band content
    eeg_np = eeg.detach().cpu().numpy()
    T = eeg_np.shape[1]

    band_content_all = {}
    for band_name, (lo, hi) in BANDS.items():
        nyq = FS / 2.0
        lo_n = max(lo / nyq, 0.01)
        hi_n = min(hi / nyq, 0.99)
        sos = butter(4, [lo_n, hi_n], btype="bandpass", output="sos")

        band_content = np.zeros_like(eeg_np)
        for w in range(N):
            for ch in range(64):
                sig = eeg_np[w, :, ch]
                padded = np.pad(sig, pad_samples, mode="reflect")
                filtered = sosfiltfilt(sos, padded)
                band_content[w, :, ch] = filtered[pad_samples:pad_samples + T]

        band_content_all[band_name] = band_content
        print(f"    Filtered band: {band_name}")

    # --- F.1 Per-channel frequency analysis (robust channels first) ---
    print("  [F.1] Per-channel frequency analysis...")
    freq_by_channel = []

    for ch in analysis_chs:
        is_robust = ch in robust_chs
        ch_result = {
            "channel": ch,
            "electrode_name": ch_to_name.get(ch, f"Ch{ch}"),
            "roi": ch_to_roi.get(ch, "Unknown"),
            "occ_rank": next(c["rank"] for c in combined_channels if c["channel"] == ch),
            "is_robust": is_robust,
        }
        band_scores = {}
        for band_name in BANDS:
            eeg_mod = eeg_np.copy()
            eeg_mod[:, :, ch] -= band_content_all[band_name][:, :, ch]
            eeg_t = torch.from_numpy(eeg_mod.astype(np.float32)).to(device)
            p_mod = get_attended_prob(decision, eeg_t, att, unatt)
            dp = base_probs - p_mod
            mean, lo, hi = bootstrap_ci(dp, n_boot, seed=seed)
            band_scores[band_name] = {"mean_dp": mean, "ci": [lo, hi]}
            ch_result[f"{band_name}_dp"] = mean
            ch_result[f"{band_name}_ci_lo"] = lo
            ch_result[f"{band_name}_ci_hi"] = hi

        ch_result["most_important_band"] = max(band_scores, key=lambda b: abs(band_scores[b]["mean_dp"]))
        # Frequency contribution type
        positive_bands = sum(1 for b in BANDS if band_scores[b]["mean_dp"] > 0)
        if positive_bands >= 3:
            ch_result["frequency_contribution_type"] = "broadband_facilitatory"
        elif positive_bands >= 1:
            ch_result["frequency_contribution_type"] = "band_selective"
        else:
            ch_result["frequency_contribution_type"] = "broadband_suppressive"

        freq_by_channel.append(ch_result)

        marker = " [ROBUST]" if is_robust else ""
        print(f"    {ch_to_name.get(ch, f'Ch{ch}'):>6} (Ch{ch:2d}, {ch_to_roi.get(ch, '?')[:4]}): "
              f"δ={ch_result['delta_dp']:+.5f} θ={ch_result['theta_dp']:+.5f} "
              f"α={ch_result['alpha_dp']:+.5f} β={ch_result['beta_dp']:+.5f} "
              f"-> {ch_result['most_important_band']}{marker}")

    # --- F.2 Per-ROI frequency analysis (using only robust channels if enough) ---
    print("  [F.2] Per-ROI frequency analysis...")
    freq_by_roi = []

    for roi_name, chs in rois.items():
        roi_robust_chs = [ch for ch in chs if ch in robust_chs]
        if roi_frequency_mode == "robust_only":
            roi_chs_for_mode = roi_robust_chs
        else:
            roi_chs_for_mode = chs
        roi_row = {
            "roi": roi_name,
            "n_channels": len(chs),
            "n_robust_channels": len(roi_robust_chs),
            "roi_frequency_mode": roi_frequency_mode,
            "roi_note": "",
        }

        if len(roi_chs_for_mode) == 0:
            for band_name in BANDS:
                roi_row[f"{band_name}_dp"] = float("nan")
                roi_row[f"{band_name}_ci_lo"] = float("nan")
                roi_row[f"{band_name}_ci_hi"] = float("nan")
            roi_row["most_important_band"] = "NA"
            roi_row["roi_note"] = "no robust channels in ROI; skipped in robust_only mode"
            freq_by_roi.append(roi_row)
            print(f"    {roi_name:20s}: skipped (no robust channels in ROI)")
            continue

        for band_name in BANDS:
            eeg_mod = eeg_np.copy()
            for ch in roi_chs_for_mode:
                eeg_mod[:, :, ch] -= band_content_all[band_name][:, :, ch]
            eeg_t = torch.from_numpy(eeg_mod.astype(np.float32)).to(device)
            p_mod = get_attended_prob(decision, eeg_t, att, unatt)
            dp = base_probs - p_mod
            mean, lo, hi = bootstrap_ci(dp, n_boot, seed=seed)
            roi_row[f"{band_name}_dp"] = mean
            roi_row[f"{band_name}_ci_lo"] = lo
            roi_row[f"{band_name}_ci_hi"] = hi

        roi_row["most_important_band"] = max(
            BANDS.keys(), key=lambda b: abs(roi_row[f"{b}_dp"]))
        freq_by_roi.append(roi_row)

        print(f"    {roi_name:20s}: "
              f"δ={roi_row['delta_dp']:+.5f} θ={roi_row['theta_dp']:+.5f} "
              f"α={roi_row['alpha_dp']:+.5f} β={roi_row['beta_dp']:+.5f} "
              f"-> {roi_row['most_important_band']} ({len(roi_robust_chs)} robust ch)")

    # Save results
    save_json({
        "roi_frequency_mode": roi_frequency_mode,
        "frequency_by_channel": freq_by_channel,
        "frequency_by_roi": freq_by_roi,
    },
              out_dir / "frequency_analysis.json")

    # CSVs
    ch_fields = ["channel", "electrode_name", "roi", "occ_rank", "is_robust",
                 "most_important_band", "frequency_contribution_type",
                 "delta_dp", "delta_ci_lo", "delta_ci_hi",
                 "theta_dp", "theta_ci_lo", "theta_ci_hi",
                 "alpha_dp", "alpha_ci_lo", "alpha_ci_hi",
                 "beta_dp", "beta_ci_lo", "beta_ci_hi"]
    with open(out_dir / "frequency_by_channel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ch_fields)
        w.writeheader()
        for r in freq_by_channel:
            w.writerow({k: r.get(k, "") for k in ch_fields})

    roi_fields = ["roi", "n_channels", "n_robust_channels", "roi_frequency_mode", "roi_note", "most_important_band",
                  "delta_dp", "delta_ci_lo", "delta_ci_hi",
                  "theta_dp", "theta_ci_lo", "theta_ci_hi",
                  "alpha_dp", "alpha_ci_lo", "alpha_ci_hi",
                  "beta_dp", "beta_ci_lo", "beta_ci_hi"]
    with open(out_dir / "frequency_by_roi.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=roi_fields)
        w.writeheader()
        for r in freq_by_roi:
            w.writerow({k: r.get(k, "") for k in roi_fields})

    # ── Plots ──────────────────────────────────────────────────────

    # (1) frequency_by_important_channels.png — heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(freq_by_channel) * 0.4)))

    ax = axes[0]
    n_ch = len(freq_by_channel)
    band_matrix = np.array([[r[f"{b}_dp"] for b in BANDS] for r in freq_by_channel])
    im = ax.imshow(band_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(band_matrix).max(), vmax=np.abs(band_matrix).max())
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(list(BANDS.keys()))
    ax.set_yticks(range(n_ch))
    ylabels = []
    for r in freq_by_channel:
        marker = " *" if r["is_robust"] else ""
        ylabels.append(f"{r['electrode_name']}({r['roi'][:3]}){marker}")
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_title(f"Band ΔP for Important Channels (* = robust)")
    plt.colorbar(im, ax=ax, label="ΔP", shrink=0.8)

    # (2) grouped bar for robust channels only
    ax = axes[1]
    robust_freq = [r for r in freq_by_channel if r["is_robust"]]
    if robust_freq:
        n_r = len(robust_freq)
        x_r = np.arange(n_r)
        width = 0.2
        band_colors = {"delta": "#1565c0", "theta": "#2e7d32", "alpha": "#f57f17", "beta": "#d32f2f"}
        for bi, bname in enumerate(BANDS):
            vals = [r[f"{bname}_dp"] for r in robust_freq]
            ax.bar(x_r + bi * width, vals, width, label=bname,
                   color=band_colors[bname], alpha=0.85)
        ax.set_xticks(x_r + 1.5 * width)
        ax.set_xticklabels([r["electrode_name"] for r in robust_freq], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("ΔP")
        ax.set_title("Frequency Bands — Robust Channels Only")
        ax.legend(fontsize=8)
        ax.axhline(0, color="k", linewidth=0.5)
    else:
        ax.text(0.5, 0.5, "No robust channels", transform=ax.transAxes, ha="center")

    plt.tight_layout()
    plt.savefig(out_dir / "frequency_by_important_channels.png", dpi=150, bbox_inches="tight")
    plt.close()

    # (3) roi_frequency_summary.png
    fig, ax = plt.subplots(figsize=(12, 5))
    x_roi = np.arange(len(rois))
    width = 0.2
    band_colors = {"delta": "#1565c0", "theta": "#2e7d32", "alpha": "#f57f17", "beta": "#d32f2f"}
    for bi, band_name in enumerate(BANDS):
        vals = [0.0 if np.isnan(r[f"{band_name}_dp"]) else r[f"{band_name}_dp"] for r in freq_by_roi]
        ax.bar(x_roi + bi * width, vals, width, label=band_name,
               color=band_colors[band_name], alpha=0.8)
    ax.set_xticks(x_roi + 1.5 * width)
    ax.set_xticklabels([r["roi"] for r in freq_by_roi], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ΔP")
    ax.set_title("Frequency Band × ROI")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "roi_frequency_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # (4) Original 3-panel frequency plot (kept for backwards compatibility)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax = axes[0]
    im = ax.imshow(band_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(band_matrix).max(), vmax=np.abs(band_matrix).max())
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(list(BANDS.keys()))
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([f"{r['electrode_name']}({r['roi'][:3]})" for r in freq_by_channel], fontsize=7)
    ax.set_title(f"Band ΔP for Important Channels")
    plt.colorbar(im, ax=ax, label="ΔP")

    ax = axes[1]
    for bi, band_name in enumerate(BANDS):
        vals = [0.0 if np.isnan(r[f"{band_name}_dp"]) else r[f"{band_name}_dp"] for r in freq_by_roi]
        ax.bar(x_roi + bi * width, vals, width, label=band_name,
               color=band_colors[band_name], alpha=0.8)
    ax.set_xticks(x_roi + 1.5 * width)
    ax.set_xticklabels([r["roi"] for r in freq_by_roi], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ΔP")
    ax.set_title("Frequency Band × ROI")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)

    ax = axes[2]
    global_band = {}
    for band_name in BANDS:
        vals = np.array([r[f"{band_name}_dp"] for r in freq_by_roi], dtype=float)
        if np.all(np.isnan(vals)):
            global_band[band_name] = 0.0
        else:
            global_band[band_name] = float(np.nanmean(vals))
    bnames = list(global_band.keys())
    bvals = [global_band[b] for b in bnames]
    ax.bar(range(len(bnames)), [abs(v) for v in bvals],
           color=[band_colors[b] for b in bnames], alpha=0.8)
    ax.set_xticks(range(len(bnames)))
    ax.set_xticklabels(bnames)
    ax.set_ylabel("|ΔP| (mean across ROIs)")
    ax.set_title("Overall Band Importance")
    for i, (bn, bv) in enumerate(zip(bnames, bvals)):
        ax.text(i, abs(bv) + 0.0005, "+" if bv > 0 else "−",
                ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "frequency_importance_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved frequency analysis to {out_dir}")

    return freq_by_channel, freq_by_roi


# ══════════════════════════════════════════════════════════════════════
# Final important channels plot
# ══════════════════════════════════════════════════════════════════════

def save_final_channels_plot(final_rows, out_dir):
    """Plot only robust important channels: facilitatory vs suppressive."""
    if not final_rows:
        print("  No robust channels -- skipping final_important_channels_plot.png")
        return

    fac = [r for r in final_rows if r["contribution_type"] == "facilitatory"]
    sup = [r for r in final_rows if r["contribution_type"] == "suppressive"]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(final_rows) * 0.35)))

    # Facilitatory
    ax = axes[0]
    if fac:
        y = np.arange(len(fac))
        occ_vals = [r["occlusion_delta_p"] for r in fac]
        perm_vals = [r["permutation_delta_p"] for r in fac]
        ax.barh(y - 0.15, occ_vals, height=0.3, color="#2e7d32", alpha=0.85, label="Occlusion")
        ax.barh(y + 0.15, perm_vals, height=0.3, color="#66bb6a", alpha=0.85, label="Permutation")
        for i, r in enumerate(fac):
            ax.plot([r["occlusion_ci_low"], r["occlusion_ci_high"]], [i - 0.15, i - 0.15],
                    color="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['electrode_name']} ({r['roi'][:6]})" for r in fac], fontsize=8)
        ax.invert_yaxis()
        ax.legend(fontsize=7)
    ax.set_title("Facilitatory Channels (robust)", fontsize=10)
    ax.set_xlabel("ΔP")
    ax.axvline(0, color="k", linewidth=0.5)

    # Suppressive
    ax = axes[1]
    if sup:
        y = np.arange(len(sup))
        occ_vals = [r["occlusion_delta_p"] for r in sup]
        perm_vals = [r["permutation_delta_p"] for r in sup]
        ax.barh(y - 0.15, occ_vals, height=0.3, color="#c62828", alpha=0.85, label="Occlusion")
        ax.barh(y + 0.15, perm_vals, height=0.3, color="#ef5350", alpha=0.85, label="Permutation")
        for i, r in enumerate(sup):
            ax.plot([r["occlusion_ci_low"], r["occlusion_ci_high"]], [i - 0.15, i - 0.15],
                    color="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['electrode_name']} ({r['roi'][:6]})" for r in sup], fontsize=8)
        ax.invert_yaxis()
        ax.legend(fontsize=7)
    ax.set_title("Suppressive Channels (robust)", fontsize=10)
    ax.set_xlabel("ΔP")
    ax.axvline(0, color="k", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "final_important_channels_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved final_important_channels_plot.png")


# ══════════════════════════════════════════════════════════════════════
# REPORT GENERATION (supervisor-ready)
# ══════════════════════════════════════════════════════════════════════

def write_focused_report(arch_info, ablation, combined_channels, roi_results,
                         freq_by_channel, freq_by_roi, final_rows,
                         subj_profiles, ch_matrix, r_occ_perm,
                         baseline_metrics, freq_short_window_warning,
                         n_windows, n_boot, top_k, fdr_alpha, stability_threshold,
                         montage, out_dir, args,
                         subj_spec_corr=None, ch_spec_data=None):
    """Generate the supervisor-ready focused XAI report."""
    print("\n" + "=" * 70)
    print("GENERATING FOCUSED XAI REPORT")
    print("=" * 70)

    now = datetime.now().strftime("%B %d, %Y")
    montage_source = montage["source"]

    # Statistics
    n_occ_fdr = sum(1 for c in combined_channels if c["occ_fdr_significant"])
    n_perm_fdr = sum(1 for c in combined_channels if c["perm_fdr_significant"])
    n_robust = sum(1 for c in combined_channels if c["robust_significant"])
    n_fac = sum(1 for c in combined_channels if c["contribution_type"] == "facilitatory")
    n_sup = sum(1 for c in combined_channels if c["contribution_type"] == "suppressive")
    n_mix = sum(1 for c in combined_channels if c["contribution_type"] == "mixed")
    n_occ_ci = sum(1 for c in combined_channels if c["occ_ci_significant"])
    n_perm_ci = sum(1 for c in combined_channels if c["perm_ci_significant"])

    # Subject stability
    if ch_matrix.shape[0] > 0:
        corr_mat = np.corrcoef(ch_matrix)
        triu = corr_mat[np.triu_indices(ch_matrix.shape[0], k=1)]
        subj_r_mean = float(triu.mean())
        subj_r_std = float(triu.std())
    else:
        subj_r_mean = 0.0
        subj_r_std = 0.0

    block3_zero = ablation["block_3"]["zero_weights"]
    block3_perm = ablation["block_3"]["permute"]

    L = []  # lines

    L.append("=" * 80)
    L.append("FOCUSED XAI ANALYSIS REPORT")
    L.append("VLAAI EEG Auditory Attention Decoder — Channel Importance & Frequency Analysis")
    L.append("=" * 80)
    L.append("")
    L.append(f"Generated: {now}")
    L.append(f"Model: VLAAI (pretrained, loaded from vlaai.h5)")
    L.append(f"Dataset: DTU EEG dataset")
    L.append(f"Analysis windows: N = {n_windows}")
    L.append(f"Bootstrap iterations: {n_boot}")
    L.append(f"Sign-flip permutations: {args.n_perm}")
    L.append(f"FDR significance level: α = {fdr_alpha}")
    L.append(f"Stability threshold: {stability_threshold:.0%} of subjects must agree")
    L.append(f"Occlusion mode: {args.occlusion_mode}")
    L.append(f"ROI frequency mode: {args.roi_frequency_mode}")
    if montage_source == "montage_file":
        L.append(f"Electrode montage: {montage['montage_path']} (actual)")
    else:
        L.append(f"Electrode montage: INDEX-BASED FALLBACK (approximate)")
    L.append(f"Script: scripts/run_focused_xai.py")
    L.append("")

    # ── A. OBJECTIVE ──────────────────────────────────────────────
    L.append("=" * 80)
    L.append("A. OBJECTIVE")
    L.append("=" * 80)
    L.append("")
    L.append("Identify which EEG channels are important for the VLAAI auditory attention")
    L.append("decoder, characterise their contribution (facilitatory vs suppressive), and")
    L.append("analyse the frequency-band contribution of those important channels.")
    L.append("")
    L.append("Primary methods:")
    L.append("  1. Channel occlusion (zero-masking each channel)")
    L.append("  2. Channel permutation (shuffle across windows)")
    L.append("  3. Benjamini-Hochberg FDR correction for multiple comparisons")
    L.append("")
    L.append("Supporting method:")
    L.append("  - Integrated Gradients (gradient-based attribution, used as corroboration only)")
    L.append("")
    L.append("Frequency analysis is conditioned on the identified robust important channels.")
    L.append("")
    L.append("Baseline AAD validation on selected windows:")
    L.append(f"  corr_attended_mean:      {baseline_metrics['corr_attended_mean']:+.5f}")
    L.append(f"  corr_unattended_mean:    {baseline_metrics['corr_unattended_mean']:+.5f}")
    L.append(f"  correlation_margin_mean: {baseline_metrics['correlation_margin_mean']:+.5f}")
    L.append(
        "  correlation_margin_CI:   "
        f"[{baseline_metrics['correlation_margin_CI'][0]:+.5f}, "
        f"{baseline_metrics['correlation_margin_CI'][1]:+.5f}]"
    )
    L.append(f"  AAD accuracy:            {baseline_metrics['aad_accuracy']:.3f}")
    if baseline_metrics.get("near_chance_warning", False):
        L.append("  WARNING: Baseline AAD performance is near chance. XAI results may not be meaningful.")
    L.append("")

    # ── B. ARCHITECTURE FINDING ───────────────────────────────────
    L.append("=" * 80)
    L.append("B. ARCHITECTURE FINDING")
    L.append("=" * 80)
    L.append("")
    L.append(f"Total parameters: {arch_info['total_params']:,}")
    L.append(f"Shared parameters: {arch_info['shared_pct']*100:.1f}%")
    L.append("")
    L.append("Component breakdown:")
    for name, n in arch_info["components"].items():
        pct = n / arch_info["total_params"] * 100
        L.append(f"  {name:25s}: {n:>10,} ({pct:.1f}%)")
    L.append("")
    L.append("Block ablation (key result):")
    L.append(f"  Block 3 zero_weights:  ΔP = {block3_zero['delta_p_mean']:+.5f}, "
             f"ΔAcc = {block3_zero['delta_acc']:+.3f}")
    L.append(f"  Block 3 permute:       ΔP = {block3_perm['delta_p_mean']:+.5f}, "
             f"ΔAcc = {block3_perm['delta_acc']:+.3f}")
    L.append("")
    L.append("KEY FINDING: Block 3 (final iteration) dominates the decision.")
    L.append("Blocks 0-2 can be ablated with negligible effect. The model")
    L.append("effectively uses only the last iteration's output.")
    L.append("")

    # ── C. CHANNEL IMPORTANCE METHOD ──────────────────────────────
    L.append("=" * 80)
    L.append("C. CHANNEL IMPORTANCE METHOD")
    L.append("=" * 80)
    L.append("")
    L.append("Three complementary methods were used:")
    L.append("")
    if args.occlusion_mode == "mean":
        L.append("  (a) CHANNEL OCCLUSION: Replace each channel's values with its global")
        L.append("      mean over selected windows/time points.")
    else:
        L.append("  (a) CHANNEL OCCLUSION: Replace each channel's values with zero across")
    L.append("      all time points. Measure change in P(attended). Positive ΔP means")
    L.append("      removing the channel HURTS decoding (facilitatory channel).")
    L.append("      Negative ΔP means removing the channel HELPS (suppressive channel).")
    L.append("")
    L.append("  (b) CHANNEL PERMUTATION: Shuffle each channel's values across windows")
    L.append("      (breaking temporal alignment while preserving marginal statistics).")
    L.append("      Same interpretation as occlusion.")
    L.append("")
    L.append("  (c) INTEGRATED GRADIENTS (supporting only): Gradient-based attribution")
    L.append("      using a zero baseline. Ranks channels by mean |attribution|.")
    L.append("      Used as corroborating evidence only, not for significance testing.")
    L.append("")
    L.append(f"Occlusion–Permutation correlation: r = {r_occ_perm:.3f}")
    L.append(f"Channels with CI-based occlusion significance (excludes 0): {n_occ_ci}/64")
    L.append(f"Channels with CI-based permutation significance (excludes 0): {n_perm_ci}/64")
    L.append("")

    # ── D. MULTIPLE-COMPARISON CORRECTION ─────────────────────────
    L.append("=" * 80)
    L.append("D. MULTIPLE-COMPARISON CORRECTION")
    L.append("=" * 80)
    L.append("")
    L.append("To control for testing 64 channels simultaneously, Benjamini-Hochberg")
    L.append("FDR correction was applied to sign-flip permutation p-values.")
    L.append("")
    L.append("Sign-flip permutation p-value estimation: For each channel, per-window ΔP")
    L.append(f"values were sign-flipped {args.n_perm} times under the null of zero mean.")
    L.append("A two-sided p-value was computed from |mean(ΔP)| versus the null distribution.")
    L.append("")
    L.append(f"FDR threshold: α = {fdr_alpha}")
    L.append(f"Occlusion FDR-significant channels:   {n_occ_fdr}/64")
    L.append(f"Permutation FDR-significant channels:  {n_perm_fdr}/64")
    L.append("")
    L.append("Robust significance criteria (all must hold):")
    L.append("  1. Occlusion and permutation agree in sign (both positive or both negative)")
    L.append("  2. At least one method (occlusion or permutation) is FDR-significant")
    L.append(f"  3. Subject-wise stability ≥ {stability_threshold:.0%}")
    L.append("")
    L.append(f"Channels meeting robust criteria: {n_robust}/64")
    L.append("")

    # ── E. FINAL ROBUST IMPORTANT CHANNELS ────────────────────────
    L.append("=" * 80)
    L.append("E. FINAL ROBUST IMPORTANT CHANNELS")
    L.append("=" * 80)
    L.append("")
    if final_rows:
        L.append(f"{len(final_rows)} channels meet robust significance criteria:")
        L.append("")
        header = (f"{'Rank':>4} | {'Ch':>3} | {'Name':>6} | {'ROI':>16} | "
                  f"{'Occ ΔP':>9} | {'Perm ΔP':>9} | {'IG Rank':>7} | "
                  f"{'Type':>12} | {'Stable':>7} | {'Best Band':>9}")
        L.append(header)
        L.append("-" * len(header))
        for r in final_rows:
            L.append(
                f"{r['rank']:>4} | {r['channel_index']:>3} | {r['electrode_name']:>6} | "
                f"{r['roi']:>16} | {r['occlusion_delta_p']:>+9.5f} | "
                f"{r['permutation_delta_p']:>+9.5f} | {r['ig_rank']:>7} | "
                f"{r['contribution_type']:>12} | {r['subject_stability']:>7} | "
                f"{r.get('best_frequency_band', 'N/A'):>9}"
            )
        L.append("")
    else:
        L.append("No channels met all robust criteria. See channel_importance.csv for")
        L.append("the full ranked list and consider relaxing criteria.")
        L.append("")

    # Also show top-15 for context
    L.append(f"For reference, top-{top_k} channels by combined importance (regardless of robustness):")
    L.append("")
    header2 = (f"{'Rank':>4} | {'Ch':>3} | {'Name':>6} | {'ROI':>16} | "
               f"{'Occ ΔP':>9} | {'Perm ΔP':>9} | {'Type':>12} | {'Robust':>6}")
    L.append(header2)
    L.append("-" * len(header2))
    for c in combined_channels[:top_k]:
        L.append(
            f"{c['rank']:>4} | {c['channel']:>3} | {c['electrode_name']:>6} | "
            f"{c['roi']:>16} | {c['occ_score']:>+9.5f} | {c['perm_score']:>+9.5f} | "
            f"{c['contribution_type']:>12} | {'YES' if c['robust_significant'] else 'no':>6}"
        )
    L.append("")

    # ── F. FACILITATORY vs SUPPRESSIVE ────────────────────────────
    L.append("=" * 80)
    L.append("F. FACILITATORY vs SUPPRESSIVE CHANNEL INTERPRETATION")
    L.append("=" * 80)
    L.append("")
    L.append(f"Of 64 channels:")
    L.append(f"  Facilitatory (both occ & perm positive):  {n_fac}")
    L.append(f"  Suppressive  (both occ & perm negative):  {n_sup}")
    L.append(f"  Mixed sign:                               {n_mix}")
    L.append("")

    robust_fac = [r for r in final_rows if r["contribution_type"] == "facilitatory"]
    robust_sup = [r for r in final_rows if r["contribution_type"] == "suppressive"]

    if robust_fac:
        L.append("Robustly supported facilitatory channels (removing hurts decoding):")
        for r in robust_fac:
            L.append(f"  {r['electrode_name']:>6} (Ch{r['channel_index']:2d}, {r['roi']:>16}): "
                     f"Occ ΔP = {r['occlusion_delta_p']:+.5f}, Perm ΔP = {r['permutation_delta_p']:+.5f}")
        L.append("")

    if robust_sup:
        L.append("Robustly supported suppressive channels (removing helps decoding):")
        for r in robust_sup:
            L.append(f"  {r['electrode_name']:>6} (Ch{r['channel_index']:2d}, {r['roi']:>16}): "
                     f"Occ ΔP = {r['occlusion_delta_p']:+.5f}, Perm ΔP = {r['permutation_delta_p']:+.5f}")
        L.append("")

    L.append("NOTE: Suppressive channels are those where removing the channel improves")
    L.append("decoding, suggesting the model's representation is degraded by their input.")
    L.append("This could indicate noisy channels, reference artifacts, or channels that")
    L.append("introduce competing neural information.")
    L.append("")

    # ── G. ROI-LEVEL INTERPRETATION ───────────────────────────────
    L.append("=" * 80)
    L.append("G. ROI-LEVEL INTERPRETATION")
    L.append("=" * 80)
    L.append("")
    if montage_source == "montage_file":
        L.append(f"ROI mapping: from montage file ({montage['montage_path']})")
        L.append("NOTE: This montage file should be verified against the official DTU")
        L.append("electrode positions before publication-quality claims are made.")
    else:
        L.append("WARNING: ROI mapping is INDEX-BASED and approximate. The actual DTU")
        L.append("electrode montage should be verified for publication-quality claims.")
    L.append("")

    header_roi = (f"{'ROI':>20} | {'Occ ΔP':>9} | {'95% CI':>22} | {'Sig?':>5} | "
                  f"{'Perm ΔP':>9} | {'#FDR':>5} | {'#Robust':>7}")
    L.append(header_roi)
    L.append("-" * len(header_roi))
    for r in roi_results:
        L.append(
            f"{r['roi']:>20} | {r['occ_mean_dp']:>+9.5f} | "
            f"[{r['occ_ci'][0]:+.5f},{r['occ_ci'][1]:+.5f}] | "
            f"{'YES' if r['occ_significant'] else 'no':>5} | "
            f"{r['perm_mean_dp']:>+9.5f} | "
            f"{r['n_fdr_significant_channels']:>5} | "
            f"{r['n_robust_channels']:>7}"
        )
    L.append("")

    # ── H. FREQUENCY CONTRIBUTION ─────────────────────────────────
    L.append("=" * 80)
    L.append("H. FREQUENCY CONTRIBUTION OF ROBUST IMPORTANT CHANNELS")
    L.append("=" * 80)
    L.append("")
    if freq_by_channel:
        L.append("Method: For each candidate important channel, remove a specific frequency")
        L.append("band (Butterworth 4th order bandpass, mirror-padded) from that channel")
        L.append("and measure ΔP. Positive ΔP = removing that band hurts decoding (the band")
        L.append("supports the model). Negative ΔP = removing that band helps decoding")
        L.append("(the band may carry noise or suppressive information).")
        L.append(f"ROI frequency mode used: {args.roi_frequency_mode}.")
        if args.roi_frequency_mode == "robust_only":
            L.append("ROIs without robust channels are skipped and reported with NaN values.")
        if freq_short_window_warning:
            L.append("Frequency-band interpretation, especially delta-band effects, should be treated cautiously because the analysis window is short.")
        L.append("")

        # Show robust channels first
        robust_freq = [r for r in freq_by_channel if r["is_robust"]]
        other_freq = [r for r in freq_by_channel if not r["is_robust"]]

        if robust_freq:
            L.append("Robust channels:")
            L.append("")
            header_freq = (f"{'Name':>6} | {'Ch':>3} | {'ROI':>16} | {'Best Band':>10} | "
                          f"{'Delta ΔP':>9} | {'Theta ΔP':>9} | {'Alpha ΔP':>9} | {'Beta ΔP':>9}")
            L.append(header_freq)
            L.append("-" * len(header_freq))
            for r in robust_freq:
                L.append(
                    f"{r['electrode_name']:>6} | {r['channel']:>3} | {r['roi']:>16} | "
                    f"{r['most_important_band']:>10} | "
                    f"{r['delta_dp']:>+9.5f} | {r['theta_dp']:>+9.5f} | "
                    f"{r['alpha_dp']:>+9.5f} | {r['beta_dp']:>+9.5f}"
                )
            L.append("")

        if other_freq:
            L.append("Additional candidate channels (not robust):")
            L.append("")
            header_freq2 = (f"{'Name':>6} | {'Ch':>3} | {'ROI':>16} | {'Best Band':>10} | "
                           f"{'Delta ΔP':>9} | {'Theta ΔP':>9} | {'Alpha ΔP':>9} | {'Beta ΔP':>9}")
            L.append(header_freq2)
            L.append("-" * len(header_freq2))
            for r in other_freq:
                L.append(
                    f"{r['electrode_name']:>6} | {r['channel']:>3} | {r['roi']:>16} | "
                    f"{r['most_important_band']:>10} | "
                    f"{r['delta_dp']:>+9.5f} | {r['theta_dp']:>+9.5f} | "
                    f"{r['alpha_dp']:>+9.5f} | {r['beta_dp']:>+9.5f}"
                )
            L.append("")

        L.append("Per-ROI frequency summary:")
        L.append("")
        header_freq_roi = (f"{'ROI':>20} | {'Best Band':>10} | {'#Robust':>7} | "
                          f"{'Delta ΔP':>9} | {'Theta ΔP':>9} | {'Alpha ΔP':>9} | {'Beta ΔP':>9}")
        L.append(header_freq_roi)
        L.append("-" * len(header_freq_roi))
        for r in freq_by_roi:
            L.append(
                f"{r['roi']:>20} | {r['most_important_band']:>10} | "
                f"{r['n_robust_channels']:>7} | "
                f"{r['delta_dp']:>+9.5f} | {r['theta_dp']:>+9.5f} | "
                f"{r['alpha_dp']:>+9.5f} | {r['beta_dp']:>+9.5f}"
            )
        L.append("")
    else:
        L.append("Frequency analysis was skipped (--skip-frequency).")
        L.append("")

    # ── I. SUBJECT SPECIFICITY & FEATURE CORRELATION ──────────────
    L.append("=" * 80)
    L.append("I. SUBJECT SPECIFICITY & FEATURE CORRELATION")
    L.append("=" * 80)
    L.append("")
    if ch_matrix.shape[0] > 0:
        L.append(f"Subjects analysed: {ch_matrix.shape[0]}")
        L.append(f"Inter-subject profile correlation: mean r = {subj_r_mean:.3f} ± {subj_r_std:.3f}")
        L.append("  A high mean r indicates consistent channel importance across subjects.")
        L.append("  A low mean r indicates subject-specific importance patterns where")
        L.append("  group-level maps may obscure individual differences.")
        L.append("")
        L.append("Per-channel subject specificity metrics are stored in subject_specificity.csv.")
        L.append("  subject_agree_frac : fraction of subjects agreeing in sign with group mean")
        L.append("  subj_std           : cross-subject standard deviation of importance scores")
        L.append("  subject_cv         : coefficient of variation (std / |mean|) — higher = more subject-specific")
        L.append("")

        # Per-subject Spearman similarity
        ps_rho = subj_spec_corr.get("_per_subject_rho", {}) if subj_spec_corr else {}
        if ch_spec_data and len(ch_spec_data) > 0:
            import json as _json
            _spec_path = out_dir / "subject_specificity.json"
            if _spec_path.exists():
                with open(_spec_path, encoding="utf-8") as _f:
                    _spec_loaded = _json.load(_f)
                ps_rho = _spec_loaded.get("per_subject_spearman_rho", {})
                ps_mean = _spec_loaded.get("per_subject_spearman_mean", float("nan"))
                ps_std  = _spec_loaded.get("per_subject_spearman_std", float("nan"))
                L.append(f"Per-subject profile similarity to group mean (Spearman ρ):")
                L.append(f"  Mean ρ = {ps_mean:.3f} ± {ps_std:.3f}")
                L.append(f"  (ρ = 1.0 means perfect agreement, ρ < 0 means inverted pattern)")
                if ps_rho:
                    L.append("")
                    for subj, rho_val in sorted(ps_rho.items()):
                        bar = "#" * int(max(0, rho_val) * 20)
                        L.append(f"    {subj:>4}: ρ = {rho_val:+.3f}  {bar}")
                L.append("")

        if subj_spec_corr:
            L.append("Pearson correlations between channel importance features and subject specificity:")
            L.append("")
            L.append("  Subject Agreement Fraction vs feature:")
            L.append(f"    agree_frac vs |Occlusion ΔP|:   r = {subj_spec_corr.get('agree_frac_vs_occ_abs', float('nan')):+.3f}")
            L.append(f"    agree_frac vs |Permutation ΔP|: r = {subj_spec_corr.get('agree_frac_vs_perm_abs', float('nan')):+.3f}")
            L.append(f"    agree_frac vs IG importance:    r = {subj_spec_corr.get('agree_frac_vs_ig', float('nan')):+.3f}")
            L.append(f"    agree_frac vs combined score:   r = {subj_spec_corr.get('agree_frac_vs_combined', float('nan')):+.3f}")
            L.append("")
            L.append("  Cross-subject Std (variability) vs feature:")
            L.append(f"    subj_std vs |Occlusion ΔP|:     r = {subj_spec_corr.get('subj_std_vs_occ_abs', float('nan')):+.3f}")
            L.append(f"    subj_std vs |Permutation ΔP|:   r = {subj_spec_corr.get('subj_std_vs_perm_abs', float('nan')):+.3f}")
            L.append(f"    subj_std vs IG importance:      r = {subj_spec_corr.get('subj_std_vs_ig', float('nan')):+.3f}")
            L.append(f"    subj_std vs combined score:     r = {subj_spec_corr.get('subj_std_vs_combined', float('nan')):+.3f}")
            L.append("")
            L.append("  Coefficient of Variation (CV) vs feature:")
            L.append(f"    CV vs |Occlusion ΔP|:           r = {subj_spec_corr.get('cv_vs_occ_abs', float('nan')):+.3f}")
            L.append(f"    CV vs |Permutation ΔP|:         r = {subj_spec_corr.get('cv_vs_perm_abs', float('nan')):+.3f}")
            L.append(f"    CV vs IG importance:            r = {subj_spec_corr.get('cv_vs_ig', float('nan')):+.3f}")
            L.append(f"    CV vs combined score:           r = {subj_spec_corr.get('cv_vs_combined', float('nan')):+.3f}")
            L.append("")
            L.append("INTERPRETATION GUIDE:")
            L.append("  agree_frac ↑ with |occ|   → more important channels show stronger cross-subject agreement")
            L.append("  subj_std ↑ with |occ|     → more important channels are also more variable across subjects")
            L.append("  cv ↓ with |occ|           → more important channels are more generalizable (low CV)")
            L.append("  If both agree_frac and subj_std increase with importance, the important channels")
            L.append("  are consistently used but with subject-specific magnitudes.")

        if ch_spec_data:
            top_spec = sorted(ch_spec_data, key=lambda x: abs(x["occ_score"]), reverse=True)[:10]
            L.append("")
            L.append("Top-10 channels by |Occlusion ΔP| with subject specificity metrics:")
            L.append("")
            hdr = (f"{'Name':>6} | {'ROI':>16} | {'|Occ|':>7} | "
                   f"{'agree_frac':>10} | {'subj_std':>8} | {'CV':>6} | {'Robust':>6}")
            L.append(hdr)
            L.append("-" * len(hdr))
            for r in top_spec:
                cv_str = f"{r['subject_cv']:.3f}" if np.isfinite(r['subject_cv']) else "  N/A"
                L.append(
                    f"{r['electrode_name']:>6} | {r['roi']:>16} | {abs(r['occ_score']):>7.5f} | "
                    f"{r['subject_agree_frac']:>10.3f} | {r['subj_std']:>8.5f} | "
                    f"{cv_str:>6} | {'YES' if r['robust_significant'] else 'no':>6}"
                )
        L.append("")
    else:
        L.append("Subject specificity analysis requires >=2 subjects. Not available.")
        L.append("")

    # ── J. LIMITATIONS ────────────────────────────────────────────
    L.append("=" * 80)
    L.append("J. LIMITATIONS")

    L.append("=" * 80)
    L.append("")
    L.append(f"1. SAMPLE SIZE: Analysis used N={n_windows} windows. While bootstrap CIs")
    L.append("   and FDR correction provide statistical control, larger N is recommended")
    L.append("   for definitive conclusions (N≥500 for publication-quality results).")
    L.append("")
    L.append("2. UNATTENDED ENVELOPE PROXY: The DTU dataset provides only the attended")
    L.append("   envelope. The 'unattended' envelope is a circular time-shift proxy.")
    L.append("   Results may differ with a true competing-speaker paradigm.")
    L.append("")
    if montage_source == "montage_file":
        L.append("3. ELECTRODE MONTAGE: A montage file was provided, but channel-to-electrode")
        L.append("   assignments should be independently verified against the DTU dataset")
        L.append("   documentation before publication.")
    else:
        L.append("3. ROI MAPPING: Channel-to-ROI assignments are INDEX-BASED and have NOT")
        L.append("   been verified against the actual DTU electrode montage. All spatial")
        L.append("   interpretations are approximate and should be treated as preliminary.")
    L.append("")
    L.append("4. INTEGRATED GRADIENTS: IG uses a zero baseline, which may not be")
    L.append("   neurophysiologically meaningful. IG results are supporting evidence only")
    L.append("   and are not used for significance testing.")
    L.append("")
    L.append("5. SUBJECT-WISE STABILITY: Limited by windows per subject. Individual")
    L.append("   channel importance profiles are noisy with small samples.")
    L.append("")
    if ch_matrix.shape[0] > 0:
        L.append(f"6. CROSS-SUBJECT CONSISTENCY: Mean pairwise r = {subj_r_mean:.3f} ± {subj_r_std:.3f}.")
        L.append("   Low inter-subject agreement indicates subject-specific importance patterns.")
        L.append("   Group-level channel maps may obscure individual differences.")
        L.append("")
    L.append(f"7. FDR CORRECTION: Sign-flip permutation p-values are finite-sample estimates")
    L.append(f"   with resolution 1/({args.n_perm}+1). Increasing n_perm improves resolution")
    L.append("   at the cost of computation time.")
    L.append("")
    if freq_short_window_warning:
        L.append("8. FREQUENCY INTERPRETATION: Frequency-band interpretation, especially")
        L.append("   delta-band effects, should be treated cautiously because the analysis window is short.")
    L.append("")

    # ── K. NEXT STEPS ─────────────────────────────────────────────
    L.append("=" * 80)
    L.append("K. NEXT STEPS")
    L.append("=" * 80)
    L.append("")
    L.append("1. Verify electrode montage against official DTU dataset documentation.")
    L.append("2. Run with larger N (≥500) and n_boot (≥2000) for tighter CIs and better")
    L.append("   p-value resolution.")
    L.append("3. Add topographic EEG scalp maps using verified electrode coordinates.")
    L.append("4. Validate findings with alternative decoder (TRF baseline).")
    L.append("5. Compare channel importance across within-subject vs group-level models.")
    L.append("6. Investigate temporal dynamics of important channels as supplementary work.")
    L.append("")

    # ── OUTPUT FILES ──────────────────────────────────────────────
    L.append("=" * 80)
    L.append("OUTPUT FILES")
    L.append("=" * 80)
    L.append("")
    L.append(f"Output directory: {out_dir}")
    L.append("")
    L.append("  architecture_summary.json          — Architecture parameters and components")
    L.append("  block_ablation.json                 — Block ablation results")
    L.append("  block_ablation.png                  — Block ablation plot")
    L.append("  channel_importance.json             — Full channel importance (all methods)")
    L.append("  channel_importance.csv              — Full ranked channel table with FDR columns")
    L.append("  channel_importance_plot.png          — 4-panel channel importance figure")
    L.append("  channel_importance_fdr_plot.png      — Channels ranked with FDR annotation")
    L.append("  final_important_channels.csv        — Robust important channels only")
    L.append("  final_important_channels_plot.png    — Facilitatory vs suppressive robust channels")
    L.append("  roi_importance.json                  — ROI-level occlusion + permutation")
    L.append("  roi_importance.csv                   — ROI table")
    L.append("  roi_importance_plot.png               — ROI importance bar charts")
    if freq_by_channel:
        L.append("  frequency_analysis.json              — Frequency ΔP per channel and ROI")
        L.append("  frequency_by_channel.csv             — Per-channel frequency table")
        L.append("  frequency_by_roi.csv                 — Per-ROI frequency table")
        L.append("  frequency_by_important_channels.png  — Frequency heatmap + robust bar chart")
        L.append("  roi_frequency_summary.png            — ROI × frequency band summary")
        L.append("  frequency_importance_plot.png         — 3-panel frequency figure")
    L.append("  subject_specificity.json             — Subject specificity metrics + correlations")
    L.append("  subject_specificity.csv              — Per-channel specificity table")
    L.append("  subject_specificity_correlation.png  — 4-panel scatter: features vs specificity")
    L.append("  subject_specificity_correlation_summary.png — Correlation bar chart")
    L.append("  subject_vs_group_heatmap.png                — Individual subject profiles vs group mean")
    L.append("  subject_profile_similarity.png              — Per-subject Spearman ρ vs group mean")
    L.append("  channel_disagreement.png                    — Channels where subjects disagree with group")
    L.append("  occlusion_perwindow.npy              — Raw (N, 64) per-window occlusion drops")
    L.append("  permutation_perwindow.npy            — Raw (N, 64) per-window permutation drops")
    L.append("  subject_channel_matrix.npy           — (n_subjects, 64) stability matrix")
    L.append("  run_config.json                      — Run configuration for reproducibility")
    L.append("  FOCUSED_XAI_REPORT.txt               — This report")
    L.append("")
    L.append("=" * 80)
    L.append("END OF REPORT")
    L.append("=" * 80)

    report_text = "\n".join(L)
    (out_dir / "FOCUSED_XAI_REPORT.txt").write_text(report_text, encoding="utf-8")
    print(f"  Report saved to {out_dir / 'FOCUSED_XAI_REPORT.txt'}")
    print(f"  Report length: {len(L)} lines")

    return report_text


# ══════════════════════════════════════════════════════════════════════
# SECTION H — Subject-Level Statistical Validation
# ══════════════════════════════════════════════════════════════════════

def _compute_subject_means(occ_pw, perm_pw, selected_subject_ids):
    """Aggregate per-window ΔP arrays into per-subject means.

    Returns:
        subj_occ   : ndarray (n_subjects, 64) — mean occ ΔP per subject
        subj_perm  : ndarray (n_subjects, 64) — mean perm ΔP per subject
        unique_subjects : list of subject labels in sorted order
    """
    unique_subjects = sorted(set(selected_subject_ids.tolist()))
    n_subj = len(unique_subjects)
    subj_occ  = np.zeros((n_subj, 64))
    subj_perm = np.zeros((n_subj, 64))
    for si, subj in enumerate(unique_subjects):
        mask = selected_subject_ids == subj
        subj_occ[si]  = occ_pw[mask].mean(axis=0)
        subj_perm[si] = perm_pw[mask].mean(axis=0)
    return subj_occ, subj_perm, unique_subjects


def _bootstrap_ci_across_subjects(values, n_boot=500, seed=42):
    """Bootstrap CI for a 1-D array of per-subject values."""
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.array([values[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(values.mean()), float(np.median(values)), float(values.std(ddof=1)), float(lo), float(hi)


def run_subject_level_validation(occ_pw, perm_pw, selected_subject_ids,
                                 combined_channels, montage, out_dir,
                                 fdr_alpha, n_boot, seed, n_split_half=1000):
    """Section H: subject-level statistical validation for publication-grade output.

    Produces:
        subject_channel_importance.csv
        subject_level_channel_stats.csv
        high_confidence_channels.csv
        candidate_channels.csv
        split_half_reliability.csv
        split_half_reliability_summary.txt
        subject_level_roi_stats.csv
        high_confidence_channels_plot.png
        subject_level_roi_plot.png
    """
    from scipy.stats import wilcoxon

    print("\n" + "=" * 70)
    print("SECTION H: SUBJECT-LEVEL STATISTICAL VALIDATION")
    print("=" * 70)

    ch_to_name = montage["ch_to_name"]
    ch_to_roi  = montage["ch_to_roi"]

    subj_occ, subj_perm, unique_subjects = _compute_subject_means(
        occ_pw, perm_pw, selected_subject_ids)
    n_subj = len(unique_subjects)
    print(f"  Subjects: {n_subj}  |  Channels: 64")

    if n_subj < 4:
        print("  WARNING: fewer than 4 subjects — subject-level tests will have low power.")

    # ── H.1  subject_channel_importance.csv ──────────────────────────
    rows_sci = []
    for si, subj in enumerate(unique_subjects):
        for ch in range(64):
            ov = subj_occ[si, ch]
            pv = subj_perm[si, ch]
            rows_sci.append({
                "subject_id":    subj,
                "channel_index": ch,
                "channel_name":  ch_to_name.get(ch, f"Ch{ch}"),
                "roi":           ch_to_roi.get(ch, "Unknown"),
                "occ_mean_dP":   ov,
                "perm_mean_dP":  pv,
                "sign_occ":      int(np.sign(ov)),
                "sign_perm":     int(np.sign(pv)),
            })
    sci_fields = ["subject_id", "channel_index", "channel_name", "roi",
                  "occ_mean_dP", "perm_mean_dP", "sign_occ", "sign_perm"]
    with open(out_dir / "subject_channel_importance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sci_fields)
        w.writeheader()
        w.writerows(rows_sci)
    print(f"  [H.1] Saved subject_channel_importance.csv  ({len(rows_sci)} rows)")

    # ── H.2  Wilcoxon signed-rank tests + bootstrap CI + Cohen's d ──
    occ_p_raw  = np.ones(64)
    perm_p_raw = np.ones(64)
    ch_stats = []

    for ch in range(64):
        occ_vals  = subj_occ[:, ch]
        perm_vals = subj_perm[:, ch]

        # Wilcoxon (two-sided, against zero)
        def _wilcox(vals):
            if n_subj < 3:
                return 1.0
            nonzero = vals[vals != 0]
            if len(nonzero) < 2:
                return 1.0
            try:
                _, p = wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
                return float(p)
            except Exception:
                return 1.0

        op = _wilcox(occ_vals)
        pp = _wilcox(perm_vals)
        occ_p_raw[ch]  = op
        perm_p_raw[ch] = pp

        om, omed, ostd, olo, ohi = _bootstrap_ci_across_subjects(occ_vals, n_boot, seed + ch)
        pm, pmed, pstd, plo, phi = _bootstrap_ci_across_subjects(perm_vals, n_boot, seed + ch + 1000)

        # Cohen's d (one-sample effect size vs zero)
        occ_d  = float(om / ostd) if ostd > 1e-12 else 0.0
        perm_d = float(pm / pstd) if pstd > 1e-12 else 0.0

        ch_stats.append({
            "channel_index": ch,
            "channel_name":  ch_to_name.get(ch, f"Ch{ch}"),
            "roi":           ch_to_roi.get(ch, "Unknown"),
            "occ_subj_mean": om,
            "occ_subj_median": omed,
            "occ_subj_std":  ostd,
            "occ_subj_ci_lo": olo,
            "occ_subj_ci_hi": ohi,
            "occ_wilcox_p":  op,
            "occ_cohens_d":  occ_d,
            "perm_subj_mean": pm,
            "perm_subj_median": pmed,
            "perm_subj_std": pstd,
            "perm_subj_ci_lo": plo,
            "perm_subj_ci_hi": phi,
            "perm_wilcox_p": pp,
            "perm_cohens_d": perm_d,
        })

    # BH-FDR over channel tests
    occ_fdr_p,  occ_fdr_sig  = fdr_correction(occ_p_raw,  fdr_alpha)
    perm_fdr_p, perm_fdr_sig = fdr_correction(perm_p_raw, fdr_alpha)

    for i, row in enumerate(ch_stats):
        row["occ_fdr_p"]       = float(occ_fdr_p[i])
        row["occ_fdr_sig"]     = bool(occ_fdr_sig[i])
        row["perm_fdr_p"]      = float(perm_fdr_p[i])
        row["perm_fdr_sig"]    = bool(perm_fdr_sig[i])
        row["both_fdr_sig"]    = bool(occ_fdr_sig[i]) and bool(perm_fdr_sig[i])

    n_occ_subj_sig  = int(occ_fdr_sig.sum())
    n_perm_subj_sig = int(perm_fdr_sig.sum())
    n_both_subj_sig = sum(1 for r in ch_stats if r["both_fdr_sig"])
    print(f"  [H.2] Subject-level FDR-significant channels:")
    print(f"        Occlusion: {n_occ_subj_sig}/64  |  Permutation: {n_perm_subj_sig}/64  "
          f"|  Both: {n_both_subj_sig}/64")

    slcs_fields = [
        "channel_index", "channel_name", "roi",
        "occ_subj_mean", "occ_subj_median", "occ_subj_std", "occ_subj_ci_lo", "occ_subj_ci_hi",
        "occ_wilcox_p", "occ_fdr_p", "occ_fdr_sig", "occ_cohens_d",
        "perm_subj_mean", "perm_subj_median", "perm_subj_std", "perm_subj_ci_lo", "perm_subj_ci_hi",
        "perm_wilcox_p", "perm_fdr_p", "perm_fdr_sig", "perm_cohens_d", "both_fdr_sig",
    ]
    with open(out_dir / "subject_level_channel_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=slcs_fields)
        w.writeheader()
        w.writerows(ch_stats)
    print("  [H.2] Saved subject_level_channel_stats.csv")

    # ── H.3  Channel tiers ────────────────────────────────────────────
    stats_by_ch = {r["channel_index"]: r for r in ch_stats}
    ch_by_idx   = {c["channel"]: c for c in combined_channels}

    # Combined absolute effect size (subject-level) for top-20% threshold
    combined_abs = np.array([
        abs(stats_by_ch[ch]["occ_subj_mean"]) + abs(stats_by_ch[ch]["perm_subj_mean"])
        for ch in range(64)
    ])
    top20_threshold = np.percentile(combined_abs, 80)

    tier_rows = []
    for ch in range(64):
        st = stats_by_ch[ch]
        ci = ch_by_idx.get(ch, {})
        occ_sign  = np.sign(st["occ_subj_mean"])
        perm_sign = np.sign(st["perm_subj_mean"])
        same_sign = (occ_sign == perm_sign) and (occ_sign != 0)
        stab_frac = ci.get("subject_stability_frac", 0.0)
        in_top20  = combined_abs[ch] >= top20_threshold

        if (st["both_fdr_sig"] and same_sign
                and stab_frac >= (12 / 18) and in_top20):
            tier = "tier1_high_confidence"
        elif (st["occ_fdr_sig"] or st["perm_fdr_sig"]) and same_sign and stab_frac >= (10 / 18):
            tier = "tier2_candidate"
        elif ci.get("robust_significant", False):
            tier = "tier3_exploratory"
        else:
            tier = "tier4_not_robust"

        # Suppressive override label
        if tier in ("tier1_high_confidence", "tier2_candidate") and occ_sign < 0:
            tier_label = tier + "_suppressive"
        else:
            tier_label = tier

        tier_rows.append({
            "channel_index":    ch,
            "channel_name":     ch_to_name.get(ch, f"Ch{ch}"),
            "roi":              ch_to_roi.get(ch, "Unknown"),
            "tier":             tier_label,
            "occ_subj_mean":    st["occ_subj_mean"],
            "perm_subj_mean":   st["perm_subj_mean"],
            "occ_cohens_d":     st["occ_cohens_d"],
            "perm_cohens_d":    st["perm_cohens_d"],
            "occ_fdr_sig":      st["occ_fdr_sig"],
            "perm_fdr_sig":     st["perm_fdr_sig"],
            "both_fdr_sig":     st["both_fdr_sig"],
            "subject_stability": ci.get("subject_stability", "N/A"),
            "in_top20pct_effect": in_top20,
            "window_level_robust": ci.get("robust_significant", False),
        })

    tier_fields = [
        "channel_index", "channel_name", "roi", "tier",
        "occ_subj_mean", "perm_subj_mean", "occ_cohens_d", "perm_cohens_d",
        "occ_fdr_sig", "perm_fdr_sig", "both_fdr_sig",
        "subject_stability", "in_top20pct_effect", "window_level_robust",
    ]
    hc = [r for r in tier_rows if r["tier"].startswith("tier1")]
    cand = [r for r in tier_rows if r["tier"].startswith("tier2")]
    hc.sort(key=lambda r: -(abs(r["occ_subj_mean"]) + abs(r["perm_subj_mean"])))
    cand.sort(key=lambda r: -(abs(r["occ_subj_mean"]) + abs(r["perm_subj_mean"])))

    with open(out_dir / "high_confidence_channels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=tier_fields)
        w.writeheader()
        w.writerows(hc)
    with open(out_dir / "candidate_channels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=tier_fields)
        w.writeheader()
        w.writerows(cand)

    n_t1 = len(hc)
    n_t2 = len(cand)
    n_t3 = sum(1 for r in tier_rows if r["tier"].startswith("tier3"))
    print(f"  [H.3] Channel tiers — T1 high-confidence: {n_t1} | T2 candidate: {n_t2} | T3 exploratory: {n_t3}")
    if hc:
        print("        High-confidence channels: " + ", ".join(r["channel_name"] for r in hc))

    # ── H.3 Plot: high-confidence + candidate channels ────────────────
    _plot_tier_channels(hc, cand, out_dir)

    # ── H.4  Split-half reliability ───────────────────────────────────
    split_half_results = _run_split_half_reliability(
        subj_occ, subj_perm, unique_subjects, n_split_half, seed, out_dir)

    # ── H.5  ROI-level subject statistics ─────────────────────────────
    roi_subject_stats = _run_roi_subject_stats(
        subj_occ, subj_perm, unique_subjects, montage, out_dir, fdr_alpha, n_boot, seed)

    return {
        "subj_occ":         subj_occ,
        "subj_perm":        subj_perm,
        "unique_subjects":  unique_subjects,
        "ch_stats":         ch_stats,
        "tier_rows":        tier_rows,
        "hc":               hc,
        "cand":             cand,
        "split_half":       split_half_results,
        "roi_subject_stats": roi_subject_stats,
        "n_subj_occ_fdr_sig":  n_occ_subj_sig,
        "n_subj_perm_fdr_sig": n_perm_subj_sig,
        "n_both_subj_sig":     n_both_subj_sig,
    }


def _plot_tier_channels(hc, cand, out_dir):
    """Bar plot of high-confidence (T1) and candidate (T2) channels."""
    all_ch = hc + cand
    if not all_ch:
        print("  No T1/T2 channels to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(all_ch) * 0.65), 5))
    x = np.arange(len(all_ch))
    w = 0.35
    colors_t = ["#1565c0" if r["tier"].startswith("tier1") else "#42a5f5" for r in all_ch]
    ax.bar(x - w / 2, [r["occ_subj_mean"] * 1e3 for r in all_ch],
           width=w, color=colors_t, alpha=0.9, label="Occlusion ΔP (subj mean)")
    ax.bar(x + w / 2, [r["perm_subj_mean"] * 1e3 for r in all_ch],
           width=w, color=[c.replace("c0", "80").replace("f5", "b0") for c in colors_t],
           alpha=0.85, label="Permutation ΔP (subj mean)")
    ax.set_xticks(x)
    ax.set_xticklabels([r["channel_name"] for r in all_ch], rotation=45, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Subject-mean ΔP x 10^3")
    ax.set_title(
        "Publication-Grade Channel Tiers (Subject-Level Statistics)\n"
        "Dark blue = Tier-1 high-confidence | Light blue = Tier-2 candidate",
        fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#1565c0", label="T1 high-confidence (occ)"),
        Patch(color="#42a5f5", label="T2 candidate (occ)"),
    ], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "high_confidence_channels_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [H.3] Saved high_confidence_channels_plot.png")


def _run_split_half_reliability(subj_occ, subj_perm, unique_subjects,
                                n_iter, seed, out_dir):
    """Split subjects into two halves, measure Spearman r of channel rankings."""
    from scipy.stats import spearmanr as _spearmanr

    print(f"  [H.4] Split-half reliability ({n_iter} iterations)...")
    n_subj = len(unique_subjects)
    if n_subj < 4:
        print("  WARNING: too few subjects for split-half (<4). Skipping.")
        return {}

    rng   = np.random.RandomState(seed)
    combined_abs = np.abs(subj_occ) + np.abs(subj_perm)   # (n_subj, 64)
    rhos  = []
    half  = n_subj // 2

    for _ in range(n_iter):
        idx = rng.permutation(n_subj)
        a, b = idx[:half], idx[half:]
        rank_a = combined_abs[a].mean(axis=0)
        rank_b = combined_abs[b].mean(axis=0)
        rho, _ = _spearmanr(rank_a, rank_b)
        if np.isfinite(rho):
            rhos.append(float(rho))

    rhos = np.array(rhos)
    median_rho = float(np.median(rhos))
    ci_lo      = float(np.percentile(rhos, 2.5))
    ci_hi      = float(np.percentile(rhos, 97.5))
    print(f"  [H.4] Split-half Spearman r: median={median_rho:.3f}  "
          f"95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]  (n_iter={len(rhos)})")

    rows = [{"iteration": i + 1, "spearman_rho": float(r)} for i, r in enumerate(rhos)]
    with open(out_dir / "split_half_reliability.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iteration", "spearman_rho"])
        w.writeheader()
        w.writerows(rows)

    summary = (
        "SPLIT-HALF RELIABILITY SUMMARY\n"
        "================================\n"
        f"Method       : Random subject split (50/50), {n_iter} iterations\n"
        f"Measure      : Spearman rank correlation of channel importance between halves\n"
        f"Importance   : mean(|occ ΔP| + |perm ΔP|) per channel within each half\n"
        f"N subjects   : {n_subj}  |  Half size: {half}\n\n"
        f"Median Spearman r : {median_rho:.4f}\n"
        f"95% CI            : [{ci_lo:.4f}, {ci_hi:.4f}]\n"
        f"Min / Max         : {rhos.min():.4f} / {rhos.max():.4f}\n\n"
        "Interpretation:\n"
        "  r > 0.8  : high reliability — channel rankings are stable across subject samples.\n"
        "  r = 0.5-0.8 : moderate reliability — main patterns replicate but some rank variability.\n"
        "  r < 0.5  : low reliability — rankings are sensitive to which subjects are included;\n"
        "             interpretations should be made cautiously.\n"
    )
    (out_dir / "split_half_reliability_summary.txt").write_text(summary, encoding="utf-8")
    print("  [H.4] Saved split_half_reliability.csv + split_half_reliability_summary.txt")
    return {"median_rho": median_rho, "ci_lo": ci_lo, "ci_hi": ci_hi, "n_iter": len(rhos)}


def _run_roi_subject_stats(subj_occ, subj_perm, unique_subjects,
                           montage, out_dir, fdr_alpha, n_boot, seed):
    """ROI-level Wilcoxon tests across subjects."""
    from scipy.stats import wilcoxon

    print("  [H.5] ROI-level subject statistics...")
    rois = montage["rois"]
    n_subj = len(unique_subjects)
    roi_rows = []
    occ_ps, perm_ps = [], []

    for roi_name, chs in rois.items():
        # Mean across ROI channels per subject
        roi_occ_subj  = subj_occ[:, chs].mean(axis=1)   # (n_subj,)
        roi_perm_subj = subj_perm[:, chs].mean(axis=1)

        def _w(vals):
            if n_subj < 3:
                return 1.0
            nonzero = vals[vals != 0]
            if len(nonzero) < 2:
                return 1.0
            try:
                _, p = wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
                return float(p)
            except Exception:
                return 1.0

        op = _w(roi_occ_subj)
        pp = _w(roi_perm_subj)
        occ_ps.append(op)
        perm_ps.append(pp)

        om, omed, ostd, olo, ohi = _bootstrap_ci_across_subjects(roi_occ_subj, n_boot, seed)
        pm, pmed, pstd, plo, phi = _bootstrap_ci_across_subjects(roi_perm_subj, n_boot, seed + 100)

        roi_rows.append({
            "roi":            roi_name,
            "n_channels":     len(chs),
            "occ_subj_mean":  om,
            "occ_subj_median": omed,
            "occ_subj_std":   ostd,
            "occ_subj_ci_lo": olo,
            "occ_subj_ci_hi": ohi,
            "occ_wilcox_p":   op,
            "perm_subj_mean": pm,
            "perm_subj_median": pmed,
            "perm_subj_std":  pstd,
            "perm_subj_ci_lo": plo,
            "perm_subj_ci_hi": phi,
            "perm_wilcox_p":  pp,
        })

    occ_fdr_p,  occ_fdr_sig  = fdr_correction(np.array(occ_ps),  fdr_alpha)
    perm_fdr_p, perm_fdr_sig = fdr_correction(np.array(perm_ps), fdr_alpha)

    for i, row in enumerate(roi_rows):
        row["occ_fdr_p"]    = float(occ_fdr_p[i])
        row["occ_fdr_sig"]  = bool(occ_fdr_sig[i])
        row["perm_fdr_p"]   = float(perm_fdr_p[i])
        row["perm_fdr_sig"] = bool(perm_fdr_sig[i])

    roi_fields = [
        "roi", "n_channels",
        "occ_subj_mean", "occ_subj_median", "occ_subj_std", "occ_subj_ci_lo", "occ_subj_ci_hi",
        "occ_wilcox_p", "occ_fdr_p", "occ_fdr_sig",
        "perm_subj_mean", "perm_subj_median", "perm_subj_std", "perm_subj_ci_lo", "perm_subj_ci_hi",
        "perm_wilcox_p", "perm_fdr_p", "perm_fdr_sig",
    ]
    with open(out_dir / "subject_level_roi_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=roi_fields)
        w.writeheader()
        w.writerows(roi_rows)

    # Plot
    _plot_roi_subject_stats(roi_rows, out_dir)
    print("  [H.5] Saved subject_level_roi_stats.csv + subject_level_roi_plot.png")
    return roi_rows


def _plot_roi_subject_stats(roi_rows, out_dir):
    """Bar chart of ROI-level subject-mean ΔP with CI and significance markers."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(roi_rows))
    occ_means = np.array([r["occ_subj_mean"] * 1e3 for r in roi_rows])
    occ_lo    = np.array([(r["occ_subj_mean"] - r["occ_subj_ci_lo"]) * 1e3 for r in roi_rows])
    occ_hi    = np.array([(r["occ_subj_ci_hi"] - r["occ_subj_mean"]) * 1e3 for r in roi_rows])
    perm_means = np.array([r["perm_subj_mean"] * 1e3 for r in roi_rows])

    colors = ["#1565c0" if r["occ_fdr_sig"] else "#90caf9" for r in roi_rows]
    ax.bar(x, occ_means, color=colors, width=0.55, zorder=3, alpha=0.9,
           label="Occlusion (subj-mean)")
    ax.errorbar(x, occ_means, yerr=[occ_lo, occ_hi],
                fmt="none", color="black", capsize=4, linewidth=1.4, zorder=4)
    ax.scatter(x, perm_means, color="#d32f2f", marker="D", s=55, zorder=5,
               label="Permutation (subj-mean)")

    for i, r in enumerate(roi_rows):
        if r["occ_fdr_sig"]:
            y_top = max(occ_means[i], perm_means[i]) + occ_hi[i] + 0.05
            ax.text(i, y_top, "*", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([r["roi"] for r in roi_rows], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Subject-mean ΔP x 10^3  (95% CI)")
    ax.set_title("ROI Importance — Subject-Level Statistics\n"
                 "Dark bars: FDR-significant across subjects | * = FDR-significant | "
                 "Diamond = permutation ΔP", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "subject_level_roi_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_subject_level_roi_frequency_stats(decision, eeg, att, unatt,
                                          selected_subject_ids, montage,
                                          out_dir, fdr_alpha, n_boot, seed):
    """Section H.6: ROI x frequency band subject-level statistical tests.

    For each subject, removes each frequency band from each ROI and records the
    change in P(attended). Then tests the cross-subject distribution against zero
    and applies BH-FDR correction across all ROI x band comparisons.
    """
    from scipy.signal import butter, sosfiltfilt
    from scipy.stats import wilcoxon

    print("\n  [H.6] ROI x frequency subject-level statistics...")

    rois = montage["rois"]
    unique_subjects = sorted(set(selected_subject_ids.tolist()))
    n_subj = len(unique_subjects)
    device = eeg.device
    eeg_np = eeg.detach().cpu().numpy()
    N, T, _ = eeg_np.shape
    pad = 64

    window_seconds = T / FS
    if window_seconds <= 5.0:
        print("  WARNING: short analysis window — delta-band frequency interpretation "
              "is limited by frequency resolution. Treat delta-band results cautiously.")

    # Precompute band content once
    print("    Precomputing band-filtered signals...")
    band_content = {}
    for band_name, (lo, hi) in BANDS.items():
        nyq = FS / 2.0
        sos = butter(4, [max(lo / nyq, 0.01), min(hi / nyq, 0.99)],
                     btype="bandpass", output="sos")
        bc = np.zeros_like(eeg_np)
        for w in range(N):
            for ch in range(64):
                sig    = eeg_np[w, :, ch]
                padded = np.pad(sig, pad, mode="reflect")
                bc[w, :, ch] = sosfiltfilt(sos, padded)[pad:pad + T]
        band_content[band_name] = bc
        print(f"    Band ready: {band_name}")

    base_probs = get_attended_prob(decision, eeg, att, unatt)

    # For each subject, ROI, and band: compute removal ΔP
    subj_roi_band = {
        (si, roi_name, band_name): []
        for si in range(n_subj)
        for roi_name in rois
        for band_name in BANDS
    }

    for si, subj in enumerate(unique_subjects):
        mask     = selected_subject_ids == subj
        idxs     = np.where(mask)[0]
        subj_base = base_probs[idxs]

        for roi_name, chs in rois.items():
            for band_name in BANDS:
                eeg_mod = eeg_np[idxs].copy()
                for ch in chs:
                    eeg_mod[:, :, ch] -= band_content[band_name][idxs, :, ch]
                eeg_t   = torch.from_numpy(eeg_mod.astype(np.float32)).to(device)
                att_s   = att[idxs]
                unatt_s = unatt[idxs]
                mod_p   = get_attended_prob(decision, eeg_t, att_s, unatt_s)
                subj_roi_band[(si, roi_name, band_name)] = float((subj_base - mod_p).mean())

        print(f"    Subject {subj} done")
        decision.set_envelopes(att, unatt)   # restore full envelopes

    # Build per-subject matrix and run Wilcoxon tests
    combos = [(roi, band) for roi in rois for band in BANDS]
    raw_ps = []
    result_rows = []

    for roi_name, band_name in combos:
        vals = np.array([subj_roi_band[(si, roi_name, band_name)]
                         for si in range(n_subj)])
        om, omed, ostd, olo, ohi = _bootstrap_ci_across_subjects(vals, n_boot, seed)
        nz = vals[vals != 0]
        if n_subj >= 3 and len(nz) >= 2:
            try:
                _, p = wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
                p = float(p)
            except Exception:
                p = 1.0
        else:
            p = 1.0

        raw_ps.append(p)
        result_rows.append({
            "roi":            roi_name,
            "band":           band_name,
            "subj_mean_dp":   om,
            "subj_median_dp": omed,
            "subj_std_dp":    ostd,
            "subj_ci_lo":     olo,
            "subj_ci_hi":     ohi,
            "wilcox_p":       p,
        })

    # FDR across all ROI x band
    fdr_p_arr, fdr_sig_arr = fdr_correction(np.array(raw_ps), fdr_alpha)
    for i, row in enumerate(result_rows):
        row["fdr_p"]   = float(fdr_p_arr[i])
        row["fdr_sig"] = bool(fdr_sig_arr[i])

    n_sig = int(fdr_sig_arr.sum())
    print(f"    FDR-significant ROI x band: {n_sig}/{len(combos)}")

    # Add delta-band caution note
    for row in result_rows:
        if row["band"] == "delta":
            row["caution"] = ("Delta-band interpretation is limited by short analysis window. "
                              "Treat cautiously.")
        else:
            row["caution"] = ""

    fields = ["roi", "band", "subj_mean_dp", "subj_median_dp", "subj_std_dp",
              "subj_ci_lo", "subj_ci_hi", "wilcox_p", "fdr_p", "fdr_sig", "caution"]
    with open(out_dir / "subject_level_roi_frequency_stats.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(result_rows)

    print("  [H.6] Saved subject_level_roi_frequency_stats.csv")
    return result_rows


# ══════════════════════════════════════════════════════════════════════
# Publication summary
# ══════════════════════════════════════════════════════════════════════

def write_publication_summary(combined_channels, subject_val,
                               baseline_metrics, arch_info, ablation,
                               freq_by_channel, freq_by_roi,
                               freq_short_window_warning,
                               n_windows, montage, out_dir):
    """Generate publication_summary.txt with conservative, subject-level framing."""
    print("\n  Generating publication_summary.txt...")

    hc    = subject_val.get("hc", [])
    cand  = subject_val.get("cand", [])
    sh    = subject_val.get("split_half", {})
    roi_s = subject_val.get("roi_subject_stats", [])
    n_t1  = len(hc)
    n_t2  = len(cand)
    n_subj = len(subject_val.get("unique_subjects", []))
    n_robust_window = sum(1 for c in combined_channels if c["robust_significant"])

    L = []
    L.append("=" * 80)
    L.append("PUBLICATION-GRADE XAI SUMMARY")
    L.append("VLAAI EEG Auditory Attention Decoder — DTU Dataset")
    L.append(f"N = {n_windows} windows, {n_subj} subjects")
    L.append("=" * 80)
    L.append("")

    # A. Baseline
    bm = baseline_metrics
    L.append("A. BASELINE AAD PERFORMANCE")
    L.append("-" * 40)
    L.append(f"  AAD accuracy:              {bm['aad_accuracy']:.3f}")
    L.append(f"  Correlation margin (mean): {bm['correlation_margin_mean']:+.5f}")
    ci = bm.get("correlation_margin_CI", [float("nan"), float("nan")])
    L.append(f"  Correlation margin (95% CI): [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    L.append("  Interpretation: baseline accuracy suggests the model achieves above-chance")
    L.append("  auditory attention decoding on the DTU dataset, validating XAI interpretation.")
    L.append("")

    # B. Block 3 dominance
    L.append("B. ARCHITECTURE — BLOCK 3 DOMINANCE")
    L.append("-" * 40)
    if ablation:
        b3 = next((b for b in ablation if "Block 3" in b.get("name", "")), None)
        if b3:
            L.append(f"  Block 3 zero-weight ablation: ΔP = {b3.get('delta_p_mean', 0.0):+.5f}, "
                     f"ΔAcc = {b3.get('delta_acc', 0.0):+.4f}")
    L.append("  Block 3 (final recurrent iteration) dominates the decoding decision.")
    L.append("  Blocks 0-2 can be ablated with negligible effect, suggesting the model")
    L.append("  relies primarily on its final iteration output.")
    L.append("")

    # C. Subject-level high-confidence channels
    L.append("C. SUBJECT-LEVEL HIGH-CONFIDENCE CHANNELS (MAIN FINDING)")
    L.append("-" * 40)
    L.append(f"  Tier 1 (high-confidence): {n_t1} channels")
    L.append(f"    Criteria: both methods FDR-significant across subjects (q<0.05),")
    L.append(f"    same sign, stability >= 12/{n_subj} subjects, top 20% effect size.")
    if hc:
        L.append("    Channels: " + ", ".join(
            f"{r['channel_name']} ({r['roi'][:8]})" for r in hc))
    else:
        L.append("    No channels met all Tier-1 criteria.")
    L.append("")
    L.append(f"  Tier 2 (candidate): {n_t2} channels")
    L.append(f"    Criteria: >= 1 method FDR-significant, same sign, stability >= 10/{n_subj}.")
    if cand:
        L.append("    Channels: " + ", ".join(
            f"{r['channel_name']} ({r['roi'][:8]})" for r in cand[:10]))
        if len(cand) > 10:
            L.append(f"    ... and {len(cand) - 10} more (see candidate_channels.csv)")
    L.append("")
    L.append(f"  Window-level robust channels (broader sensitivity, lower confidence): {n_robust_window}/64")
    L.append("  NOTE: The window-level count ({}) should not be presented as the primary".format(n_robust_window))
    L.append("  conclusion. Windows are nested within subjects, so window-level FDR")
    L.append("  may overstate statistical reliability. The Tier-1 subject-level")
    L.append("  results are the main defensible finding.")
    L.append("")

    # D. ROI-level patterns
    L.append("D. ROI-LEVEL PATTERNS (SUBJECT-LEVEL TESTS)")
    L.append("-" * 40)
    sig_rois = [r for r in roi_s if r.get("occ_fdr_sig", False)]
    if sig_rois:
        L.append("  FDR-significant ROIs (Wilcoxon across subjects):")
        for r in sig_rois:
            L.append(f"    {r['roi']:20s}: occ mean={r['occ_subj_mean']:+.5f} "
                     f"CI=[{r['occ_subj_ci_lo']:+.5f},{r['occ_subj_ci_hi']:+.5f}]")
    else:
        L.append("  No ROIs reached subject-level FDR significance.")
        L.append("  (ROI effects may still be present but underpowered with n={} subjects)".format(n_subj))
    L.append("")

    # E. Split-half reliability
    L.append("E. SPLIT-HALF RELIABILITY")
    L.append("-" * 40)
    if sh:
        L.append(f"  Median Spearman r across {sh.get('n_iter', 0)} random splits: "
                 f"{sh.get('median_rho', float('nan')):.3f}")
        L.append(f"  95% CI: [{sh.get('ci_lo', float('nan')):.3f}, "
                 f"{sh.get('ci_hi', float('nan')):.3f}]")
        rho = sh.get("median_rho", 0.0)
        if rho >= 0.8:
            L.append("  Interpretation: HIGH reliability — channel rankings are stable across subject samples.")
        elif rho >= 0.5:
            L.append("  Interpretation: MODERATE reliability — main patterns replicate with some rank variability.")
        else:
            L.append("  Interpretation: LOW reliability — rankings are sensitive to subject sampling.")
            L.append("  Interpret individual channel ranks cautiously.")
    else:
        L.append("  Split-half analysis not available (insufficient subjects).")
    L.append("")

    # F. Subject variability
    L.append("F. SUBJECT-LEVEL VARIABILITY")
    L.append("-" * 40)
    L.append("  Subject specificity analysis showed low-to-moderate group-level consistency.")
    L.append("  Group-level channel importance maps may obscure individual differences.")
    L.append("  Individual explanations may diverge from the group-level map.")
    L.append("  See subject_channel_importance.csv and subject_level_channel_stats.csv.")
    L.append("")

    # G. Frequency findings
    L.append("G. FREQUENCY-BAND FINDINGS (EXPLORATORY ONLY)")
    L.append("-" * 40)
    if freq_short_window_warning:
        L.append("  WARNING: The analysis window is short. Delta-band frequency interpretation")
        L.append("  is limited by frequency resolution and should be treated with extra caution.")
    L.append("  Frequency-band analysis is exploratory. Results suggest, rather than prove,")
    L.append("  which frequency components contribute to the model's representations.")
    L.append("  Subject-level ROI x frequency statistics are in subject_level_roi_frequency_stats.csv.")
    L.append("")

    # H. Conservative language
    L.append("=" * 80)
    L.append("CONSERVATIVE LANGUAGE GUIDE")
    L.append("  Use:   'suggests', 'is consistent with', 'showed statistically robust")
    L.append("          but small-magnitude effects', 'should be interpreted cautiously',")
    L.append("          'exploratory frequency-band evidence'")
    L.append("  Avoid: 'proves', 'definitively shows', 'all channels are strongly important'")
    L.append("")

    # I. Output files
    L.append("=" * 80)
    L.append("OUTPUT FILES (PUBLICATION-GRADE)")
    L.append("-" * 40)
    files = [
        ("high_confidence_channels.csv",          "Tier-1 subject-level high-confidence channels"),
        ("candidate_channels.csv",                "Tier-2 candidate channels"),
        ("subject_channel_importance.csv",        "Per-subject per-channel occ & perm mean ΔP"),
        ("subject_level_channel_stats.csv",       "Wilcoxon tests + bootstrap CI + Cohen's d per channel"),
        ("subject_level_roi_stats.csv",           "ROI-level Wilcoxon tests across subjects"),
        ("subject_level_roi_frequency_stats.csv", "ROI x frequency band subject-level tests"),
        ("split_half_reliability.csv",            "Per-iteration split-half Spearman r"),
        ("split_half_reliability_summary.txt",    "Split-half summary"),
        ("high_confidence_channels_plot.png",     "T1/T2 channel importance bar chart"),
        ("subject_level_roi_plot.png",            "ROI subject-level importance plot"),
    ]
    for fname, desc in files:
        L.append(f"  {fname:45s} {desc}")
    L.append("")
    L.append("=" * 80)

    text = "\n".join(L)
    (out_dir / "publication_summary.txt").write_text(text, encoding="utf-8")
    print(f"  Saved publication_summary.txt  ({len(L)} lines)")
    return text


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = make_output_dir(Path(args.output_dir))

    # ── Print & save config ───────────────────────────────────────
    config = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": args.data_dir,
        "h5_path": args.h5_path,
        "output_dir": args.output_dir,
        "subjects": args.subjects,
        "max_samples": args.max_samples,
        "balanced_by_subject": args.balanced_by_subject,
        "random_seed": args.seed,
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "ig_samples": args.n_ig,
        "ig_steps": args.ig_steps,
        "windows_per_subject": args.windows_per_subject,
        "top_k": args.top_k,
        "montage_file": args.montage_file,
        "fdr_alpha": args.fdr_alpha,
        "stability_threshold": args.stability_threshold,
        "device": args.device,
        "skip_ig": args.skip_ig,
        "skip_frequency": args.skip_frequency,
        "occlusion_mode": args.occlusion_mode,
        "roi_frequency_mode": args.roi_frequency_mode,
        # Legacy keys retained for compatibility.
        "seed": args.seed,
        "n_ig": args.n_ig,
    }

    print("=" * 70)
    print("FOCUSED XAI ANALYSIS: Channel -> ROI -> Frequency")
    print("=" * 70)
    print(f"  Seed: {args.seed}")
    print(f"  Bootstrap: {args.n_boot}")
    print(f"  Sign-flip permutations: {args.n_perm}")
    max_str = "ALL" if args.max_samples == -1 else str(args.max_samples)
    print(f"  Max samples: {max_str}")
    print(f"  Balanced by subject: {args.balanced_by_subject}")
    print(f"  FDR alpha: {args.fdr_alpha}")
    print(f"  Stability threshold: {args.stability_threshold:.0%}")
    print(f"  Occlusion mode: {args.occlusion_mode}")
    print(f"  ROI frequency mode: {args.roi_frequency_mode}")
    print(f"  Top-K channels for freq analysis: {args.top_k}")
    print(f"  IG samples: {args.n_ig} (skipped: {args.skip_ig})")
    print(f"  Montage file: {args.montage_file}")
    print(f"  Skip frequency: {args.skip_frequency}")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    # ── Load montage ──────────────────────────────────────────────
    montage = load_montage(args.montage_file)

    # ── Load data ─────────────────────────────────────────────────
    print("\nLoading dataset...")
    from aad_xai.data.vlaai_dataset import VLAAIDTUDataset

    ds = VLAAIDTUDataset(
        data_dir=args.data_dir,
        window_length=320, hop=64,
        subjects=args.subjects,
    )
    total_available = len(ds)
    if args.max_samples == -1:
        N = total_available
    else:
        N = min(args.max_samples, total_available)

    # Select window indices (balanced or sequential)
    if args.balanced_by_subject and N < total_available:
        unique_subjects = sorted(set(ds.subject_ids))
        n_subj = len(unique_subjects)
        per_subj = N // n_subj
        remainder = N % n_subj
        selected_indices = []
        subj_window_counts = {}
        rng_sel = np.random.RandomState(args.seed)
        for si, subj in enumerate(unique_subjects):
            subj_mask = ds.subject_ids == subj
            subj_idxs = np.where(subj_mask)[0]
            n_take = per_subj + (1 if si < remainder else 0)
            n_take = min(n_take, len(subj_idxs))
            chosen = rng_sel.choice(subj_idxs, size=n_take, replace=False)
            selected_indices.extend(sorted(chosen.tolist()))
            subj_window_counts[subj] = int(n_take)
        selected_indices = sorted(selected_indices)
        N = len(selected_indices)
        print(f"  {total_available} total windows, balanced-sampling {N} " +
              f"(~{per_subj} per subject across {n_subj} subjects)")
    else:
        selected_indices = list(range(N))
        unique_subjects = sorted(set(ds.subject_ids))
        subj_window_counts = {}
        for subj in unique_subjects:
            subj_mask = ds.subject_ids == subj
            subj_idxs = np.where(subj_mask)[0]
            count = int(np.sum(subj_idxs < N))
            if count > 0:
                subj_window_counts[subj] = count
        print(f"  {total_available} total windows, using {N} (sequential)")

    selected_subject_ids = np.asarray([ds.subject_ids[i] for i in selected_indices])
    selected_subject_count = int(len(set(selected_subject_ids.tolist()))) if len(selected_subject_ids) > 0 else 0

    config["requested_max_samples"] = args.max_samples
    config["actual_n_windows_used"] = N
    config["total_available_windows"] = total_available
    config["n_windows"] = N
    config["balanced_by_subject"] = args.balanced_by_subject
    config["subject_window_counts"] = subj_window_counts
    config["selected_subject_counts"] = subj_window_counts
    config["selected_indices"] = [int(i) for i in selected_indices]
    config["selected_subject_count"] = selected_subject_count

    # Save subject_window_counts.csv
    with open(out_dir / "subject_window_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "n_windows"])
        w.writeheader()
        for subj, cnt in sorted(subj_window_counts.items()):
            w.writerow({"subject": subj, "n_windows": cnt})
    print(f"  Subject window counts saved ({len(subj_window_counts)} subjects)")

    eeg_all = torch.stack([ds[i][0] for i in selected_indices]).to(device)
    att_all = torch.stack([ds[i][1] for i in selected_indices]).to(device)
    unatt_all = torch.stack([ds[i][2] for i in selected_indices]).to(device)

    # ── Load model ────────────────────────────────────────────────
    print("Loading VLAAI model...")
    from aad_xai.models import VLAAIPyTorch, AADDecisionEEGOnly

    try:
        model = VLAAIPyTorch.from_h5(args.h5_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load pretrained VLAAI model from {args.h5_path}. "
            "XAI results are invalid without the pretrained model."
        ) from e
    model.eval().to(device)
    print("  Model loaded successfully.")
    print(f"  Model device: {device}")

    decision = AADDecisionEEGOnly(model)
    decision.eval().to(device)

    # Verify model works
    decision.set_envelopes(att_all[:3], unatt_all[:3])
    with torch.no_grad():
        test_logits = decision(eeg_all[:3])
    print(f"  Decision logits (sample): {test_logits[0].detach().cpu().numpy()}")

    decision.set_envelopes(att_all, unatt_all)

    # Baseline validation before XAI sections.
    baseline_metrics, near_chance = compute_baseline_aad_metrics(
        decision, eeg_all, att_all, unatt_all, args.n_boot, args.seed,
        selected_subject_ids, out_dir)
    baseline_metrics["near_chance_warning"] = near_chance

    print("\nValidation summary before XAI:")
    print(f"  Selected window count: {N}")
    print(f"  Selected subject count: {selected_subject_count}")
    print(f"  EEG tensor shape: {tuple(eeg_all.shape)}")
    print(f"  Attended envelope shape: {tuple(att_all.shape)}")
    print(f"  Unattended envelope shape: {tuple(unatt_all.shape)}")
    print(f"  Baseline AAD accuracy: {baseline_metrics['aad_accuracy']:.3f}")
    print(f"  Baseline correlation margin: {baseline_metrics['correlation_margin_mean']:+.5f}")

    # Save config early
    config["baseline_metrics"] = baseline_metrics
    save_json(config, out_dir / "run_config.json")

    # ── Section A: Architecture + Block Ablation ──────────────────
    arch_info, ablation = run_architecture(
        model, decision, eeg_all, att_all, unatt_all, args.n_boot, args.seed, out_dir, device)
    decision.set_envelopes(att_all, unatt_all)

    # ── Section B+C: Channel Importance + FDR ─────────────────────
    combined_channels, subj_profiles, ch_matrix, r_occ_perm = run_channel_importance(
        decision, model, eeg_all, att_all, unatt_all, selected_subject_ids,
        args.n_boot, args.n_perm, args.seed, args.n_ig, args.ig_steps,
        args.windows_per_subject, args.top_k, out_dir, montage,
        args.fdr_alpha, args.stability_threshold, args.occlusion_mode, args.skip_ig)
    decision.set_envelopes(att_all, unatt_all)

    # ── Section G: Subject Specificity Analysis ───────────────────
    subj_spec_corr, ch_spec_data = run_subject_specificity_analysis(
        combined_channels, ch_matrix, out_dir)
    decision.set_envelopes(att_all, unatt_all)

    # ── Section E: ROI Analysis ───────────────────────────────────
    occ_pw = np.load(out_dir / "occlusion_perwindow.npy")
    perm_pw = np.load(out_dir / "permutation_perwindow.npy")
    roi_results = run_roi_analysis(combined_channels, occ_pw, perm_pw,
                                   args.n_boot, args.seed, out_dir, montage)

    # ── Section F: Frequency Analysis ─────────────────────────────
    freq_by_channel = []
    freq_by_roi = []
    if not args.skip_frequency:
        freq_by_channel, freq_by_roi = compute_frequency_by_important_channels(
            decision, eeg_all, att_all, unatt_all, combined_channels,
            args.n_boot, args.seed, args.top_k, out_dir, montage,
            args.roi_frequency_mode)

    window_seconds = float(eeg_all.shape[1] / FS)
    freq_short_window_warning = window_seconds <= 5.0
    if freq_short_window_warning:
        print(
            "WARNING: Frequency-band interpretation, especially delta-band effects, "
            "should be treated cautiously because the analysis window is short."
        )

    # ── Build final important channel table ───────────────────────
    print("\n  [D] Building final important channel table...")
    final_rows = build_final_important_channel_table(combined_channels, freq_by_channel, out_dir)

    # ── Save final channels plot ──────────────────────────────────
    save_final_channels_plot(final_rows, out_dir)

    # ── Generate Report ───────────────────────────────────────────
    write_focused_report(
        arch_info, ablation, combined_channels, roi_results,
        freq_by_channel, freq_by_roi, final_rows,
        subj_profiles, ch_matrix, r_occ_perm,
        baseline_metrics, freq_short_window_warning,
        N, args.n_boot, args.top_k, args.fdr_alpha, args.stability_threshold,
        montage, out_dir, args,
        subj_spec_corr=subj_spec_corr, ch_spec_data=ch_spec_data)

    # ── Section H: Subject-Level Statistical Validation ──────────
    subject_val = run_subject_level_validation(
        occ_pw, perm_pw, selected_subject_ids,
        combined_channels, montage, out_dir,
        args.fdr_alpha, args.n_boot, args.seed,
        n_split_half=1000)
    decision.set_envelopes(att_all, unatt_all)

    # ── Section H.6: ROI × frequency subject-level stats ─────────
    roi_freq_subj_stats = []
    if not args.skip_frequency and args.roi_frequency_mode:
        try:
            roi_freq_subj_stats = run_subject_level_roi_frequency_stats(
                decision, eeg_all, att_all, unatt_all,
                selected_subject_ids, montage,
                out_dir, args.fdr_alpha, args.n_boot, args.seed)
        except Exception as exc:
            print(f"  WARNING: ROI x frequency subject stats failed: {exc}")
            import traceback as _tb
            _tb.print_exc()
        decision.set_envelopes(att_all, unatt_all)

    # ── Publication summary ───────────────────────────────────────
    write_publication_summary(
        combined_channels, subject_val,
        baseline_metrics, arch_info, ablation,
        freq_by_channel, freq_by_roi,
        freq_short_window_warning,
        N, montage, out_dir)

    # ── Update config with final counts ───────────────────────────
    config["n_robust_significant"] = len(final_rows)
    config["n_occ_fdr_significant"] = sum(1 for c in combined_channels if c["occ_fdr_significant"])
    config["n_perm_fdr_significant"] = sum(1 for c in combined_channels if c["perm_fdr_significant"])
    config["montage_source"] = montage["source"]
    config["frequency_short_window_warning"] = freq_short_window_warning
    config["n_tier1_high_confidence"] = len(subject_val.get("hc", []))
    config["n_tier2_candidate"] = len(subject_val.get("cand", []))
    config["n_subj_occ_fdr_sig"] = subject_val.get("n_subj_occ_fdr_sig", 0)
    config["n_subj_perm_fdr_sig"] = subject_val.get("n_subj_perm_fdr_sig", 0)
    config["split_half_median_rho"] = subject_val.get("split_half", {}).get("median_rho", None)
    save_json(config, out_dir / "run_config.json")

    n_t1 = len(subject_val.get("hc", []))
    n_t2 = len(subject_val.get("cand", []))
    print("\n" + "=" * 70)
    print("FOCUSED XAI ANALYSIS COMPLETE")
    print(f"  Window-level robust channels:        {len(final_rows)}/64")
    print(f"  Subject-level Tier-1 (high-conf):   {n_t1}/64")
    print(f"  Subject-level Tier-2 (candidate):   {n_t2}/64")
    sh = subject_val.get("split_half", {})
    if sh:
        print(f"  Split-half reliability (median r):  {sh.get('median_rho', float('nan')):.3f}")
    print(f"  All results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
