"""Comprehensive XAI analysis for pretrained VLAAI auditory attention decoder.

Sections implemented (A–L):
  A. Architecture inspection & gradient flow documentation
  B. Robust block ablation (4 modes: zero-weight, zero-activation, mean, permute)
  C. Channel / ROI occlusion with bootstrap CIs
  D. Temporal occlusion with CIs and significance tests
  E. Layer probing with proper CV (subject-level splits, standardisation inside folds)
  F. GradCAM + Integrated Gradients with normalised layer comparison
  G. Enhanced sanity checks (Spearman, Pearson, SSIM, top-k overlap)
  H. Subject-wise consistency with clustering / dendrogram
  I. Frequency-band analysis with Butterworth filtering + edge-artifact handling
  J. Correct vs incorrect using true labels (correlation-based)
  K. Clean output folder structure
  L. CLI arguments, fixed seeds, shape safeguards

Usage:
    python scripts/run_xai_comprehensive.py
    python scripts/run_xai_comprehensive.py --max-samples 500 --seed 42 --n-boot 500
    python scripts/run_xai_comprehensive.py --sections B C D --subjects S1 S2
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

ALL_SECTIONS = list("ABCDEFGHIJ")


def parse_args():
    p = argparse.ArgumentParser(description="VLAAI Comprehensive XAI Analysis")
    p.add_argument("--data-dir", type=str,
                    default=str(ROOT / "external" / "vlaai" / "evaluation_datasets" / "DTU"))
    p.add_argument("--h5-path", type=str,
                    default=str(ROOT / "external" / "vlaai" / "pretrained_models" / "vlaai.h5"))
    p.add_argument("--output-dir", type=str, default=str(ROOT / "xai_results"))
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=1000, help="Bootstrap iterations")
    p.add_argument("--windows-per-subject", type=int, default=50,
                    help="Max windows per subject in Section H (default 50).")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--sections", nargs="*", default=ALL_SECTIONS,
                    help="Sections to run (A-J). Default: all.")
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
# Output directory helpers (Section K)
# ══════════════════════════════════════════════════════════════════════

def make_output_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "root": base,
        "architecture": base / "A_architecture",
        "ablation": base / "B_ablation",
        "channel_occlusion": base / "C_channel_occlusion",
        "temporal_occlusion": base / "D_temporal_occlusion",
        "probing": base / "E_probing",
        "attribution": base / "F_attribution",
        "sanity": base / "G_sanity_checks",
        "subject_wise": base / "H_subject_wise",
        "frequency_band": base / "I_frequency_band",
        "correct_incorrect": base / "J_correct_incorrect",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════

FS = 64  # Hz
ROIS = OrderedDict([
    ("Frontal",        list(range(0, 12))),
    ("Fronto-Central", list(range(12, 18))),
    ("Central",        list(range(18, 30))),
    ("Temporal",       list(range(30, 42))),
    ("Parietal",       list(range(42, 54))),
    ("Occipital",      list(range(54, 64))),
])


# ══════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════

def get_attended_prob(decision, eeg, att, unatt):
    """P(attended) for a batch — safe envelope setting + softmax."""
    decision.set_envelopes(att, unatt)
    with torch.no_grad():
        logits = decision(eeg)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95,
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


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


# ══════════════════════════════════════════════════════════════════════
# SECTION A — Architecture inspection
# ══════════════════════════════════════════════════════════════════════

def section_a(model, dirs, device):
    print("\n" + "=" * 70)
    print("SECTION A: ARCHITECTURE INSPECTION & GRADIENT FLOW")
    print("=" * 70)
    out = dirs["architecture"]

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Per-component breakdown
    components = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        components[name] = n

    # Shared vs unique
    shared_names = {"extractor", "output_context", "final_dense"}
    shared_params = sum(v for k, v in components.items() if k in shared_names)
    unique_params = sum(v for k, v in components.items() if k not in shared_names)

    lines = [
        "VLAAI Architecture Summary",
        "=" * 50,
        f"Total parameters: {total_params:,}",
        f"Trainable:        {trainable:,}",
        "",
        "Component breakdown:",
    ]
    for name, n in components.items():
        pct = n / total_params * 100
        lines.append(f"  {name:25s}: {n:>10,} ({pct:.1f}%)")
    lines.extend([
        "",
        f"Shared parameters (extractor + output_context + final_dense): {shared_params:,} ({shared_params/total_params*100:.1f}%)",
        f"Unique parameters (block_denses ×4):                          {unique_params:,} ({unique_params/total_params*100:.1f}%)",
        "",
        "Forward pass (4 iterations):",
        "  x = zeros(B, C, T)",
        "  for i in 0..3:",
        "    ext_out = extractor(eeg + x)       # shared 5-layer conv stack",
        "    x = block_denses[i](ext_out)        # unique per-block Linear 128→64",
        "    x = output_context(x)               # shared causal conv 64→64, k=32",
        "  out = final_dense(x)                  # Linear 64→1",
        "",
        "Key insight: x is OVERWRITTEN each iteration (not accumulated).",
        "Each block sees eeg + previous_block_output as input.",
        "",
        "Gradient flow:",
        "  Block 3 → shortest path to loss (3 modules: final_dense ← output_context ← block_denses[3])",
        "  Block 0 → longest path (~24 nonlinearities through 3 additional extractor+dense+context passes)",
        "  Additive skip (eeg_cf + x) provides gradient highway at each iteration.",
    ])

    # Verify gradient flow empirically
    model.train()
    x_test = torch.randn(2, 320, 64, device=device, requires_grad=False)
    x_test.requires_grad_(True)
    out = model(x_test)
    loss = out.sum()
    loss.backward()

    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = float(p.grad.norm().item())
    model.eval()

    lines.extend([
        "",
        "Empirical gradient norms (sum loss):",
    ])
    for name in sorted(grad_norms, key=grad_norms.get, reverse=True)[:15]:
        lines.append(f"  {name:50s}: {grad_norms[name]:.6f}")

    report = "\n".join(lines)
    (dirs["architecture"] / "architecture_report.txt").write_text(report, encoding="utf-8")
    save_json({"params": components, "total": total_params, "shared_pct": shared_params / total_params,
               "grad_norms": grad_norms}, dirs["architecture"] / "architecture.json")
    print(report)
    print(f"\n  Saved to {dirs['architecture']}")


# ══════════════════════════════════════════════════════════════════════
# SECTION B — Robust block ablation
# ══════════════════════════════════════════════════════════════════════

def section_b(model, decision, eeg, att, unatt, n_boot, seed, dirs, device):
    print("\n" + "=" * 70)
    print("SECTION B: ROBUST BLOCK ABLATION (4 MODES)")
    print("=" * 70)
    out = dirs["ablation"]
    rng = np.random.RandomState(seed)

    base_probs = get_attended_prob(decision, eeg, att, unatt)
    base_mean, base_lo, base_hi = bootstrap_ci(base_probs, n_boot, seed=seed)

    # Compute accuracy + AUC baseline
    base_correct = (base_probs > 0.5).astype(float)
    base_acc = base_correct.mean()

    results = {"baseline": {"mean_p": base_mean, "ci_lo": base_lo, "ci_hi": base_hi,
                             "accuracy": float(base_acc)}}
    N = eeg.shape[0]

    modes = ["zero_weights", "zero_activations", "mean_replace", "permute"]

    for bi in range(4):
        results[f"block_{bi}"] = {}
        for mode in modes:
            # Save originals
            orig_w = model.block_denses[bi].weight.data.clone()
            orig_b = model.block_denses[bi].bias.data.clone()

            if mode == "zero_weights":
                model.block_denses[bi].weight.data.zero_()
                model.block_denses[bi].bias.data.zero_()
                abl_probs = get_attended_prob(decision, eeg, att, unatt)

            elif mode == "zero_activations":
                # Hook to zero output of block_denses[bi]
                handle = model.block_denses[bi].register_forward_hook(
                    lambda mod, inp, out: torch.zeros_like(out))
                abl_probs = get_attended_prob(decision, eeg, att, unatt)
                handle.remove()

            elif mode == "mean_replace":
                # Replace output with dataset mean activation
                activations = []
                def capture_hook(mod, inp, out):
                    activations.append(out.detach())
                handle = model.block_denses[bi].register_forward_hook(capture_hook)
                _ = get_attended_prob(decision, eeg, att, unatt)
                handle.remove()
                mean_act = activations[0].mean(dim=0, keepdim=True)
                handle = model.block_denses[bi].register_forward_hook(
                    lambda mod, inp, out, _m=mean_act: _m.expand_as(out))
                abl_probs = get_attended_prob(decision, eeg, att, unatt)
                handle.remove()

            elif mode == "permute":
                # Permute activations across batch
                def perm_hook(mod, inp, out, _rng=rng):
                    perm = torch.from_numpy(_rng.permutation(out.shape[0])).long()
                    return out[perm]
                handle = model.block_denses[bi].register_forward_hook(perm_hook)
                abl_probs = get_attended_prob(decision, eeg, att, unatt)
                handle.remove()

            # Restore weights (only needed for zero_weights mode)
            model.block_denses[bi].weight.data = orig_w
            model.block_denses[bi].bias.data = orig_b

            delta_p = base_probs - abl_probs
            mean_dp, lo_dp, hi_dp = bootstrap_ci(delta_p, n_boot, seed=seed)
            abl_acc = (abl_probs > 0.5).mean()

            results[f"block_{bi}"][mode] = {
                "delta_p_mean": mean_dp, "delta_p_ci_lo": lo_dp, "delta_p_ci_hi": hi_dp,
                "ablated_acc": float(abl_acc), "delta_acc": float(base_acc - abl_acc),
            }
            print(f"  Block {bi} [{mode:16s}]: ΔP = {mean_dp:+.5f} [{lo_dp:+.5f}, {hi_dp:+.5f}], "
                  f"ΔAcc = {base_acc - abl_acc:+.3f}")

    save_json(results, out / "block_ablation.json")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_pos = np.arange(4)
    width = 0.18
    colors = {"zero_weights": "#d32f2f", "zero_activations": "#1976d2",
              "mean_replace": "#388e3c", "permute": "#f57c00"}

    for mi, mode in enumerate(modes):
        dps = [results[f"block_{bi}"][mode]["delta_p_mean"] for bi in range(4)]
        lo = [results[f"block_{bi}"][mode]["delta_p_ci_lo"] for bi in range(4)]
        hi = [results[f"block_{bi}"][mode]["delta_p_ci_hi"] for bi in range(4)]
        errs = [[d - l for d, l in zip(dps, lo)], [h - d for d, h in zip(dps, hi)]]
        axes[0].bar(x_pos + mi * width, dps, width, yerr=errs, label=mode,
                    color=colors[mode], alpha=0.85, capsize=3)

    axes[0].set_xticks(x_pos + 1.5 * width)
    axes[0].set_xticklabels([f"Block {i}" for i in range(4)])
    axes[0].set_ylabel("ΔP(attended)")
    axes[0].set_title("Block Ablation: ΔP by Mode")
    axes[0].legend(fontsize=8)
    axes[0].axhline(0, color="k", linewidth=0.5)

    # Accuracy drop
    for mi, mode in enumerate(modes):
        daccs = [results[f"block_{bi}"][mode]["delta_acc"] for bi in range(4)]
        axes[1].bar(x_pos + mi * width, daccs, width, label=mode,
                    color=colors[mode], alpha=0.85)
    axes[1].set_xticks(x_pos + 1.5 * width)
    axes[1].set_xticklabels([f"Block {i}" for i in range(4)])
    axes[1].set_ylabel("ΔAccuracy")
    axes[1].set_title("Block Ablation: Accuracy Drop")
    axes[1].legend(fontsize=8)
    axes[1].axhline(0, color="k", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out / "block_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# SECTION C — Channel / ROI occlusion with bootstrap CIs
# ══════════════════════════════════════════════════════════════════════

def section_c(decision, eeg, att, unatt, n_boot, seed, dirs, device):
    print("\n" + "=" * 70)
    print("SECTION C: CHANNEL / ROI OCCLUSION WITH BOOTSTRAP CIs")
    print("=" * 70)
    out = dirs["channel_occlusion"]
    N = eeg.shape[0]

    base_probs = get_attended_prob(decision, eeg, att, unatt)

    # Per-window, per-channel drop
    channel_drops_pw = np.zeros((N, 64))  # per-window
    for ch in range(64):
        eeg_m = eeg.clone()
        eeg_m[:, :, ch] = 0.0
        m_probs = get_attended_prob(decision, eeg_m, att, unatt)
        channel_drops_pw[:, ch] = base_probs - m_probs

    # Bootstrap CI per channel
    ch_results = []
    for ch in range(64):
        mean, lo, hi = bootstrap_ci(channel_drops_pw[:, ch], n_boot, seed=seed)
        ch_results.append({"channel": ch, "mean_dp": mean, "ci_lo": lo, "ci_hi": hi})

    # ROI aggregation
    roi_results = []
    for roi_name, chs in ROIS.items():
        roi_drops = channel_drops_pw[:, chs].mean(axis=1)  # per window
        mean, lo, hi = bootstrap_ci(roi_drops, n_boot, seed=seed)
        roi_results.append({"roi": roi_name, "mean_dp": mean, "ci_lo": lo, "ci_hi": hi,
                            "channels": chs})

    save_json({"channels": ch_results, "rois": roi_results}, out / "channel_occlusion.json")
    np.save(out / "channel_drops_perwindow.npy", channel_drops_pw)

    # CSV
    import csv
    with open(out / "channel_occlusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "mean_dp", "ci_lo", "ci_hi"])
        w.writeheader()
        w.writerows(ch_results)

    # Print top 15
    sorted_ch = sorted(ch_results, key=lambda x: abs(x["mean_dp"]), reverse=True)
    print("  Top-15 channels by |ΔP|:")
    for r in sorted_ch[:15]:
        print(f"    Ch {r['channel']:2d}: {r['mean_dp']:+.5f} [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]")

    print("\n  ROI summary:")
    for r in roi_results:
        print(f"    {r['roi']:20s}: ΔP = {r['mean_dp']:+.5f} [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Full channel bar chart
    ax = axes[0]
    ch_means = np.array([r["mean_dp"] for r in ch_results])
    colors = ["#d32f2f" if d > 0 else "#1976d2" for d in ch_means]
    ax.bar(range(64), ch_means, color=colors, alpha=0.8)
    ch_lo = np.array([r["ci_lo"] for r in ch_results])
    ch_hi = np.array([r["ci_hi"] for r in ch_results])
    ax.fill_between(range(64), ch_lo, ch_hi, alpha=0.15, color="gray")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Channel")
    ax.set_ylabel("ΔP(attended)")
    ax.set_title("Channel Occlusion with 95% CI")

    # ROI bar chart
    ax = axes[1]
    roi_names = [r["roi"] for r in roi_results]
    roi_means = [r["mean_dp"] for r in roi_results]
    roi_errs = [[r["mean_dp"] - r["ci_lo"] for r in roi_results],
                [r["ci_hi"] - r["mean_dp"] for r in roi_results]]
    ax.bar(range(len(roi_names)), [abs(m) for m in roi_means], yerr=roi_errs,
           color="#6a1b9a", alpha=0.7, capsize=5)
    ax.set_xticks(range(len(roi_names)))
    ax.set_xticklabels(roi_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|ΔP(attended)| ± CI")
    ax.set_title("ROI Occlusion Importance")

    # Top-20 channel sorted
    ax = axes[2]
    top20 = sorted_ch[:20]
    ax.barh(range(20), [r["mean_dp"] for r in top20],
            xerr=[[r["mean_dp"] - r["ci_lo"] for r in top20],
                  [r["ci_hi"] - r["mean_dp"] for r in top20]],
            color=["#d32f2f" if r["mean_dp"] > 0 else "#1976d2" for r in top20],
            alpha=0.8, capsize=3)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"Ch {r['channel']}" for r in top20], fontsize=8)
    ax.set_xlabel("ΔP(attended)")
    ax.set_title("Top-20 Channels (sorted)")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out / "channel_occlusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    return channel_drops_pw


# ══════════════════════════════════════════════════════════════════════
# SECTION D — Temporal occlusion with CIs + significance tests
# ══════════════════════════════════════════════════════════════════════

def section_d(decision, eeg, att, unatt, n_boot, seed, dirs, device):
    print("\n" + "=" * 70)
    print("SECTION D: TEMPORAL OCCLUSION WITH CIs + SIGNIFICANCE TESTS")
    print("=" * 70)
    out = dirs["temporal_occlusion"]
    N = eeg.shape[0]

    base_probs = get_attended_prob(decision, eeg, att, unatt)

    win_t, hop_t = 64, 32  # 1s window, 0.5s hop
    T = eeg.shape[1]
    starts = list(range(0, T - win_t + 1, hop_t))
    n_starts = len(starts)

    # Per-window temporal drops
    temporal_pw = np.zeros((N, n_starts))
    for si, s in enumerate(starts):
        eeg_m = eeg.clone()
        eeg_m[:, s:s + win_t, :] = 0.0
        m_probs = get_attended_prob(decision, eeg_m, att, unatt)
        temporal_pw[:, si] = base_probs - m_probs

    # Bootstrap CIs
    t_results = []
    for si, s in enumerate(starts):
        mean, lo, hi = bootstrap_ci(temporal_pw[:, si], n_boot, seed=seed)
        t_results.append({
            "start_sample": int(s), "start_sec": s / FS,
            "end_sec": (s + win_t) / FS,
            "mean_dp": mean, "ci_lo": lo, "ci_hi": hi,
        })

    # Significance test: early vs late half
    from scipy.stats import mannwhitneyu
    mid = n_starts // 2
    early_drops = temporal_pw[:, :mid].mean(axis=1)
    late_drops = temporal_pw[:, mid:].mean(axis=1)
    try:
        stat, pval = mannwhitneyu(early_drops, late_drops, alternative="two-sided")
    except Exception:
        stat, pval = 0.0, 1.0

    sig_test = {
        "early_mean": float(early_drops.mean()),
        "late_mean": float(late_drops.mean()),
        "mann_whitney_U": float(stat),
        "p_value": float(pval),
        "significant_at_0.05": pval < 0.05,
    }

    save_json({"temporal": t_results, "early_vs_late": sig_test},
              out / "temporal_occlusion.json")
    np.save(out / "temporal_drops_perwindow.npy", temporal_pw)

    print("  Temporal sensitivity (1s mask, 0.5s hop):")
    sorted_t = sorted(t_results, key=lambda x: abs(x["mean_dp"]), reverse=True)
    for r in sorted_t[:8]:
        print(f"    t={r['start_sec']:.1f}-{r['end_sec']:.1f}s: "
              f"ΔP = {r['mean_dp']:+.5f} [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]")
    print(f"\n  Early vs Late: early={sig_test['early_mean']:+.5f}, "
          f"late={sig_test['late_mean']:+.5f}, p={sig_test['p_value']:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    t_secs = [r["start_sec"] for r in t_results]
    means = [r["mean_dp"] for r in t_results]
    lo = [r["ci_lo"] for r in t_results]
    hi = [r["ci_hi"] for r in t_results]
    ax.fill_between(t_secs, lo, hi, alpha=0.2, color="#1976d2")
    ax.plot(t_secs, means, color="#1976d2", linewidth=2)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔP(attended)")
    ax.set_title("Temporal Occlusion with 95% CI")

    ax = axes[1]
    ax.boxplot([early_drops, late_drops], tick_labels=["Early half", "Late half"])
    ax.set_ylabel("Mean ΔP per window")
    ax.set_title(f"Early vs Late (p={pval:.4f})")

    plt.tight_layout()
    plt.savefig(out / "temporal_occlusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# SECTION E — Layer probing with proper CV
# ══════════════════════════════════════════════════════════════════════

def section_e(model, eeg, att, labels, subject_ids, n_boot, seed, dirs, device):
    print("\n" + "=" * 70)
    print("SECTION E: LAYER PROBING WITH SUBJECT-LEVEL CV")
    print("=" * 70)
    out = dirs["probing"]

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from aad_xai.xai.probes_vlaai import extract_all_activations_pt

    # Extract activations
    print("  Extracting activations from all layers...")
    activations = extract_all_activations_pt(model, eeg, recursive=True)

    # Filter to important layers only (conv, ln, dense)
    key_layers = [k for k in activations if any(t in k for t in ["conv", "ln", "block_denses", "output_context", "final_dense"])]
    if not key_layers:
        key_layers = list(activations.keys())

    print(f"  {len(key_layers)} layers selected for probing")

    # Subject-level leave-one-subject-out CV
    unique_subj = np.unique(subject_ids)
    n_folds = min(len(unique_subj), 5)  # cap folds

    if n_folds < 2:
        print("  WARNING: Need ≥2 subjects for CV. Falling back to random split.")
        n_folds = 0

    results = {}
    for layer in key_layers:
        act = activations[layer]
        if act.shape[0] != len(labels):
            continue

        if n_folds >= 2:
            # Subject-level K-fold
            fold_accs, fold_aucs = [], []
            for fold_i in range(n_folds):
                test_subj = unique_subj[fold_i::n_folds]
                test_mask = np.isin(subject_ids, test_subj)
                train_mask = ~test_mask

                if train_mask.sum() < 5 or test_mask.sum() < 5:
                    continue
                if len(np.unique(labels[train_mask])) < 2:
                    continue

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(act[train_mask])
                X_te = scaler.transform(act[test_mask])
                y_tr = labels[train_mask]
                y_te = labels[test_mask]

                clf = LogisticRegression(max_iter=2000, random_state=seed)
                clf.fit(X_tr, y_tr)
                pred = clf.predict(X_te)
                fold_accs.append(accuracy_score(y_te, pred))
                try:
                    proba = clf.predict_proba(X_te)[:, 1]
                    fold_aucs.append(roc_auc_score(y_te, proba))
                except Exception:
                    fold_aucs.append(float("nan"))

            if fold_accs:
                results[layer] = {
                    "accuracy_mean": float(np.mean(fold_accs)),
                    "accuracy_std": float(np.std(fold_accs)),
                    "auc_mean": float(np.nanmean(fold_aucs)),
                    "auc_std": float(np.nanstd(fold_aucs)),
                    "n_folds": len(fold_accs),
                }
        else:
            # Fallback: random split with standardisation
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(
                act, labels, test_size=0.2, random_state=seed, stratify=labels)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = LogisticRegression(max_iter=2000, random_state=seed)
            clf.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, clf.predict(X_te))
            try:
                auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
            except Exception:
                auc = float("nan")
            results[layer] = {"accuracy_mean": acc, "auc_mean": auc, "n_folds": 1}

    # Sort by accuracy
    sorted_layers = sorted(results.items(), key=lambda x: x[1]["accuracy_mean"], reverse=True)
    print("\n  Probe accuracy by layer (top 15):")
    for name, r in sorted_layers[:15]:
        std_str = f" ± {r.get('accuracy_std', 0):.3f}" if "accuracy_std" in r else ""
        auc_str = f", AUC={r['auc_mean']:.3f}" if not np.isnan(r.get("auc_mean", float("nan"))) else ""
        print(f"    {name:50s}: {r['accuracy_mean']:.3f}{std_str}{auc_str}")

    save_json(results, out / "probe_results.json")

    # Plot
    if sorted_layers:
        fig, ax = plt.subplots(1, 1, figsize=(10, max(4, len(sorted_layers[:20]) * 0.35)))
        names = [n.replace("decoder.", "") for n, _ in sorted_layers[:20]]
        accs = [r["accuracy_mean"] for _, r in sorted_layers[:20]]
        stds = [r.get("accuracy_std", 0) for _, r in sorted_layers[:20]]
        colors = ["#2e7d32" if a > 0.55 else "#f57f17" if a > 0.5 else "#d32f2f" for a in accs]
        ax.barh(range(len(names)), accs, xerr=stds, color=colors, alpha=0.8, capsize=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.axvline(0.5, color="k", linestyle="--", linewidth=1, label="chance")
        ax.set_xlabel("Accuracy ± std")
        ax.set_title("Attention Probes (subject-level CV)")
        ax.invert_yaxis()
        ax.legend()
        plt.tight_layout()
        plt.savefig(out / "probes.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# SECTION F — GradCAM + Integrated Gradients
# ══════════════════════════════════════════════════════════════════════

def section_f(decision, model, eeg, att, unatt, dirs, device):
    print("\n" + "=" * 70)
    print("SECTION F: GRADCAM + INTEGRATED GRADIENTS")
    print("=" * 70)
    out = dirs["attribution"]

    from aad_xai.xai import gradcam_all_blocks
    from captum.attr import IntegratedGradients

    n_attr = min(20, eeg.shape[0])
    eeg_attr = eeg[:n_attr]
    decision.set_envelopes(att[:n_attr], unatt[:n_attr])

    # GradCAM on all conv layers
    print("  Running GradCAM on all Conv1d layers...")
    gc_all = gradcam_all_blocks(decision, eeg_attr, target_class=1)

    gc_summary = {}
    for name, attr_t in gc_all.items():
        arr = attr_t.detach().cpu().numpy()
        gc_summary[name] = {
            "mean_abs": float(np.abs(arr).mean()),
            "max": float(arr.max()),
            "shape": list(arr.shape),
        }

    # Normalise across layers for comparison
    all_means = [v["mean_abs"] for v in gc_summary.values()]
    max_mean = max(all_means) if all_means else 1.0
    for name in gc_summary:
        gc_summary[name]["normalised"] = gc_summary[name]["mean_abs"] / max_mean

    # Save GradCAM arrays
    for name, attr_t in gc_all.items():
        safe_name = name.replace(".", "_")
        np.save(out / f"gradcam_{safe_name}.npy", attr_t.detach().cpu().numpy())

    # Integrated Gradients
    print("  Running Integrated Gradients...")
    ig = IntegratedGradients(decision)
    eeg_ig = eeg_attr.clone().requires_grad_(True)
    try:
        ig_attr = ig.attribute(eeg_ig, target=1, n_steps=50,
                               baselines=torch.zeros_like(eeg_ig))
        ig_arr = ig_attr.detach().cpu().numpy()
        np.save(out / "ig_attributions.npy", ig_arr)

        # Channel-level IG importance
        ig_ch_importance = np.abs(ig_arr).mean(axis=(0, 1))  # (64,)
        ig_summary = {
            "mean_abs": float(np.abs(ig_arr).mean()),
            "channel_importance_top10": [int(c) for c in np.argsort(ig_ch_importance)[::-1][:10]],
        }
        print(f"  IG mean |attr| = {ig_summary['mean_abs']:.6f}")
        print(f"  IG top-10 channels: {ig_summary['channel_importance_top10']}")
    except Exception as e:
        print(f"  IG failed: {e}")
        ig_arr = None
        ig_summary = {"error": str(e)}

    save_json({"gradcam": gc_summary, "ig": ig_summary}, out / "attribution_summary.json")

    # Plot: GradCAM layer comparison
    sorted_gc = sorted(gc_summary.items(), key=lambda x: x[1]["mean_abs"], reverse=True)
    if sorted_gc:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        names = [n.replace("decoder.", "")[:40] for n, _ in sorted_gc]
        vals = [v["normalised"] for _, v in sorted_gc]
        ax.barh(range(len(names)), vals, color="#e65100", alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Normalised GradCAM")
        ax.set_title("GradCAM per Conv1d Layer (normalised)")
        ax.invert_yaxis()

        # IG channel importance
        ax = axes[1]
        if ig_arr is not None:
            ax.bar(range(64), ig_ch_importance,
                   color=["#d32f2f" if ig_ch_importance[i] > np.percentile(ig_ch_importance, 75) else "#1976d2"
                          for i in range(64)], alpha=0.8)
            ax.set_xlabel("Channel")
            ax.set_ylabel("Mean |IG attribution|")
            ax.set_title("Integrated Gradients: Channel Importance")
        else:
            ax.text(0.5, 0.5, "IG failed", ha="center", va="center", transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(out / "attribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# SECTION G — Enhanced sanity checks
# ══════════════════════════════════════════════════════════════════════

def section_g(decision, model, eeg, att, unatt, dirs, device):
    print("\n" + "=" * 70)
    print("SECTION G: ENHANCED SANITY CHECKS")
    print("=" * 70)
    out = dirs["sanity"]

    from aad_xai.xai.sanity_checks import cascading_randomization
    from captum.attr import IntegratedGradients
    from scipy.stats import spearmanr, pearsonr

    n_san = min(10, eeg.shape[0])
    eeg_san = eeg[:n_san]
    decision.set_envelopes(att[:n_san], unatt[:n_san])

    # Define IG attribution function
    def attr_fn(mdl, x):
        ig = IntegratedGradients(mdl)
        x_req = x.clone().requires_grad_(True)
        attr = ig.attribute(x_req, target=1, n_steps=30,
                           baselines=torch.zeros_like(x_req))
        return attr

    print("  Running cascading randomisation...")
    try:
        sanity_maps = cascading_randomization(decision, attr_fn, eeg_san)
    except Exception as e:
        print(f"  Cascading randomisation failed: {e}")
        sanity_maps = {}

    if not sanity_maps:
        print("  Skipping sanity check analysis (no results)")
        return

    original = sanity_maps.get("__original__")
    if original is None:
        print("  No original attribution found")
        return

    orig_flat = original.flatten()
    results = {}

    for layer_name, attr_arr in sanity_maps.items():
        if layer_name == "__original__":
            continue
        layer_flat = attr_arr.flatten()

        # Multiple similarity metrics
        l2_norm = float(np.linalg.norm(attr_arr))
        l2_orig = float(np.linalg.norm(original))
        l2_ratio = l2_norm / (l2_orig + 1e-10)

        try:
            spearman_r, spearman_p = spearmanr(orig_flat, layer_flat)
        except Exception:
            spearman_r, spearman_p = 0.0, 1.0

        try:
            pearson_r, pearson_p = pearsonr(orig_flat, layer_flat)
        except Exception:
            pearson_r, pearson_p = 0.0, 1.0

        # Top-k overlap
        k = max(1, len(orig_flat) // 10)
        top_k_orig = set(np.argsort(np.abs(orig_flat))[-k:])
        top_k_layer = set(np.argsort(np.abs(layer_flat))[-k:])
        topk_overlap = len(top_k_orig & top_k_layer) / k

        # SSIM-inspired structural similarity (simplified for 1D)
        mu_o, mu_l = orig_flat.mean(), layer_flat.mean()
        sig_o, sig_l = orig_flat.std(), layer_flat.std()
        sig_ol = np.mean((orig_flat - mu_o) * (layer_flat - mu_l))
        c1, c2 = 1e-6, 1e-6
        ssim = ((2 * mu_o * mu_l + c1) * (2 * sig_ol + c2)) / \
               ((mu_o**2 + mu_l**2 + c1) * (sig_o**2 + sig_l**2 + c2))

        results[layer_name] = {
            "l2_norm": l2_norm, "l2_ratio": l2_ratio,
            "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "topk_overlap": float(topk_overlap),
            "ssim": float(ssim),
        }

        # Determine pass/fail
        passed = spearman_r < 0.5 or l2_ratio < 0.5
        status = "PASS" if passed else "FAIL"
        print(f"  {layer_name:25s}: Spearman={spearman_r:+.3f}, Pearson={pearson_r:+.3f}, "
              f"TopK={topk_overlap:.2f}, SSIM={ssim:.3f}, L2ratio={l2_ratio:.2f} [{status}]")

    save_json({"original_l2": float(np.linalg.norm(original)), "layers": results},
              out / "sanity_checks.json")

    # Plot
    if results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        names = list(results.keys())
        spearman_vals = [results[n]["spearman_r"] for n in names]
        topk_vals = [results[n]["topk_overlap"] for n in names]
        ssim_vals = [results[n]["ssim"] for n in names]

        for ax, vals, title, yl in zip(
            axes, [spearman_vals, topk_vals, ssim_vals],
            ["Spearman r vs Original", "Top-10% Overlap", "SSIM vs Original"],
            ["Spearman r", "Overlap fraction", "SSIM"],
        ):
            colors = ["#2e7d32" if abs(v) < 0.5 else "#d32f2f" for v in vals]
            ax.barh(range(len(names)), vals, color=colors, alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel(yl)
            ax.set_title(title)
            ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(out / "sanity_checks.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# SECTION H — Subject-wise consistency with clustering
# ══════════════════════════════════════════════════════════════════════

def section_h(decision, ds, n_boot, seed, dirs, device, windows_per_subject: int = 50):
    print("\n" + "=" * 70)
    print("SECTION H: SUBJECT-WISE CONSISTENCY & CLUSTERING")
    print("=" * 70)
    out = dirs["subject_wise"]

    # Use the FULL dataset's subject list (not the --max-samples truncated slice)
    # so that all subjects are always represented regardless of max_samples.
    all_subject_ids = ds.subject_ids
    unique_subjects = sorted(set(all_subject_ids))
    print(f"  {len(unique_subjects)} subjects found in full dataset: {unique_subjects}")

    if len(unique_subjects) < 2:
        print("  Need ≥2 subjects. Skipping.")
        return

    subj_ch_profiles = {}  # subj → (64,) channel importance
    subj_roi_profiles = {}

    for subj in unique_subjects:
        mask = all_subject_ids == subj
        idxs = np.where(mask)[0]
        n_s = min(windows_per_subject, len(idxs))
        idxs = idxs[:n_s]

        eeg_s = torch.stack([ds[i][0] for i in idxs])
        att_s = torch.stack([ds[i][1] for i in idxs])
        unatt_s = torch.stack([ds[i][2] for i in idxs])

        base_p = get_attended_prob(decision, eeg_s, att_s, unatt_s)

        ch_drops = np.zeros(64)
        for ch in range(64):
            eeg_m = eeg_s.clone()
            eeg_m[:, :, ch] = 0.0
            m_p = get_attended_prob(decision, eeg_m, att_s, unatt_s)
            ch_drops[ch] = (base_p - m_p).mean()

        subj_ch_profiles[subj] = ch_drops
        subj_roi_profiles[subj] = {r: float(np.abs(ch_drops[c]).mean()) for r, c in ROIS.items()}

        top_roi = max(subj_roi_profiles[subj], key=subj_roi_profiles[subj].get)
        print(f"  {subj}: {n_s} windows, mean P={base_p.mean():.3f}, top ROI={top_roi}")

    # Pairwise correlation matrix
    subj_list = sorted(subj_ch_profiles.keys())
    ch_matrix = np.array([subj_ch_profiles[s] for s in subj_list])
    corr_matrix = np.corrcoef(ch_matrix)
    triu_idx = np.triu_indices(len(subj_list), k=1)
    pairwise = corr_matrix[triu_idx]

    print(f"\n  Cross-subject consistency (pairwise r):")
    print(f"    Mean = {pairwise.mean():.3f} ± {pairwise.std():.3f}")
    print(f"    Range = [{pairwise.min():.3f}, {pairwise.max():.3f}]")

    save_json({
        "subjects": subj_list,
        "pairwise_r_mean": float(pairwise.mean()),
        "pairwise_r_std": float(pairwise.std()),
        "roi_profiles": subj_roi_profiles,
    }, out / "subject_consistency.json")
    np.save(out / "subject_channel_matrix.npy", ch_matrix)
    np.save(out / "subject_corr_matrix.npy", corr_matrix)

    # Plots: heatmap + dendrogram
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Correlation heatmap
    ax = axes[0]
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(subj_list)))
    ax.set_xticklabels(subj_list, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(subj_list)))
    ax.set_yticklabels(subj_list, fontsize=7)
    ax.set_title("Subject Pairwise Correlation")
    plt.colorbar(im, ax=ax)

    # Dendrogram
    ax = axes[1]
    dist_matrix = 1 - corr_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)  # ensure non-negative
    try:
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method="ward")
        dendrogram(Z, labels=subj_list, ax=ax, leaf_rotation=45, leaf_font_size=7)
        ax.set_title("Subject Clustering (Ward)")
        ax.set_ylabel("Distance (1 - r)")
    except Exception as e:
        ax.text(0.5, 0.5, f"Dendrogram failed:\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)

    # ROI profiles heatmap
    ax = axes[2]
    roi_matrix = np.array([[subj_roi_profiles[s][r] for r in ROIS] for s in subj_list])
    im2 = ax.imshow(roi_matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ROIS)))
    ax.set_xticklabels(list(ROIS.keys()), rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(subj_list)))
    ax.set_yticklabels(subj_list, fontsize=7)
    ax.set_title("Subject x ROI Importance")
    plt.colorbar(im2, ax=ax)

    plt.tight_layout()
    plt.savefig(out / "subject_consistency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# SECTION I — Frequency-band analysis
# ══════════════════════════════════════════════════════════════════════

def section_i(decision, eeg, att, unatt, n_boot, seed, dirs, device):
    print("\n" + "=" * 70)
    print("SECTION I: FREQUENCY-BAND ANALYSIS (BUTTERWORTH)")
    print("=" * 70)
    out = dirs["frequency_band"]

    from scipy.signal import butter, sosfiltfilt

    BANDS = OrderedDict([
        ("delta",  (0.5, 4.0)),
        ("theta",  (4.0, 8.0)),
        ("alpha",  (8.0, 13.0)),
        ("beta",   (13.0, 30.0)),
    ])

    N = eeg.shape[0]
    base_probs = get_attended_prob(decision, eeg, att, unatt)

    # Edge-artifact handling: pad → filter → unpad
    pad_samples = 64  # 1s padding to handle edge artifacts

    band_results = []
    band_roi_results = []

    for band_name, (lo, hi) in BANDS.items():
        nyq = FS / 2.0
        lo_n = max(lo / nyq, 0.01)
        hi_n = min(hi / nyq, 0.99)
        sos = butter(4, [lo_n, hi_n], btype="bandpass", output="sos")

        eeg_np = eeg.numpy()  # (N, T, 64)
        T = eeg_np.shape[1]

        # Pad → filter → unpad to handle edge artifacts
        band_content = np.zeros_like(eeg_np)
        for w in range(N):
            for ch in range(64):
                sig = eeg_np[w, :, ch]
                # Mirror-pad
                padded = np.pad(sig, pad_samples, mode="reflect")
                filtered = sosfiltfilt(sos, padded)
                band_content[w, :, ch] = filtered[pad_samples:pad_samples + T]

        eeg_no_band = torch.from_numpy((eeg_np - band_content).astype(np.float32))
        p_no_band = get_attended_prob(decision, eeg_no_band, att, unatt)
        drops_pw = base_probs - p_no_band

        mean, lo_ci, hi_ci = bootstrap_ci(drops_pw, n_boot, seed=seed)
        band_results.append({
            "band": band_name, "freq_range": f"{lo}-{hi} Hz",
            "mean_dp": mean, "ci_lo": lo_ci, "ci_hi": hi_ci,
        })

        # ROI × band interaction
        roi_row = {"band": band_name}
        for roi_name, chs in ROIS.items():
            eeg_roi_np = eeg_np.copy()
            for w in range(N):
                for ch in chs:
                    sig = eeg_roi_np[w, :, ch]
                    padded = np.pad(sig, pad_samples, mode="reflect")
                    filtered = sosfiltfilt(sos, padded)
                    eeg_roi_np[w, :, ch] -= filtered[pad_samples:pad_samples + T]

            p_roi = get_attended_prob(decision, torch.from_numpy(eeg_roi_np.astype(np.float32)),
                                      att, unatt)
            roi_dp = (base_probs - p_roi).mean()
            roi_row[roi_name] = float(roi_dp)
        band_roi_results.append(roi_row)

        print(f"  {band_name:8s} ({lo:.1f}-{hi:.1f} Hz): "
              f"ΔP = {mean:+.5f} [{lo_ci:+.5f}, {hi_ci:+.5f}]")

    save_json({"bands": band_results, "band_roi": band_roi_results},
              out / "frequency_band.json")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bnames = [r["band"] for r in band_results]
    bmeans = [abs(r["mean_dp"]) for r in band_results]
    berrs = [[abs(r["mean_dp"]) - abs(r["ci_lo"]) for r in band_results],
             [abs(r["ci_hi"]) - abs(r["mean_dp"]) for r in band_results]]
    band_colors = {"delta": "#1565c0", "theta": "#2e7d32", "alpha": "#f57f17", "beta": "#d32f2f"}
    ax.bar(range(len(bnames)), bmeans,
           color=[band_colors.get(b, "gray") for b in bnames], alpha=0.8, capsize=5)
    ax.set_xticks(range(len(bnames)))
    ax.set_xticklabels(bnames)
    ax.set_ylabel("|ΔP(attended)|")
    ax.set_title("Frequency Band Importance")

    # Band × ROI heatmap
    ax = axes[1]
    br_matrix = np.array([[r[roi] for roi in ROIS] for r in band_roi_results])
    im = ax.imshow(np.abs(br_matrix), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(ROIS)))
    ax.set_xticklabels(list(ROIS.keys()), rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(bnames)))
    ax.set_yticklabels(bnames)
    ax.set_title("Band x ROI Importance")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(out / "frequency_band.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# SECTION J — Correct vs incorrect (correlation-based labels)
# ══════════════════════════════════════════════════════════════════════

def section_j(decision, model, eeg, att, unatt, n_boot, seed, dirs, device):
    print("\n" + "=" * 70)
    print("SECTION J: CORRECT vs INCORRECT (CORRELATION-BASED LABELS)")
    print("=" * 70)
    out = dirs["correct_incorrect"]
    N = eeg.shape[0]

    # True labels: r(predicted_env, attended_env) > r(predicted_env, unattended_env)
    with torch.no_grad():
        pred_env = model(eeg)  # (N, T, 1)

    corr_att = np.zeros(N)
    corr_unatt = np.zeros(N)
    for i in range(N):
        p = pred_env[i, :, 0].cpu().numpy()
        a = att[i, :, 0].cpu().numpy()
        u = unatt[i, :, 0].cpu().numpy()
        corr_att[i] = np.corrcoef(p, a)[0, 1]
        corr_unatt[i] = np.corrcoef(p, u)[0, 1]

    # True label: correct if correlation with attended > correlation with unattended
    correct_mask = corr_att > corr_unatt
    n_correct = correct_mask.sum()
    n_incorrect = (~correct_mask).sum()
    accuracy = n_correct / N

    print(f"  Correct (r_att > r_unatt): {n_correct}/{N} = {accuracy:.1%}")
    print(f"  Mean r_att = {corr_att.mean():.4f}, Mean r_unatt = {corr_unatt.mean():.4f}")

    if n_correct < 5 or n_incorrect < 5:
        print("  Need ≥5 in each group. Skipping detailed analysis.")
        save_json({"n_correct": int(n_correct), "n_incorrect": int(n_incorrect),
                    "accuracy": float(accuracy)}, out / "correct_incorrect.json")
        return

    # Channel occlusion for each group
    groups = {"correct": correct_mask, "incorrect": ~correct_mask}
    group_ch_drops = {}
    group_roi = {}

    for gname, gmask in groups.items():
        idxs = np.where(gmask)[0][:100]
        eeg_g = eeg[idxs]
        att_g = att[idxs]
        unatt_g = unatt[idxs]
        base_g = get_attended_prob(decision, eeg_g, att_g, unatt_g)

        ch_drops = np.zeros(64)
        for ch in range(64):
            eeg_m = eeg_g.clone()
            eeg_m[:, :, ch] = 0.0
            m_p = get_attended_prob(decision, eeg_m, att_g, unatt_g)
            ch_drops[ch] = (base_g - m_p).mean()

        group_ch_drops[gname] = ch_drops
        group_roi[gname] = {r: float(np.abs(ch_drops[c]).mean()) for r, c in ROIS.items()}

    # Top-10 comparison
    top10_correct = set(np.argsort(np.abs(group_ch_drops["correct"]))[-10:])
    top10_incorrect = set(np.argsort(np.abs(group_ch_drops["incorrect"]))[-10:])
    overlap = len(top10_correct & top10_incorrect)

    # Statistical comparison
    from scipy.stats import ttest_ind
    diff = group_ch_drops["correct"] - group_ch_drops["incorrect"]
    t_stat, t_pval = ttest_ind(
        np.abs(group_ch_drops["correct"]),
        np.abs(group_ch_drops["incorrect"]),
    )

    results = {
        "n_correct": int(n_correct), "n_incorrect": int(n_incorrect),
        "accuracy": float(accuracy),
        "top10_overlap": overlap,
        "channel_diff_t_stat": float(t_stat), "channel_diff_p_value": float(t_pval),
        "roi_correct": group_roi["correct"],
        "roi_incorrect": group_roi["incorrect"],
    }
    save_json(results, out / "correct_incorrect.json")
    np.save(out / "correct_channel_drops.npy", group_ch_drops["correct"])
    np.save(out / "incorrect_channel_drops.npy", group_ch_drops["incorrect"])

    print(f"  Top-10 channel overlap: {overlap}/10")
    print(f"  Channel importance difference: t={t_stat:.3f}, p={t_pval:.4f}")

    print("\n  ROI comparison:")
    for roi in ROIS:
        c = group_roi["correct"][roi]
        ic = group_roi["incorrect"][roi]
        print(f"    {roi:20s}: correct={c:.5f}, incorrect={ic:.5f}, diff={c-ic:+.5f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Channel comparison (top 20 by difference)
    ax = axes[0]
    diff_order = np.argsort(np.abs(diff))[::-1][:20]
    x_pos = np.arange(20)
    ax.barh(x_pos - 0.15, [group_ch_drops["correct"][c] for c in diff_order],
            height=0.3, color="#2e7d32", alpha=0.8, label="Correct")
    ax.barh(x_pos + 0.15, [group_ch_drops["incorrect"][c] for c in diff_order],
            height=0.3, color="#d32f2f", alpha=0.8, label="Incorrect")
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f"Ch {c}" for c in diff_order], fontsize=7)
    ax.set_xlabel("ΔP(attended)")
    ax.set_title("Top-20 Channels: Correct vs Incorrect")
    ax.legend(fontsize=8)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.invert_yaxis()

    # ROI comparison
    ax = axes[1]
    roi_names = list(ROIS.keys())
    x_r = np.arange(len(roi_names))
    ax.bar(x_r - 0.15, [group_roi["correct"][r] for r in roi_names],
           width=0.3, color="#2e7d32", alpha=0.8, label="Correct")
    ax.bar(x_r + 0.15, [group_roi["incorrect"][r] for r in roi_names],
           width=0.3, color="#d32f2f", alpha=0.8, label="Incorrect")
    ax.set_xticks(x_r)
    ax.set_xticklabels(roi_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|ΔP|")
    ax.set_title("ROI: Correct vs Incorrect")
    ax.legend(fontsize=8)

    # Scatter: r_att vs r_unatt
    ax = axes[2]
    ax.scatter(corr_att[correct_mask], corr_unatt[correct_mask],
               alpha=0.4, c="#2e7d32", label="Correct", s=15)
    ax.scatter(corr_att[~correct_mask], corr_unatt[~correct_mask],
               alpha=0.4, c="#d32f2f", label="Incorrect", s=15)
    ax.plot([-.5, 1], [-.5, 1], "k--", linewidth=0.5)
    ax.set_xlabel("r(pred, attended)")
    ax.set_ylabel("r(pred, unattended)")
    ax.set_title("Prediction Correlation Space")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out / "correct_incorrect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    dirs = make_output_dirs(Path(args.output_dir))

    sections = [s.upper() for s in args.sections]

    print("=" * 70)
    print("VLAAI COMPREHENSIVE XAI ANALYSIS")
    print(f"  Seed: {args.seed}, Bootstrap: {args.n_boot}, Max samples: {args.max_samples}")
    print(f"  Sections: {', '.join(sections)}")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading dataset...")
    from aad_xai.data.vlaai_dataset import VLAAIDTUDataset

    ds = VLAAIDTUDataset(
        data_dir=args.data_dir,
        window_length=320, hop=64,
        subjects=args.subjects,
    )
    N = min(args.max_samples, len(ds))
    print(f"  {len(ds)} total windows, using {N}")

    eeg_all = torch.stack([ds[i][0] for i in range(N)])
    att_all = torch.stack([ds[i][1] for i in range(N)])
    unatt_all = torch.stack([ds[i][2] for i in range(N)])

    # Subject IDs for the subset
    subject_ids = ds.subject_ids[:N]

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("Loading VLAAI model...")
    from aad_xai.models import VLAAIPyTorch, AADDecisionEEGOnly

    try:
        model = VLAAIPyTorch.from_h5(args.h5_path)
    except Exception as e:
        print(f"  H5 failed ({e}), using random init")
        model = VLAAIPyTorch()
    model.eval().to(device)

    decision = AADDecisionEEGOnly(model)
    decision.eval().to(device)

    # Compute correlation-based binary labels
    with torch.no_grad():
        pred_env = model(eeg_all)
    corrs = np.array([np.corrcoef(pred_env[i, :, 0].cpu().numpy(),
                                   att_all[i, :, 0].cpu().numpy())[0, 1]
                       for i in range(N)])
    median_r = np.median(corrs)
    labels = (corrs >= median_r).astype(np.int64)
    print(f"  Binary labels: {labels.sum()} high, {(1 - labels).sum()} low (median r={median_r:.4f})")

    # Quick sanity
    decision.set_envelopes(att_all[:3], unatt_all[:3])
    with torch.no_grad():
        test_logits = decision(eeg_all[:3])
    print(f"  Decision logits (sample): {test_logits[0].cpu().numpy()}")

    # ------------------------------------------------------------------
    # Run sections
    # ------------------------------------------------------------------
    if "A" in sections:
        section_a(model, dirs, device)

    if "B" in sections:
        decision.set_envelopes(att_all, unatt_all)
        section_b(model, decision, eeg_all, att_all, unatt_all,
                  args.n_boot, args.seed, dirs, device)

    if "C" in sections:
        decision.set_envelopes(att_all, unatt_all)
        section_c(decision, eeg_all, att_all, unatt_all,
                  args.n_boot, args.seed, dirs, device)

    if "D" in sections:
        decision.set_envelopes(att_all, unatt_all)
        section_d(decision, eeg_all, att_all, unatt_all,
                  args.n_boot, args.seed, dirs, device)

    if "E" in sections:
        section_e(model, eeg_all, att_all, labels, subject_ids,
                  args.n_boot, args.seed, dirs, device)

    if "F" in sections:
        section_f(decision, model, eeg_all, att_all, unatt_all, dirs, device)

    if "G" in sections:
        section_g(decision, model, eeg_all, att_all, unatt_all, dirs, device)

    if "H" in sections:
        section_h(decision, ds, args.n_boot, args.seed, dirs, device,
                  windows_per_subject=args.windows_per_subject)

    if "I" in sections:
        decision.set_envelopes(att_all, unatt_all)
        section_i(decision, eeg_all, att_all, unatt_all,
                  args.n_boot, args.seed, dirs, device)

    if "J" in sections:
        section_j(decision, model, eeg_all, att_all, unatt_all,
                  args.n_boot, args.seed, dirs, device)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"  All results saved to: {args.output_dir}")
    print(f"  Sections run: {', '.join(sections)}")
    print("=" * 70)

    # Save run config
    save_json({
        "seed": args.seed, "n_boot": args.n_boot, "max_samples": args.max_samples,
        "n_windows": N, "n_subjects": len(set(subject_ids)),
        "sections": sections, "device": args.device,
    }, dirs["root"] / "run_config.json")


if __name__ == "__main__":
    main()
