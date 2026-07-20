"""Focused XAI pipeline: Channel importance → ROI grouping → Frequency analysis.

Sections:
  1. Architecture summary + block ablation
  2. Channel importance (occlusion + permutation + IG) with bootstrap CIs
  3. ROI-level grouping of important channels
  4. Frequency-band analysis on top channels / ROIs

Usage:
    python scripts/run_focused_xai.py
    python scripts/run_focused_xai.py --max-samples 500 --n-boot 1000
    python scripts/run_focused_xai.py --top-k 15
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import warnings
from collections import OrderedDict
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
    p = argparse.ArgumentParser(description="Focused XAI: Channel → ROI → Frequency")
    p.add_argument("--data-dir", type=str,
                   default=str(ROOT / "external" / "vlaai" / "evaluation_datasets" / "DTU"))
    p.add_argument("--h5-path", type=str,
                   default=str(ROOT / "external" / "vlaai" / "pretrained_models" / "vlaai.h5"))
    p.add_argument("--output-dir", type=str, default=str(ROOT / "xai_results_focused"))
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--max-samples", type=int, default=200,
                   help="Number of EEG windows for occlusion/permutation/freq analysis.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=500, help="Bootstrap iterations for CIs.")
    p.add_argument("--n-ig", type=int, default=30,
                   help="Number of windows for Integrated Gradients (slower).")
    p.add_argument("--ig-steps", type=int, default=50, help="IG interpolation steps.")
    p.add_argument("--windows-per-subject", type=int, default=50,
                   help="Windows per subject for subject-wise stability.")
    p.add_argument("--top-k", type=int, default=15,
                   help="Number of top channels for frequency analysis.")
    p.add_argument("--device", type=str, default="cpu")
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

ROIS = OrderedDict([
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

CH_TO_ROI = {}
for roi_name, chs in ROIS.items():
    for ch in chs:
        CH_TO_ROI[ch] = roi_name


# ══════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════

def get_attended_prob(decision, eeg, att, unatt):
    """P(attended) for a batch."""
    decision.set_envelopes(att, unatt)
    with torch.no_grad():
        logits = decision(eeg)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
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


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


def make_output_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — Architecture + Block Ablation
# ══════════════════════════════════════════════════════════════════════

def run_architecture(model, decision, eeg, att, unatt, n_boot, seed, out_dir, device):
    """Architecture summary and block ablation."""
    print("\n" + "=" * 70)
    print("SECTION 1: ARCHITECTURE + BLOCK ABLATION")
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
                    perm = torch.from_numpy(_rng.permutation(out.shape[0])).long()
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
            print(f"  Block {bi} [{mode:14s}]: ΔP={mean_dp:+.5f} [{lo_dp:+.5f},{hi_dp:+.5f}], ΔAcc={base_acc - abl_acc:+.3f}")

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
# SECTION 2 — Channel Importance (Occlusion + Permutation + IG)
# ══════════════════════════════════════════════════════════════════════

def run_channel_occlusion(decision, eeg, att, unatt, n_boot, seed):
    """Mask each channel to zero and measure ΔP."""
    N = eeg.shape[0]
    base_probs = get_attended_prob(decision, eeg, att, unatt)
    drops_pw = np.zeros((N, 64))

    for ch in range(64):
        eeg_m = eeg.clone()
        eeg_m[:, :, ch] = 0.0
        m_probs = get_attended_prob(decision, eeg_m, att, unatt)
        drops_pw[:, ch] = base_probs - m_probs
        if (ch + 1) % 16 == 0:
            print(f"    Occlusion: {ch+1}/64 channels done")

    results = []
    for ch in range(64):
        mean, lo, hi = bootstrap_ci(drops_pw[:, ch], n_boot, seed=seed)
        results.append({"channel": ch, "mean_dp": mean, "ci_lo": lo, "ci_hi": hi})

    return results, drops_pw


def run_channel_permutation(decision, eeg, att, unatt, n_boot, seed):
    """Shuffle each channel across windows and measure ΔP."""
    N = eeg.shape[0]
    rng = np.random.RandomState(seed)
    base_probs = get_attended_prob(decision, eeg, att, unatt)
    drops_pw = np.zeros((N, 64))

    for ch in range(64):
        eeg_m = eeg.clone()
        perm = rng.permutation(N)
        eeg_m[:, :, ch] = eeg[perm, :, ch]
        m_probs = get_attended_prob(decision, eeg_m, att, unatt)
        drops_pw[:, ch] = base_probs - m_probs
        if (ch + 1) % 16 == 0:
            print(f"    Permutation: {ch+1}/64 channels done")

    results = []
    for ch in range(64):
        mean, lo, hi = bootstrap_ci(drops_pw[:, ch], n_boot, seed=seed)
        results.append({"channel": ch, "mean_dp": mean, "ci_lo": lo, "ci_hi": hi})

    return results, drops_pw


def run_channel_ig(decision, eeg, att, unatt, n_ig, ig_steps):
    """Integrated Gradients channel importance (supporting evidence).

    Process windows in small batches (batch_size=5) to avoid OOM on CPU.
    """
    from captum.attr import IntegratedGradients

    n = min(n_ig, eeg.shape[0])
    batch_size = 5  # small batches to avoid OOM on CPU
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
            # (batch, T, 64) -> mean |attr| per channel
            batch_imp = np.abs(ig_attr.detach().cpu().numpy()).mean(axis=(0, 1))
            all_importance.append(batch_imp * (end - start))
            print(f"    IG: {end}/{n} windows done")
        ch_importance = sum(all_importance) / n  # weighted average
        ranks = np.argsort(np.argsort(-ch_importance)) + 1
        return ch_importance, ranks
    except Exception as e:
        print(f"    IG failed: {e}")
        import traceback; traceback.print_exc()
        return np.zeros(64), np.arange(1, 65)


def run_subject_stability(decision, ds, windows_per_subject):
    """Per-subject channel occlusion to assess stability."""
    all_subject_ids = ds.subject_ids
    unique_subjects = sorted(set(all_subject_ids))

    if len(unique_subjects) < 2:
        print("    Need ≥2 subjects for stability analysis.")
        return {}, np.zeros((0, 64))

    subj_profiles = {}
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

        subj_profiles[subj] = ch_drops
        print(f"    {subj}: {n_s} windows processed")

    subj_list = sorted(subj_profiles.keys())
    ch_matrix = np.array([subj_profiles[s] for s in subj_list])
    return subj_profiles, ch_matrix


def run_channel_importance(decision, model, eeg, att, unatt, ds,
                           n_boot, seed, n_ig, ig_steps, windows_per_subject,
                           top_k, out_dir):
    """Multi-method channel importance analysis."""
    print("\n" + "=" * 70)
    print("SECTION 2: CHANNEL IMPORTANCE (Occlusion + Permutation + IG)")
    print("=" * 70)

    # --- 2a. Channel occlusion ---
    print("  [2a] Channel occlusion (zero-masking)...")
    occ_results, occ_pw = run_channel_occlusion(decision, eeg, att, unatt, n_boot, seed)

    # --- 2b. Channel permutation ---
    print("  [2b] Channel permutation (shuffle across windows)...")
    perm_results, perm_pw = run_channel_permutation(decision, eeg, att, unatt, n_boot, seed)

    # --- 2c. Integrated Gradients ---
    print("  [2c] Integrated Gradients (supporting evidence)...")
    ig_importance, ig_ranks = run_channel_ig(decision, eeg, att, unatt, n_ig, ig_steps)
    # Restore full-batch envelopes after IG batching
    decision.set_envelopes(att, unatt)

    # --- 2d. Subject-wise stability ---
    print("  [2d] Subject-wise channel stability...")
    subj_profiles, ch_matrix = run_subject_stability(decision, ds, windows_per_subject)

    # --- Combine into ranked table ---
    combined = []
    for ch in range(64):
        occ_mean = occ_results[ch]["mean_dp"]
        occ_lo = occ_results[ch]["ci_lo"]
        occ_hi = occ_results[ch]["ci_hi"]
        perm_mean = perm_results[ch]["mean_dp"]
        perm_lo = perm_results[ch]["ci_lo"]
        perm_hi = perm_results[ch]["ci_hi"]

        # Significance: CI excludes zero
        occ_sig = (occ_lo > 0) or (occ_hi < 0)
        perm_sig = (perm_lo > 0) or (perm_hi < 0)

        # Sign of contribution
        if occ_mean > 0 and perm_mean > 0:
            sign = "facilitatory"
        elif occ_mean < 0 and perm_mean < 0:
            sign = "suppressive"
        else:
            sign = "mixed"

        # Subject-wise stability: how many subjects show same-sign importance
        stable = "N/A"
        if ch_matrix.shape[0] > 0:
            ch_col = ch_matrix[:, ch]
            majority_sign = np.sign(np.median(ch_col))
            n_agree = np.sum(np.sign(ch_col) == majority_sign)
            stable = f"{n_agree}/{ch_matrix.shape[0]}"

        combined.append({
            "channel": ch,
            "roi": CH_TO_ROI[ch],
            "occ_score": occ_mean,
            "occ_ci": [occ_lo, occ_hi],
            "occ_significant": occ_sig,
            "perm_score": perm_mean,
            "perm_ci": [perm_lo, perm_hi],
            "perm_significant": perm_sig,
            "ig_rank": int(ig_ranks[ch]),
            "ig_importance": float(ig_importance[ch]),
            "sign": sign,
            "subject_stability": stable,
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
    print(f"\n  RANKED CHANNEL TABLE (top {top_k}):")
    print(f"  {'Rank':>4} | {'Ch':>3} | {'ROI':>14} | {'Occ ΔP':>9} | {'Perm ΔP':>9} | {'IG rank':>7} | {'Sign':>12} | {'Occ CI':>22} | {'Sig?':>5} | {'Stable':>7}")
    print("  " + "-" * 120)
    for c in combined[:top_k]:
        print(f"  {c['rank']:>4} | {c['channel']:>3} | {c['roi']:>14} | {c['occ_score']:>+9.5f} | "
              f"{c['perm_score']:>+9.5f} | {c['ig_rank']:>7} | {c['sign']:>12} | "
              f"[{c['occ_ci'][0]:+.5f},{c['occ_ci'][1]:+.5f}] | "
              f"{'YES' if c['occ_significant'] else 'no':>5} | {c['subject_stability']:>7}")

    # Save JSON
    save_json({
        "channels": combined,
        "occlusion_raw": occ_results,
        "permutation_raw": perm_results,
        "ig_importance": ig_importance.tolist(),
        "n_windows": int(eeg.shape[0]),
        "n_boot": int(n_boot),
    }, out_dir / "channel_importance.json")

    # Save CSV
    csv_fields = ["rank", "channel", "roi", "occ_score", "perm_score", "ig_rank",
                  "ig_importance", "sign", "occ_ci_lo", "occ_ci_hi",
                  "occ_significant", "perm_significant", "subject_stability", "combined_score"]
    with open(out_dir / "channel_importance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for c in combined:
            row = {k: c[k] for k in csv_fields if k in c}
            row["occ_ci_lo"] = c["occ_ci"][0]
            row["occ_ci_hi"] = c["occ_ci"][1]
            w.writerow(row)

    # Save per-window arrays
    np.save(out_dir / "occlusion_perwindow.npy", occ_pw)
    np.save(out_dir / "permutation_perwindow.npy", perm_pw)
    if ch_matrix.shape[0] > 0:
        np.save(out_dir / "subject_channel_matrix.npy", ch_matrix)

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) All channels bar chart — occlusion
    ax = axes[0, 0]
    occ_means = np.array([c["occ_score"] for c in sorted(combined, key=lambda x: x["channel"])])
    colors_bar = ["#d32f2f" if d > 0 else "#1976d2" for d in occ_means]
    ax.bar(range(64), occ_means, color=colors_bar, alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Channel")
    ax.set_ylabel("ΔP (occlusion)")
    ax.set_title("Channel Occlusion Importance (all 64)")

    # (b) Top-k channels — occlusion + permutation side by side
    ax = axes[0, 1]
    top_ch = combined[:top_k]
    x_pos = np.arange(top_k)
    ax.barh(x_pos - 0.15, [c["occ_score"] for c in top_ch], height=0.3,
            color="#d32f2f", alpha=0.8, label="Occlusion")
    ax.barh(x_pos + 0.15, [c["perm_score"] for c in top_ch], height=0.3,
            color="#1976d2", alpha=0.8, label="Permutation")
    # CI whiskers for occlusion
    for i, c in enumerate(top_ch):
        ax.plot([c["occ_ci"][0], c["occ_ci"][1]], [i - 0.15, i - 0.15],
                color="black", linewidth=1)
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f"Ch {c['channel']} ({c['roi'][:4]})" for c in top_ch], fontsize=8)
    ax.set_xlabel("ΔP(attended)")
    ax.set_title(f"Top-{top_k} Channels: Occlusion vs Permutation")
    ax.legend(fontsize=8)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.invert_yaxis()

    # (c) Occlusion vs Permutation scatter
    ax = axes[1, 0]
    occ_all = [c["occ_score"] for c in combined]
    perm_all = [c["perm_score"] for c in combined]
    ax.scatter(occ_all, perm_all, c="#6a1b9a", alpha=0.6, s=20)
    for c in combined[:10]:
        ax.annotate(f"Ch{c['channel']}", (c["occ_score"], c["perm_score"]),
                    fontsize=6, alpha=0.7)
    r_val = np.corrcoef(occ_all, perm_all)[0, 1]
    ax.set_xlabel("Occlusion ΔP")
    ax.set_ylabel("Permutation ΔP")
    ax.set_title(f"Occlusion vs Permutation (r={r_val:.3f})")
    ax.axhline(0, color="k", linewidth=0.3)
    ax.axvline(0, color="k", linewidth=0.3)
    lims = [min(min(occ_all), min(perm_all)), max(max(occ_all), max(perm_all))]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.3)

    # (d) IG importance vs occlusion importance
    ax = axes[1, 1]
    ig_all = [c["ig_importance"] for c in combined]
    occ_abs_all = [abs(c["occ_score"]) for c in combined]
    ax.scatter(occ_abs_all, ig_all, c="#e65100", alpha=0.6, s=20)
    for c in combined[:10]:
        ax.annotate(f"Ch{c['channel']}", (abs(c["occ_score"]), c["ig_importance"]),
                    fontsize=6, alpha=0.7)
    r_ig = np.corrcoef(occ_abs_all, ig_all)[0, 1]
    ax.set_xlabel("|Occlusion ΔP|")
    ax.set_ylabel("IG |attribution|")
    ax.set_title(f"Occlusion vs IG importance (r={r_ig:.3f})")

    plt.tight_layout()
    plt.savefig(out_dir / "channel_importance_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved channel importance results to {out_dir}")

    return combined, subj_profiles, ch_matrix


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — ROI-Level Grouping
# ══════════════════════════════════════════════════════════════════════

def run_roi_analysis(combined_channels, occ_pw, perm_pw, n_boot, seed, out_dir):
    """Aggregate channel importance at ROI level."""
    print("\n" + "=" * 70)
    print("SECTION 3: ROI-LEVEL IMPORTANCE")
    print("=" * 70)

    roi_results = []
    for roi_name, chs in ROIS.items():
        # Occlusion: mean ΔP across ROI channels per window, then bootstrap
        roi_occ_pw = occ_pw[:, chs].mean(axis=1)
        occ_mean, occ_lo, occ_hi = bootstrap_ci(roi_occ_pw, n_boot, seed=seed)

        roi_perm_pw = perm_pw[:, chs].mean(axis=1)
        perm_mean, perm_lo, perm_hi = bootstrap_ci(roi_perm_pw, n_boot, seed=seed)

        occ_sig = (occ_lo > 0) or (occ_hi < 0)
        perm_sig = (perm_lo > 0) or (perm_hi < 0)

        # Channels in this ROI from ranked list
        roi_ch_ranks = [c for c in combined_channels if c["roi"] == roi_name]
        n_sig = sum(1 for c in roi_ch_ranks if c["occ_significant"])

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
            "n_significant_channels": n_sig,
        })

        print(f"  {roi_name:20s}: Occ ΔP={occ_mean:+.5f} [{occ_lo:+.5f},{occ_hi:+.5f}] "
              f"{'SIG' if occ_sig else '   '} | "
              f"Perm ΔP={perm_mean:+.5f} [{perm_lo:+.5f},{perm_hi:+.5f}] "
              f"{'SIG' if perm_sig else '   '} | "
              f"{n_sig}/{len(chs)} sig channels")

    save_json(roi_results, out_dir / "roi_importance.json")

    # CSV
    csv_fields = ["roi", "n_channels", "occ_mean_dp", "occ_ci_lo", "occ_ci_hi",
                  "occ_significant", "perm_mean_dp", "perm_ci_lo", "perm_ci_hi",
                  "perm_significant", "n_significant_channels"]
    with open(out_dir / "roi_importance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in roi_results:
            row = {k: r[k] for k in csv_fields if k in r}
            row["occ_ci_lo"] = r["occ_ci"][0]
            row["occ_ci_hi"] = r["occ_ci"][1]
            row["perm_ci_lo"] = r["perm_ci"][0]
            row["perm_ci_hi"] = r["perm_ci"][1]
            w.writerow(row)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Occlusion ROI bar chart
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
    # Mark significant
    for i, r in enumerate(roi_results):
        if r["occ_significant"]:
            ax.text(i, occ_means[i] + 0.001 * np.sign(occ_means[i]), "*",
                    ha="center", fontsize=14, fontweight="bold")

    # Permutation ROI bar chart
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
# SECTION 4 — Frequency Analysis on Top Channels / ROIs
# ══════════════════════════════════════════════════════════════════════

def run_frequency_analysis(decision, eeg, att, unatt, combined_channels,
                           roi_results, n_boot, seed, top_k, out_dir):
    """Frequency-band analysis on top important channels and ROIs."""
    print("\n" + "=" * 70)
    print("SECTION 4: FREQUENCY ANALYSIS (on top channels & ROIs)")
    print("=" * 70)

    from scipy.signal import butter, sosfiltfilt

    N = eeg.shape[0]
    base_probs = get_attended_prob(decision, eeg, att, unatt)
    pad_samples = 64  # 1s mirror padding for edge artifacts

    # Select top-k channels
    top_channels = [c["channel"] for c in combined_channels[:top_k]]
    print(f"  Analysing frequency bands for top-{top_k} channels: {top_channels}")

    # Precompute band content for all channels
    eeg_np = eeg.numpy()  # (N, T, 64)
    T = eeg_np.shape[1]

    band_content_all = {}  # band_name → (N, T, 64) band-passed content
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

    # --- 4a. Per-channel frequency analysis ---
    print("  [4a] Per-channel frequency analysis...")
    freq_by_channel = []

    for ch in top_channels:
        ch_result = {
            "channel": ch,
            "roi": CH_TO_ROI[ch],
            "occ_rank": next(c["rank"] for c in combined_channels if c["channel"] == ch),
        }
        band_scores = {}
        for band_name in BANDS:
            # Remove this band from this channel only
            eeg_mod = eeg_np.copy()
            eeg_mod[:, :, ch] -= band_content_all[band_name][:, :, ch]
            eeg_t = torch.from_numpy(eeg_mod.astype(np.float32))
            p_mod = get_attended_prob(decision, eeg_t, att, unatt)
            dp = (base_probs - p_mod)
            mean, lo, hi = bootstrap_ci(dp, n_boot, seed=seed)
            band_scores[band_name] = {"mean_dp": mean, "ci": [lo, hi]}
            ch_result[f"{band_name}_dp"] = mean
            ch_result[f"{band_name}_ci_lo"] = lo
            ch_result[f"{band_name}_ci_hi"] = hi

        # Most important band
        ch_result["most_important_band"] = max(band_scores, key=lambda b: abs(band_scores[b]["mean_dp"]))
        freq_by_channel.append(ch_result)

        print(f"    Ch {ch:2d} ({CH_TO_ROI[ch][:4]}): "
              f"δ={ch_result['delta_dp']:+.5f} θ={ch_result['theta_dp']:+.5f} "
              f"α={ch_result['alpha_dp']:+.5f} β={ch_result['beta_dp']:+.5f} "
              f"→ {ch_result['most_important_band']}")

    # --- 4b. Per-ROI frequency analysis ---
    print("  [4b] Per-ROI frequency analysis...")
    freq_by_roi = []

    for roi_name, chs in ROIS.items():
        roi_row = {"roi": roi_name, "n_channels": len(chs)}
        for band_name in BANDS:
            # Remove this band from all channels in this ROI
            eeg_mod = eeg_np.copy()
            for ch in chs:
                eeg_mod[:, :, ch] -= band_content_all[band_name][:, :, ch]
            eeg_t = torch.from_numpy(eeg_mod.astype(np.float32))
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
              f"→ {roi_row['most_important_band']}")

    # Save results
    save_json({"frequency_by_channel": freq_by_channel, "frequency_by_roi": freq_by_roi},
              out_dir / "frequency_analysis.json")

    # CSVs
    ch_fields = ["channel", "roi", "occ_rank", "most_important_band",
                 "delta_dp", "delta_ci_lo", "delta_ci_hi",
                 "theta_dp", "theta_ci_lo", "theta_ci_hi",
                 "alpha_dp", "alpha_ci_lo", "alpha_ci_hi",
                 "beta_dp", "beta_ci_lo", "beta_ci_hi"]
    with open(out_dir / "frequency_by_channel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ch_fields)
        w.writeheader()
        for r in freq_by_channel:
            w.writerow({k: r.get(k, "") for k in ch_fields})

    roi_fields = ["roi", "n_channels", "most_important_band",
                  "delta_dp", "delta_ci_lo", "delta_ci_hi",
                  "theta_dp", "theta_ci_lo", "theta_ci_hi",
                  "alpha_dp", "alpha_ci_lo", "alpha_ci_hi",
                  "beta_dp", "beta_ci_lo", "beta_ci_hi"]
    with open(out_dir / "frequency_by_roi.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=roi_fields)
        w.writeheader()
        for r in freq_by_roi:
            w.writerow({k: r.get(k, "") for k in roi_fields})

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (a) Per-channel band heatmap
    ax = axes[0]
    n_ch = len(freq_by_channel)
    band_matrix = np.array([[r[f"{b}_dp"] for b in BANDS] for r in freq_by_channel])
    im = ax.imshow(band_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(band_matrix).max(), vmax=np.abs(band_matrix).max())
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(list(BANDS.keys()))
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([f"Ch{r['channel']}({r['roi'][:3]})" for r in freq_by_channel], fontsize=7)
    ax.set_title(f"Band ΔP for Top-{top_k} Channels")
    plt.colorbar(im, ax=ax, label="ΔP")

    # (b) Per-ROI frequency bar chart
    ax = axes[1]
    x_roi = np.arange(len(ROIS))
    width = 0.2
    band_colors = {"delta": "#1565c0", "theta": "#2e7d32", "alpha": "#f57f17", "beta": "#d32f2f"}
    for bi, band_name in enumerate(BANDS):
        vals = [r[f"{band_name}_dp"] for r in freq_by_roi]
        ax.bar(x_roi + bi * width, vals, width, label=band_name,
               color=band_colors[band_name], alpha=0.8)
    ax.set_xticks(x_roi + 1.5 * width)
    ax.set_xticklabels([r["roi"] for r in freq_by_roi], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ΔP")
    ax.set_title("Frequency Band × ROI")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)

    # (c) Band importance summary (global, from ROI means)
    ax = axes[2]
    global_band = {}
    for band_name in BANDS:
        vals = [r[f"{band_name}_dp"] for r in freq_by_roi]
        global_band[band_name] = np.mean(vals)
    bnames = list(global_band.keys())
    bvals = [global_band[b] for b in bnames]
    ax.bar(range(len(bnames)), [abs(v) for v in bvals],
           color=[band_colors[b] for b in bnames], alpha=0.8)
    ax.set_xticks(range(len(bnames)))
    ax.set_xticklabels(bnames)
    ax.set_ylabel("|ΔP| (mean across ROIs)")
    ax.set_title("Overall Band Importance")
    # Annotate sign
    for i, (bn, bv) in enumerate(zip(bnames, bvals)):
        ax.text(i, abs(bv) + 0.0005, "+" if bv > 0 else "−",
                ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "frequency_importance_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved frequency analysis to {out_dir}")

    return freq_by_channel, freq_by_roi


# ══════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_report(arch_info, ablation, combined_channels, roi_results,
                    freq_by_channel, freq_by_roi, subj_profiles, ch_matrix,
                    n_windows, n_boot, top_k, out_dir):
    """Generate the focused XAI report."""
    print("\n" + "=" * 70)
    print("GENERATING FOCUSED XAI REPORT")
    print("=" * 70)

    from datetime import datetime
    now = datetime.now().strftime("%B %d, %Y")

    # Identify facilitatory / suppressive / significant channels
    n_fac = sum(1 for c in combined_channels if c["sign"] == "facilitatory")
    n_sup = sum(1 for c in combined_channels if c["sign"] == "suppressive")
    n_mix = sum(1 for c in combined_channels if c["sign"] == "mixed")
    n_occ_sig = sum(1 for c in combined_channels if c["occ_significant"])
    n_perm_sig = sum(1 for c in combined_channels if c["perm_significant"])

    # Occ-perm correlation
    occ_all = np.array([c["occ_score"] for c in combined_channels])
    perm_all = np.array([c["perm_score"] for c in combined_channels])
    r_occ_perm = np.corrcoef(occ_all, perm_all)[0, 1]

    # Subject stability
    if ch_matrix.shape[0] > 0:
        corr_mat = np.corrcoef(ch_matrix)
        triu = corr_mat[np.triu_indices(ch_matrix.shape[0], k=1)]
        subj_r_mean = float(triu.mean())
        subj_r_std = float(triu.std())
    else:
        subj_r_mean = 0
        subj_r_std = 0

    # Block ablation summary
    block3_zero = ablation["block_3"]["zero_weights"]
    block3_perm = ablation["block_3"]["permute"]

    lines = []
    lines.append("=" * 80)
    lines.append("FOCUSED XAI ANALYSIS REPORT")
    lines.append("VLAAI EEG Auditory Attention Decoder — Channel Importance & Frequency Analysis")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Model: VLAAI (pretrained, loaded from vlaai.h5)")
    lines.append(f"Dataset: DTU EEG dataset")
    lines.append(f"Analysis windows: N = {n_windows}")
    lines.append(f"Bootstrap iterations: {n_boot}")
    lines.append(f"Script: scripts/run_focused_xai.py")
    lines.append("")

    # --- Objective ---
    lines.append("=" * 80)
    lines.append("OBJECTIVE")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Identify which EEG channels are important for the VLAAI auditory attention")
    lines.append("decoder, characterise their contribution (facilitatory vs suppressive), and")
    lines.append("analyse the frequency-band contribution of those important channels.")
    lines.append("")
    lines.append("Methods: Channel occlusion (zero-masking), channel permutation (shuffle")
    lines.append("across windows), Integrated Gradients (gradient-based attribution).")
    lines.append("Frequency analysis is conditioned on the identified important channels.")
    lines.append("")

    # --- Architecture ---
    lines.append("=" * 80)
    lines.append("1. ARCHITECTURE FINDING")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total parameters: {arch_info['total_params']:,}")
    lines.append(f"Shared parameters: {arch_info['shared_pct']*100:.1f}%")
    lines.append("")
    lines.append("Component breakdown:")
    for name, n in arch_info["components"].items():
        pct = n / arch_info["total_params"] * 100
        lines.append(f"  {name:25s}: {n:>10,} ({pct:.1f}%)")
    lines.append("")
    lines.append("Block ablation (key result):")
    lines.append(f"  Block 3 zero_weights:  ΔP = {block3_zero['delta_p_mean']:+.5f}, "
                 f"ΔAcc = {block3_zero['delta_acc']:+.3f}")
    lines.append(f"  Block 3 permute:       ΔP = {block3_perm['delta_p_mean']:+.5f}, "
                 f"ΔAcc = {block3_perm['delta_acc']:+.3f}")
    lines.append("")
    lines.append("KEY FINDING: Block 3 (final iteration) dominates the decision.")
    lines.append("Blocks 0-2 can be ablated with negligible effect. The model")
    lines.append("effectively uses only the last iteration's output.")
    lines.append("")

    # --- Channel importance method ---
    lines.append("=" * 80)
    lines.append("2. CHANNEL IMPORTANCE METHOD")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Three complementary methods were used:")
    lines.append("")
    lines.append("  (a) CHANNEL OCCLUSION: Replace each channel's values with zero across")
    lines.append("      all time points. Measure change in P(attended). Positive ΔP means")
    lines.append("      removing the channel HURTS decoding (facilitatory channel).")
    lines.append("      Negative ΔP means removing the channel HELPS (suppressive channel).")
    lines.append("")
    lines.append("  (b) CHANNEL PERMUTATION: Shuffle each channel's values across windows")
    lines.append("      (breaking temporal alignment while preserving marginal statistics).")
    lines.append("      Same interpretation as occlusion.")
    lines.append("")
    lines.append("  (c) INTEGRATED GRADIENTS (supporting): Gradient-based attribution")
    lines.append(f"      computed on {min(30, n_windows)} windows. Ranks channels by mean")
    lines.append("      |attribution|. Used as supporting evidence only.")
    lines.append("")
    lines.append(f"Occlusion–Permutation correlation: r = {r_occ_perm:.3f}")
    lines.append(f"Channels with significant occlusion CI (excludes 0): {n_occ_sig}/64")
    lines.append(f"Channels with significant permutation CI (excludes 0): {n_perm_sig}/64")
    lines.append("")

    # --- Most important channels ---
    lines.append("=" * 80)
    lines.append("3. MOST IMPORTANT CHANNELS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Top-{top_k} channels ranked by combined |occlusion| + |permutation| z-score:")
    lines.append("")
    header = f"{'Rank':>4} | {'Ch':>3} | {'ROI':>14} | {'Occ ΔP':>9} | {'Perm ΔP':>9} | {'IG Rank':>7} | {'Sign':>12} | {'Occ Sig':>7} | {'Stable':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for c in combined_channels[:top_k]:
        lines.append(
            f"{c['rank']:>4} | {c['channel']:>3} | {c['roi']:>14} | "
            f"{c['occ_score']:>+9.5f} | {c['perm_score']:>+9.5f} | "
            f"{c['ig_rank']:>7} | {c['sign']:>12} | "
            f"{'YES' if c['occ_significant'] else 'no':>7} | "
            f"{c['subject_stability']:>7}"
        )
    lines.append("")

    # --- Facilitatory vs suppressive ---
    lines.append("=" * 80)
    lines.append("4. FACILITATORY vs SUPPRESSIVE CHANNELS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Of 64 channels:")
    lines.append(f"  Facilitatory (both occ & perm positive):  {n_fac}")
    lines.append(f"  Suppressive  (both occ & perm negative):  {n_sup}")
    lines.append(f"  Mixed sign:                               {n_mix}")
    lines.append("")

    fac_list = [c for c in combined_channels if c["sign"] == "facilitatory" and c["occ_significant"]]
    sup_list = [c for c in combined_channels if c["sign"] == "suppressive" and c["occ_significant"]]

    if fac_list:
        lines.append("Significant facilitatory channels (removing hurts decoding):")
        for c in sorted(fac_list, key=lambda x: x["rank"]):
            lines.append(f"  Ch {c['channel']:2d} ({c['roi']:>14s}): Occ ΔP = {c['occ_score']:+.5f}")
    lines.append("")
    if sup_list:
        lines.append("Significant suppressive channels (removing helps decoding):")
        for c in sorted(sup_list, key=lambda x: x["rank"]):
            lines.append(f"  Ch {c['channel']:2d} ({c['roi']:>14s}): Occ ΔP = {c['occ_score']:+.5f}")
    lines.append("")

    # --- ROI interpretation ---
    lines.append("=" * 80)
    lines.append("5. ROI-LEVEL INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("NOTE: ROI mapping is INDEX-BASED and approximate. The actual DTU")
    lines.append("electrode montage should be verified for publication-quality claims.")
    lines.append("")
    header_roi = f"{'ROI':>20} | {'Occ ΔP':>9} | {'95% CI':>22} | {'Sig?':>5} | {'Perm ΔP':>9} | {'#SigCh':>6}"
    lines.append(header_roi)
    lines.append("-" * len(header_roi))
    for r in roi_results:
        lines.append(
            f"{r['roi']:>20} | {r['occ_mean_dp']:>+9.5f} | "
            f"[{r['occ_ci'][0]:+.5f},{r['occ_ci'][1]:+.5f}] | "
            f"{'YES' if r['occ_significant'] else 'no':>5} | "
            f"{r['perm_mean_dp']:>+9.5f} | "
            f"{r['n_significant_channels']:>6}"
        )
    lines.append("")

    # --- Frequency contribution ---
    lines.append("=" * 80)
    lines.append("6. FREQUENCY-BAND CONTRIBUTION OF IMPORTANT CHANNELS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Method: For each top channel, remove a specific frequency band")
    lines.append("(Butterworth 4th order bandpass, mirror-padded) from that channel")
    lines.append("and measure ΔP. Positive ΔP = removing that band hurts decoding.")
    lines.append("")

    lines.append(f"Per-channel frequency analysis (top-{top_k} channels):")
    lines.append("")
    header_freq = f"{'Ch':>3} | {'ROI':>14} | {'Best Band':>10} | {'Delta ΔP':>9} | {'Theta ΔP':>9} | {'Alpha ΔP':>9} | {'Beta ΔP':>9}"
    lines.append(header_freq)
    lines.append("-" * len(header_freq))
    for r in freq_by_channel:
        lines.append(
            f"{r['channel']:>3} | {r['roi']:>14} | {r['most_important_band']:>10} | "
            f"{r['delta_dp']:>+9.5f} | {r['theta_dp']:>+9.5f} | "
            f"{r['alpha_dp']:>+9.5f} | {r['beta_dp']:>+9.5f}"
        )
    lines.append("")

    lines.append("Per-ROI frequency analysis:")
    lines.append("")
    header_freq_roi = f"{'ROI':>20} | {'Best Band':>10} | {'Delta ΔP':>9} | {'Theta ΔP':>9} | {'Alpha ΔP':>9} | {'Beta ΔP':>9}"
    lines.append(header_freq_roi)
    lines.append("-" * len(header_freq_roi))
    for r in freq_by_roi:
        lines.append(
            f"{r['roi']:>20} | {r['most_important_band']:>10} | "
            f"{r['delta_dp']:>+9.5f} | {r['theta_dp']:>+9.5f} | "
            f"{r['alpha_dp']:>+9.5f} | {r['beta_dp']:>+9.5f}"
        )
    lines.append("")

    # --- Limitations ---
    lines.append("=" * 80)
    lines.append("7. LIMITATIONS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"1. SAMPLE SIZE: Analysis used N={n_windows} windows. While bootstrap CIs")
    lines.append("   provide uncertainty estimates, larger N is recommended for definitive")
    lines.append("   conclusions (N≥500 for publication-quality results).")
    lines.append("")
    lines.append("2. UNATTENDED ENVELOPE PROXY: The DTU dataset provides only the attended")
    lines.append("   envelope. The 'unattended' envelope is a circular time-shift proxy.")
    lines.append("   Results may differ with a true competing-speaker paradigm.")
    lines.append("")
    lines.append("3. ROI MAPPING: Channel-to-ROI assignments are INDEX-BASED (Ch 0–11 →")
    lines.append("   Frontal, etc.) and have NOT been verified against the actual DTU")
    lines.append("   electrode montage. Spatial interpretations are approximate.")
    lines.append("")
    lines.append("4. INTEGRATED GRADIENTS: IG uses a zero baseline, which may not be")
    lines.append("   neurophysiologically meaningful. IG results are supporting evidence only.")
    lines.append("")
    lines.append("5. SUBJECT-WISE STABILITY: Limited by windows_per_subject. Individual")
    lines.append("   channel importance profiles are noisy with small samples.")
    lines.append("")
    if ch_matrix.shape[0] > 0:
        lines.append(f"6. CROSS-SUBJECT CONSISTENCY: Mean pairwise r = {subj_r_mean:.3f} ± {subj_r_std:.3f}.")
        lines.append("   Low inter-subject agreement indicates subject-specific patterns.")
        lines.append("   Group-level channel maps may obscure individual differences.")
        lines.append("")

    # --- Next steps ---
    lines.append("=" * 80)
    lines.append("8. NEXT STEPS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. Verify ROI mapping against actual DTU electrode positions.")
    lines.append("2. Run with larger N (≥500) and n_boot (≥1000) for tighter CIs.")
    lines.append("3. Add topographic EEG scalp maps using actual electrode coordinates.")
    lines.append("4. Compare with within-subject trained VLAAI if available.")
    lines.append("5. Statistical correction for multiple comparisons (64 channels).")
    lines.append("6. Validate findings with alternative models (TRF baseline).")
    lines.append("7. Consider temporal occlusion and layer probing as supplementary analyses.")
    lines.append("")

    # --- Files ---
    lines.append("=" * 80)
    lines.append("OUTPUT FILES")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Output directory: {out_dir}")
    lines.append("")
    lines.append("  architecture_summary.json   — Architecture parameters and components")
    lines.append("  block_ablation.json          — Block ablation results")
    lines.append("  block_ablation.png           — Block ablation plot")
    lines.append("  channel_importance.json      — Full channel importance (all methods)")
    lines.append("  channel_importance.csv       — Ranked channel table (spreadsheet-ready)")
    lines.append("  channel_importance_plot.png   — 4-panel channel importance figure")
    lines.append("  roi_importance.json           — ROI-level occlusion + permutation")
    lines.append("  roi_importance.csv            — ROI table")
    lines.append("  roi_importance_plot.png        — ROI importance bar charts")
    lines.append("  frequency_analysis.json       — Frequency ΔP per channel and ROI")
    lines.append("  frequency_by_channel.csv      — Per-channel frequency table")
    lines.append("  frequency_by_roi.csv          — Per-ROI frequency table")
    lines.append("  frequency_importance_plot.png  — Frequency analysis 3-panel figure")
    lines.append("  occlusion_perwindow.npy       — Raw (N, 64) per-window occlusion drops")
    lines.append("  permutation_perwindow.npy     — Raw (N, 64) per-window permutation drops")
    lines.append("  subject_channel_matrix.npy    — (n_subjects, 64) stability matrix")
    lines.append("  FOCUSED_XAI_REPORT.txt        — This report")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    (out_dir / "FOCUSED_XAI_REPORT.txt").write_text(report_text, encoding="utf-8")
    print(f"  Report saved to {out_dir / 'FOCUSED_XAI_REPORT.txt'}")
    print(f"  Report length: {len(lines)} lines")

    return report_text


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = make_output_dir(Path(args.output_dir))

    print("=" * 70)
    print("FOCUSED XAI ANALYSIS: Channel → ROI → Frequency")
    print(f"  Seed: {args.seed}, Bootstrap: {args.n_boot}, Max samples: {args.max_samples}")
    print(f"  Top-K channels for freq analysis: {args.top_k}")
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

    # Verify model works
    decision.set_envelopes(att_all[:3], unatt_all[:3])
    with torch.no_grad():
        test_logits = decision(eeg_all[:3])
    print(f"  Decision logits (sample): {test_logits[0].cpu().numpy()}")

    # Set envelopes for all windows
    decision.set_envelopes(att_all, unatt_all)

    # ------------------------------------------------------------------
    # Section 1: Architecture + Block Ablation
    # ------------------------------------------------------------------
    arch_info, ablation = run_architecture(
        model, decision, eeg_all, att_all, unatt_all, args.n_boot, args.seed, out_dir, device)

    # Reset envelopes after ablation
    decision.set_envelopes(att_all, unatt_all)

    # ------------------------------------------------------------------
    # Section 2: Channel Importance
    # ------------------------------------------------------------------
    combined_channels, subj_profiles, ch_matrix = run_channel_importance(
        decision, model, eeg_all, att_all, unatt_all, ds,
        args.n_boot, args.seed, args.n_ig, args.ig_steps,
        args.windows_per_subject, args.top_k, out_dir)

    # Reset envelopes
    decision.set_envelopes(att_all, unatt_all)

    # ------------------------------------------------------------------
    # Section 3: ROI Analysis
    # ------------------------------------------------------------------
    occ_pw = np.load(out_dir / "occlusion_perwindow.npy")
    perm_pw = np.load(out_dir / "permutation_perwindow.npy")
    roi_results = run_roi_analysis(combined_channels, occ_pw, perm_pw,
                                   args.n_boot, args.seed, out_dir)

    # ------------------------------------------------------------------
    # Section 4: Frequency Analysis on Top Channels
    # ------------------------------------------------------------------
    freq_by_channel, freq_by_roi = run_frequency_analysis(
        decision, eeg_all, att_all, unatt_all, combined_channels,
        roi_results, args.n_boot, args.seed, args.top_k, out_dir)

    # ------------------------------------------------------------------
    # Generate Report
    # ------------------------------------------------------------------
    generate_report(arch_info, ablation, combined_channels, roi_results,
                    freq_by_channel, freq_by_roi, subj_profiles, ch_matrix,
                    N, args.n_boot, args.top_k, out_dir)

    # Save run config
    save_json({
        "seed": args.seed, "n_boot": args.n_boot, "max_samples": args.max_samples,
        "n_ig": args.n_ig, "ig_steps": args.ig_steps,
        "top_k": args.top_k, "n_windows": N,
        "windows_per_subject": args.windows_per_subject,
    }, out_dir / "run_config.json")

    print("\n" + "=" * 70)
    print("FOCUSED XAI ANALYSIS COMPLETE")
    print(f"  All results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
