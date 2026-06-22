"""
generate_publication_xai_figures.py

Generate publication-ready figures from the focused VLAAI XAI analysis results.

Run from the project root directory:
    python scripts/generate_publication_xai_figures.py

Input:  Kaggle_results/Full run results/
Output: xai_results/publication_figures/
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate publication-ready XAI figures from focused VLAAI analysis results."
    )
    p.add_argument(
        "--input-dir",
        default="Kaggle_results/Full run results",
        help="Directory containing the XAI result files (default: local Kaggle_results folder).",
    )
    p.add_argument(
        "--output-dir",
        default="xai_results/publication_figures",
        help="Directory to write figures and summary files into.",
    )
    return p.parse_args()


_args = _parse_args()

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_DIR  = Path(_args.input_dir)
OUTPUT_DIR = Path(_args.output_dir)

FIG_DPI = 300
FONT_SIZE = 12
TITLE_SIZE = 13
LABEL_SIZE = 11
TICK_SIZE = 10

# Colour palette
COLOR_OCC   = "#2166AC"   # blue   — occlusion
COLOR_PERM  = "#D6604D"   # orange — permutation
COLOR_FACIL = "#1A7A4A"   # green  — facilitatory
COLOR_SUPP  = "#C2382A"   # red    — suppressive
COLOR_CORE  = "#2166AC"   # blue   — core channels
COLOR_DIST  = "#92C5DE"   # light  — distributed channels

STRONG_ROIS = {"Fronto-Central", "Temporal", "Central", "Centro-Parietal"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ── Helpers ────────────────────────────────────────────────────────────────────

def save_fig(fig, stem):
    for ext in ("png", "pdf"):
        path = OUTPUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {stem}.png / .pdf")


def load_csv(name, required=False):
    path = INPUT_DIR / name
    if not path.exists():
        msg = f"WARNING: Missing input file: {path}"
        if required:
            print(msg)
            sys.exit(1)
        print(msg)
        return None
    df = pd.read_csv(path)
    print(f"  Loaded: {name} ({len(df)} rows, {len(df.columns)} cols)")
    return df


def load_npy(name):
    path = INPUT_DIR / name
    if not path.exists():
        print(f"WARNING: Missing input file: {path}")
        return None
    arr = np.load(path)
    print(f"  Loaded: {name}  shape={arr.shape}")
    return arr


def stability_to_float(s):
    """Convert '12/18' → 0.667."""
    try:
        num, den = str(s).split("/")
        return int(num) / int(den)
    except Exception:
        return np.nan


# ── Figure 1: Top-15 Channel Importance ───────────────────────────────────────

def fig1_top15(df_robust):
    top15 = df_robust.head(15).copy()
    x = np.arange(len(top15))
    w = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, top15["occlusion_delta_p"] * 1e3,
           width=w, color=COLOR_OCC, label="Occlusion ΔP", zorder=3)
    ax.bar(x + w / 2, top15["permutation_delta_p"] * 1e3,
           width=w, color=COLOR_PERM, label="Permutation ΔP", alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(top15["electrode_name"], fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Importance (ΔP × 10³)", fontsize=LABEL_SIZE)
    ax.set_title(
        "Figure 1: Top-15 Robust EEG Channels — VLAAI Auditory Attention Decoder\n"
        "Occlusion ΔP and permutation ΔP shown side by side; sorted by combined importance.",
        fontsize=TITLE_SIZE, fontweight="bold", pad=10,
    )
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.legend(fontsize=LABEL_SIZE, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # ROI annotation below x-tick labels
    for i, (_, row) in enumerate(top15.iterrows()):
        ax.text(
            i, -0.07, row["roi"],
            ha="center", va="top", fontsize=8, color="gray",
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xlabel("EEG Channel  (ROI shown below)", fontsize=LABEL_SIZE)

    fig.tight_layout()
    save_fig(fig, "fig1_top15_channel_importance")
    plt.close(fig)


# ── Figure 2: Core vs Distributed ─────────────────────────────────────────────

def fig2_core_vs_distributed(df_robust):
    top15 = df_robust.head(15).copy()
    rest  = df_robust.iloc[15:].copy()

    combined_core = (top15["occlusion_delta_p"].abs() + top15["permutation_delta_p"].abs()) / 2
    combined_rest = (rest["occlusion_delta_p"].abs()  + rest["permutation_delta_p"].abs())  / 2

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [2, 3]})

    # Left — core
    ax = axes[0]
    y_c = np.arange(len(top15))
    ax.barh(y_c, combined_core.values * 1e3, color=COLOR_CORE, height=0.65, zorder=3)
    ax.set_yticks(y_c)
    ax.set_yticklabels(top15["electrode_name"].tolist(),
                       fontsize=LABEL_SIZE, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |ΔP| × 10³", fontsize=LABEL_SIZE)
    ax.set_title("Core Channels (Top 15)\nStrongest practical effect",
                 fontsize=TITLE_SIZE - 1, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, zorder=0)

    # Dashed line at max of distributed channels for scale reference
    max_dist = float(combined_rest.max() * 1e3)
    ax.axvline(max_dist, linestyle="--", color="gray", linewidth=1.1,
               label=f"Max distributed = {max_dist:.3f}")
    ax.legend(fontsize=9, framealpha=0.9)

    # Right — distributed
    ax2 = axes[1]
    y_r = np.arange(len(rest))
    ax2.barh(y_r, combined_rest.values * 1e3, color=COLOR_DIST, height=0.65, zorder=3)
    ax2.set_yticks(y_r)
    ax2.set_yticklabels(rest["electrode_name"].tolist(), fontsize=TICK_SIZE)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean |ΔP| × 10³", fontsize=LABEL_SIZE)
    ax2.set_title(
        f"Distributed Channels (Robust ranks 16–{len(df_robust)})\n"
        "Statistically robust but small effect size",
        fontsize=TITLE_SIZE - 1, fontweight="bold",
    )
    ax2.grid(axis="x", alpha=0.3, zorder=0)

    fig.suptitle(
        "Figure 2: Core vs. Distributed Channel Importance\n"
        "45/64 channels met robust criteria; the core top-15 show the strongest practical contributions.",
        fontsize=TITLE_SIZE, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "fig2_core_vs_distributed_channels")
    plt.close(fig)


# ── Figure 3: ROI-Level Importance ────────────────────────────────────────────

def fig3_roi(df_roi):
    df_roi = df_roi.sort_values("occ_mean_dp", ascending=False).reset_index(drop=True)
    rois = df_roi["roi"].tolist()
    occ  = df_roi["occ_mean_dp"].values * 1e3
    lo   = (df_roi["occ_mean_dp"] - df_roi["occ_ci_lo"]).values * 1e3
    hi   = (df_roi["occ_ci_hi"]   - df_roi["occ_mean_dp"]).values * 1e3
    perm = df_roi["perm_mean_dp"].values * 1e3

    x = np.arange(len(rois))
    colors = [COLOR_CORE if r in STRONG_ROIS else COLOR_DIST for r in rois]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, occ, color=colors, width=0.55, zorder=3)
    ax.errorbar(x, occ, yerr=[lo, hi],
                fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)
    ax.scatter(x, perm, color=COLOR_PERM, marker="D", s=65, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(rois, fontsize=LABEL_SIZE, rotation=30, ha="right")
    ax.set_ylabel("ROI Importance (ΔP × 10³)", fontsize=LABEL_SIZE)
    ax.set_title(
        "Figure 3: ROI-Level Channel Importance\n"
        "Occlusion ΔP (bars) with 95% bootstrap CI; permutation ΔP (◆).\n"
        "Dark blue bars: Fronto-Central, Temporal, Central, Centro-Parietal.",
        fontsize=TITLE_SIZE, fontweight="bold", pad=10,
    )
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    handles = [
        mpatches.Patch(color=COLOR_CORE, label="Strongest ROIs"),
        mpatches.Patch(color=COLOR_DIST, label="Other ROIs"),
        plt.Line2D([0], [0], marker="D", color=COLOR_PERM, linestyle="none",
                   markersize=8, label="Permutation ΔP"),
        plt.Line2D([0], [0], color="black", linewidth=1.5,
                   label="95% bootstrap CI"),
    ]
    ax.legend(handles=handles, fontsize=LABEL_SIZE - 1, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, "fig3_roi_importance")
    plt.close(fig)


# ── Figure 4: Facilitatory vs Suppressive ─────────────────────────────────────

def fig4_facil_supp(df_robust):
    facil = df_robust[df_robust["contribution_type"] == "facilitatory"].sort_values(
        "occlusion_delta_p", ascending=False
    ).copy()
    supp  = df_robust[df_robust["contribution_type"] == "suppressive"].sort_values(
        "occlusion_delta_p", ascending=True
    ).copy()

    n_f, n_s = len(facil), len(supp)
    h = max(6, max(n_f, n_s) * 0.30 + 2)
    fig, axes = plt.subplots(1, 2, figsize=(16, h))

    w_main = 0.55
    w_perm = 0.25

    def _dual_bars(ax, df, main_color, perm_label_side, title_txt, xlabel_txt):
        y = np.arange(len(df))
        occ_vals  = df["occlusion_delta_p"].values  * 1e3
        perm_vals = df["permutation_delta_p"].values * 1e3
        ax.barh(y, occ_vals,  height=w_main, color=main_color,
                label="Occlusion ΔP", zorder=3)
        ax.barh(y, perm_vals, height=w_perm, color=main_color,
                alpha=0.45, label="Permutation ΔP", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(df["electrode_name"].tolist(), fontsize=TICK_SIZE)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel_txt, fontsize=LABEL_SIZE)
        ax.set_title(title_txt, fontsize=TITLE_SIZE - 1, fontweight="bold")
        ax.legend(fontsize=LABEL_SIZE - 1, loc=perm_label_side, framealpha=0.9)
        ax.grid(axis="x", alpha=0.3, zorder=0)

    _dual_bars(
        axes[0], facil, COLOR_FACIL, "lower right",
        f"Facilitatory Channels (n={n_f})\nRemoving hurts decoding",
        "ΔP × 10³  (positive = facilitatory)",
    )
    _dual_bars(
        axes[1], supp, COLOR_SUPP, "lower left",
        f"Suppressive Channels (n={n_s})\nRemoving helps decoding",
        "ΔP × 10³  (negative = suppressive)",
    )

    fig.suptitle(
        "Figure 4: Facilitatory vs. Suppressive Robust Channels\n"
        "Facilitatory: both occlusion & permutation ΔP > 0  |  "
        "Suppressive: both ΔP < 0",
        fontsize=TITLE_SIZE, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "fig4_facilitatory_suppressive_channels")
    plt.close(fig)


# ── Figure 5: Subject-Specificity ─────────────────────────────────────────────

def fig5_subject_specificity(df_robust, subj_matrix):
    top15_idx   = df_robust.head(15)["channel_index"].tolist()
    top15_names = df_robust.head(15)["electrode_name"].tolist()

    if subj_matrix is not None and subj_matrix.ndim == 2 and subj_matrix.shape[1] >= max(top15_idx) + 1:
        mat = subj_matrix[:, top15_idx]          # (n_subjects, 15)
        n_subj = mat.shape[0]
        subj_labels = [f"S{i+1}" for i in range(n_subj)]

        # Row-normalise so per-subject pattern is visible
        row_max = np.max(np.abs(mat), axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        mat_norm = mat / row_max

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left — channels × subjects heatmap
        ax = axes[0]
        im = ax.imshow(mat_norm.T, aspect="auto", cmap="RdBu_r",
                       vmin=-1, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(top15_names)))
        ax.set_yticklabels(top15_names, fontsize=TICK_SIZE)
        ax.set_xticks(range(n_subj))
        ax.set_xticklabels(subj_labels, fontsize=TICK_SIZE, rotation=45, ha="right")
        ax.set_xlabel("Subject", fontsize=LABEL_SIZE)
        ax.set_ylabel("EEG Channel (Top 15)", fontsize=LABEL_SIZE)
        ax.set_title("Subject × Channel Importance\n(row-normalised per subject)",
                     fontsize=TITLE_SIZE - 1, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Relative importance", fontsize=LABEL_SIZE - 1)

        # Right — pairwise Spearman ρ heatmap
        ax2 = axes[1]
        try:
            from scipy.stats import spearmanr
            corr_mat = np.eye(n_subj)
            for i in range(n_subj):
                for j in range(i + 1, n_subj):
                    r, _ = spearmanr(mat[i], mat[j])
                    corr_mat[i, j] = corr_mat[j, i] = float(r)
            mean_r = float(np.mean(corr_mat[np.triu_indices(n_subj, k=1)]))
        except ImportError:
            warnings.warn("scipy not available — using Pearson r for subject similarity.")
            corr_mat = np.corrcoef(mat)
            mean_r = float(np.mean(corr_mat[np.triu_indices(n_subj, k=1)]))

        im2 = ax2.imshow(corr_mat, aspect="auto", cmap="RdBu_r",
                         vmin=-1, vmax=1, interpolation="nearest")
        ticks = list(range(n_subj))
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(subj_labels, fontsize=TICK_SIZE, rotation=45, ha="right")
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(subj_labels, fontsize=TICK_SIZE)
        ax2.set_title("Pairwise Spearman ρ\n(channel importance profiles, top-15 channels)",
                      fontsize=TITLE_SIZE - 1, fontweight="bold")
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.04)
        cbar2.set_label("Spearman ρ", fontsize=LABEL_SIZE - 1)
        ax2.text(
            0.5, -0.18,
            f"Mean pairwise ρ = {mean_r:.3f} — low group-level consistency.\n"
            "Individual explanations may differ substantially from the group-level map.",
            transform=ax2.transAxes, ha="center", fontsize=9, color="dimgray",
            style="italic",
        )
    else:
        # Fallback — subject stability bar chart
        if subj_matrix is not None:
            print("  WARNING: subject_channel_matrix shape unexpected; "
                  "falling back to stability fraction plot.")
        stab = df_robust.head(15).copy()
        stab["stab_frac"] = stab["subject_stability"].apply(stability_to_float)

        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(stab))
        ax.bar(x, stab["stab_frac"], color=COLOR_CORE, zorder=3)
        ax.axhline(0.5, linestyle="--", color="gray", linewidth=1.2,
                   label="50% stability threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(stab["electrode_name"], fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Subject stability (fraction of subjects agreeing)", fontsize=LABEL_SIZE)
        ax.set_title(
            "Figure 5: Subject Stability of Top-15 Channels\n"
            "(subject_channel_matrix.npy unavailable — showing stability fraction)",
            fontsize=TITLE_SIZE, fontweight="bold",
        )
        ax.legend(fontsize=LABEL_SIZE - 1)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.text(
            0.5, -0.18,
            "Note: Group-level maps may obscure individual differences in auditory attention decoding.",
            transform=ax.transAxes, ha="center", fontsize=9, color="dimgray", style="italic",
        )
        fig.tight_layout()
        save_fig(fig, "fig5_subject_specificity")
        plt.close(fig)
        return

    fig.suptitle(
        "Figure 5: Subject-Specificity of EEG Channel Importance\n"
        "Group-level maps may obscure meaningful individual differences.",
        fontsize=TITLE_SIZE, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig5_subject_specificity")
    plt.close(fig)


# ── Figure 6: Exploratory Frequency Contribution ──────────────────────────────

def fig6_frequency(df_freq, df_robust):
    top15_names = df_robust.head(15)["electrode_name"].tolist()
    df_f = df_freq[df_freq["electrode_name"].isin(top15_names)].copy()

    # Restore robust ranking order
    order_map = {n: i for i, n in enumerate(top15_names)}
    df_f["_ord"] = df_f["electrode_name"].map(order_map)
    df_f = df_f.sort_values("_ord").reset_index(drop=True)

    if df_f.empty:
        print("  WARNING: no frequency data found for top-15 channels -- skipping fig6.")
        return

    bands        = ["delta", "theta", "alpha", "beta"]
    cols         = [f"{b}_dp" for b in bands]
    band_colors  = ["#FFD700", "#74ADD1", "#ABDDA4", "#F46D43"]
    band_hatches = ["//", None, None, None]   # delta hatched = interpret cautiously

    x = np.arange(len(df_f))
    n = len(bands)
    w = 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (band, col, color, hatch) in enumerate(zip(bands, cols, band_colors, band_hatches)):
        vals = df_f[col].values * 1e3
        ax.bar(
            x + offsets[i], vals, width=w,
            color=color, hatch=hatch, label=band.capitalize(),
            edgecolor="gray", linewidth=0.5, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df_f["electrode_name"], fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Frequency-band contribution (ΔP × 10³)", fontsize=LABEL_SIZE)
    ax.set_title(
        "Figure 6 [EXPLORATORY]: Frequency-Band Contribution — Top-15 Robust Channels\n"
        "Delta bars are hatched (//); delta-band effects should be interpreted cautiously "
        "due to short analysis window.",
        fontsize=TITLE_SIZE, fontweight="bold", pad=10,
    )
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.legend(title="Band  (Delta = // = interpret cautiously)",
              fontsize=LABEL_SIZE - 1, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    ax.text(
        0.5, -0.21,
        "EXPLORATORY NOTE: Frequency-band interpretation, especially delta-band effects (hatched bars),\n"
        "should be treated cautiously because the analysis window is short.",
        transform=ax.transAxes, ha="center", fontsize=9, color="#8B0000",
        style="italic",
        bbox=dict(facecolor="lightyellow", edgecolor="gray", boxstyle="round,pad=0.3"),
    )
    fig.tight_layout()
    save_fig(fig, "fig6_exploratory_frequency_contribution")
    plt.close(fig)


# ── Summary CSV ────────────────────────────────────────────────────────────────

def make_summary_table(df_robust, df_freq):
    out = df_robust.copy()
    out["channel_name"]       = out["electrode_name"]
    out["combined_importance"] = (
        out["occlusion_delta_p"].abs() + out["permutation_delta_p"].abs()
    ) / 2
    out["robust_status"] = out["robust_significant"].apply(
        lambda v: "robust" if bool(v) else "not_robust"
    )

    # Enrich with frequency best-band if a richer freq CSV is available
    if df_freq is not None and "most_important_band" in df_freq.columns:
        freq_map = df_freq.set_index("electrode_name")["most_important_band"].to_dict()
        if "best_frequency_band" not in out.columns:
            out["best_frequency_band"] = out["channel_name"].map(freq_map)
        else:
            out["best_frequency_band"] = out["best_frequency_band"].fillna(
                out["channel_name"].map(freq_map)
            )

    ordered_cols = [
        "channel_name", "channel_index", "roi",
        "occlusion_delta_p", "permutation_delta_p", "combined_importance",
        "contribution_type", "robust_status", "subject_stability",
        "best_frequency_band",
    ]
    available = [c for c in ordered_cols if c in out.columns]
    df_out = out[available]

    path = OUTPUT_DIR / "publication_xai_summary.csv"
    df_out.to_csv(path, index=False)
    print(f"  Saved: publication_xai_summary.csv  ({len(df_out)} rows, "
          f"{len(df_out.columns)} cols)")


# ── Interpretation text ────────────────────────────────────────────────────────

INTERPRETATION = """\
PUBLICATION XAI INTERPRETATION SUMMARY
VLAAI EEG Auditory Attention Decoder — DTU Dataset
Generated from focused XAI analysis (N = 8100 windows, 18 subjects)
================================================================================

1. DISTRIBUTED REPRESENTATION

The VLAAI model is consistent with a distributed EEG representation for
auditory attention decoding. Of 64 EEG channels tested, 45 met conservative
robust significance criteria (sign agreement across occlusion and permutation,
at least one FDR-significant method, and subject-wise stability ≥ 50%). This
suggests the decoder does not rely exclusively on a small set of electrodes,
but rather draws from a broad spatial pattern across the scalp.

2. STRONGEST PRACTICAL EFFECTS

Although 45 channels were statistically robust, effect sizes vary considerably.
The strongest practical effects are concentrated in central, fronto-central,
temporal, and centro-parietal regions. Channels C4, FC4, Cz, CP4, and FT8
showed the largest combined occlusion and permutation importance scores, with
the remaining robust channels showing statistically reliable but substantially
smaller magnitudes.

Interpretation should focus primarily on the top 10–15 channels, where
practical effect sizes are most meaningful. The full set of 45 robust channels
is consistent with distributed processing but should not be interpreted as
indicating that all channels contribute equally or strongly.

3. FACILITATORY AND SUPPRESSIVE CONTRIBUTIONS

The majority of robust channels are facilitatory (removing them hurts
decoding), suggesting they carry information used by the model. A smaller
subset of channels showed robust suppressive effects (removing them improves
decoding), which may indicate channels carrying noise, reference artifacts,
or competing neural information not relevant to auditory attention. These
suppressive effects are small in magnitude and should be interpreted cautiously.

4. ROI-LEVEL FINDINGS

At the region-of-interest level, the fronto-central and central ROIs showed
the largest mean ΔP values with tight confidence intervals. Temporal and
centro-parietal ROIs also showed robust positive effects. The frontal ROI as
a whole did not reach significance by occlusion alone (CI includes zero),
though individual channels within it were robust by combined criteria.

5. FREQUENCY-BAND CONTRIBUTIONS (EXPLORATORY)

Frequency-band analysis was conducted by selectively removing individual
bands from robust channels and measuring the resulting change in decoding
accuracy. Results are consistent with contributions from delta, theta, and
beta-band activity in many of the strongest channels. However, these findings
should be considered exploratory:

  - Delta-band effects should be interpreted cautiously because the analysis
    window is short, limiting frequency resolution at low frequencies.
  - Theta and beta effects were observed in several top channels (e.g.,
    C4 best band: theta; Cz best band: beta), which is broadly consistent
    with known EEG correlates of attention, but cannot be treated as
    confirmatory evidence from this analysis alone.
  - Exploratory frequency evidence suggests, rather than proves, that
    these bands are relevant to the model's representations.

6. SUBJECT-LEVEL VARIABILITY

Subject-specificity analysis showed low group-level consistency in channel
importance profiles (mean pairwise correlation r ≈ 0.037 ± 0.262 across
subjects). This indicates that individual channel importance maps differ
substantially from the group-level map. Individual explanations may diverge
meaningfully from the aggregate results presented here, and the group-level
map should be interpreted as a population tendency rather than a universal
finding applicable to all individuals.

7. ARCHITECTURE: BLOCK 3 DOMINANCE

Architecture ablation showed that Block 3 (the final iteration of the VLAAI
recurrent structure) dominates the decoding decision (Block 3 zero-weight
ablation: ΔP = +0.02652, ΔAcc = +0.624). Blocks 0–2 can be ablated with
negligible effect on performance. This is consistent with the model's final
iteration integrating the contextual information most relevant for the
attention decision, though the underlying mechanism warrants further
investigation.

================================================================================
BASELINE PERFORMANCE

  AAD accuracy:              0.646
  Correlation margin (mean): +0.10830
  Correlation margin (CI):   [+0.10207, +0.11438]

================================================================================
KEY LIMITATIONS

  - Baseline accuracy of 0.646 reflects modest but above-chance performance.
  - The DTU unattended envelope is a circular time-shift proxy, not a true
    competing-speaker signal; results may differ in a real two-talker paradigm.
  - Electrode montage assignments should be verified against the official
    DTU dataset documentation before publication-quality conclusions are made.
  - All ΔP effect sizes are small in absolute magnitude. Statistical
    robustness does not imply large practical importance.
  - IG attributions use a zero baseline, which may not be neurophysiologically
    meaningful; IG results were used as corroborating evidence only.

================================================================================
CONSERVATIVE LANGUAGE GUIDE

  Use:   "suggests", "is consistent with",
         "showed statistically robust but small-magnitude effects",
         "should be interpreted cautiously",
         "exploratory frequency-band evidence"

  Avoid: "proves", "definitively shows",
         "the model definitely uses delta-band neural activity",
         "all 45 channels are strongly important"

================================================================================
"""


def make_interpretation():
    path = OUTPUT_DIR / "publication_xai_interpretation.txt"
    path.write_text(INTERPRETATION, encoding="utf-8")
    print(f"  Saved: publication_xai_interpretation.txt")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nOutput directory : " + str(OUTPUT_DIR.resolve()))
    print("Input directory  : " + str(INPUT_DIR.resolve()) + "\n")

    # --- Load inputs ---
    print("Loading input files ...")
    df_robust = load_csv("final_important_channels.csv", required=True)
    df_roi    = load_csv("roi_importance.csv")
    df_freq   = load_csv("frequency_by_channel.csv")
    _         = load_csv("channel_importance.csv")
    subj_mat  = load_npy("subject_channel_matrix.npy")

    generated, skipped = [], []

    # --- Figures ---
    print("\n--- Figure 1: Top-15 Channel Importance ---")
    fig1_top15(df_robust)
    generated.append("fig1_top15_channel_importance")

    print("\n--- Figure 2: Core vs Distributed ---")
    fig2_core_vs_distributed(df_robust)
    generated.append("fig2_core_vs_distributed_channels")

    print("\n--- Figure 3: ROI-Level Importance ---")
    if df_roi is not None:
        fig3_roi(df_roi)
        generated.append("fig3_roi_importance")
    else:
        print("  SKIPPED -- roi_importance.csv missing.")
        skipped.append("fig3_roi_importance")

    print("\n--- Figure 4: Facilitatory vs Suppressive ---")
    fig4_facil_supp(df_robust)
    generated.append("fig4_facilitatory_suppressive_channels")

    print("\n--- Figure 5: Subject-Specificity ---")
    fig5_subject_specificity(df_robust, subj_mat)
    generated.append("fig5_subject_specificity")

    print("\n--- Figure 6: Frequency Contribution (Exploratory) ---")
    if df_freq is not None:
        fig6_frequency(df_freq, df_robust)
        generated.append("fig6_exploratory_frequency_contribution")
    else:
        print("  SKIPPED -- frequency_by_channel.csv missing.")
        skipped.append("fig6_exploratory_frequency_contribution")

    print("\n--- Summary Table ---")
    make_summary_table(df_robust, df_freq)

    print("\n--- Interpretation Text ---")
    make_interpretation()

    # --- Report ---
    print("\n" + "=" * 64)
    print("Generated %d figure(s):" % len(generated))
    for name in generated:
        print("  [OK]  %s.png / .pdf" % name)
    if skipped:
        print("\nSkipped %d figure(s) (missing input files):" % len(skipped))
        for name in skipped:
            print("  [--]  %s" % name)
    print("\nAll outputs -> " + str(OUTPUT_DIR.resolve()))
    print("=" * 64)


if __name__ == "__main__":
    main()
