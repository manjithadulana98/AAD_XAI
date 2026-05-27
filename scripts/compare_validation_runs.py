"""Compare focused XAI results across multiple runs at different dataset sizes.

Reads outputs from multiple run folders and generates comparison metrics,
overlap tables, and stability plots.

Usage:
    python scripts/compare_validation_runs.py \
      --runs xai_results_focused xai_results_focused_n500 \
      --output-dir xai_results_validation_comparison

    python scripts/compare_validation_runs.py \
      --runs xai_results_focused xai_results_focused_n500 \
             xai_results_focused_n1000 xai_results_focused_full \
      --output-dir xai_results_validation_comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Compare focused XAI runs at different N")
    p.add_argument("--runs", nargs="+", required=True,
                   help="Paths to run output directories")
    p.add_argument("--output-dir", type=str,
                   default=str(ROOT / "xai_results_validation_comparison"))
    p.add_argument("--high-confidence-min-runs", type=int, default=2,
                   help="Min runs for 'moderate' recommendation (default 2)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


def save_csv(rows: list[dict], path: Path, fieldnames: list[str] | None = None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ══════════════════════════════════════════════════════════════════════
# Load run data
# ══════════════════════════════════════════════════════════════════════

class RunData:
    """Holds data from one XAI run."""
    def __init__(self, run_dir: str):
        self.path = Path(run_dir)
        self.name = self.path.name
        self.exists = self.path.exists() and (self.path / "channel_importance.json").exists()

        if not self.exists:
            self.config = None
            self.channels = None
            self.hc_channels = None
            self.robust_channels = None
            self.freq_hc = None
            self.n_windows = 0
            self.label = f"{self.name} (MISSING)"
            return

        self.config = load_json(self.path / "run_config.json")

        # Channel importance
        ci = load_json(self.path / "channel_importance.json")
        self.channels = ci["channels"] if ci else None

        # High-confidence channels
        self.hc_channels = load_csv_rows(self.path / "high_confidence_channels.csv")

        # Robust channels
        self.robust_channels = load_csv_rows(self.path / "final_important_channels.csv")

        # Frequency for HC channels
        self.freq_hc = load_csv_rows(self.path / "frequency_by_high_confidence_channels.csv")

        # N windows
        self.n_windows = 0
        if self.config:
            self.n_windows = self.config.get("actual_n_windows_used",
                             self.config.get("n_windows",
                             self.config.get("max_samples", 0)))

        self.label = f"{self.name} (N={self.n_windows})"

    def get_channel_ranks(self) -> dict[int, int]:
        """Return {channel_index: rank}."""
        if not self.channels:
            return {}
        return {c["channel"]: c["rank"] for c in self.channels}

    def get_hc_set(self) -> set[int]:
        """Return set of high-confidence channel indices."""
        if not self.hc_channels:
            return set()
        return {int(c["channel"]) if "channel" in c else int(c["rank"])
                for c in self.hc_channels
                if "channel" in c}

    def get_robust_set(self) -> set[int]:
        """Return set of robust channel indices."""
        if not self.robust_channels:
            return set()
        return {int(c["channel_index"]) for c in self.robust_channels}

    def get_contribution_type(self, ch: int) -> str:
        """Get contribution_type for a channel."""
        if not self.channels:
            return "unknown"
        for c in self.channels:
            if c["channel"] == ch:
                return c.get("contribution_type", "unknown")
        return "unknown"

    def get_best_freq_band(self, ch: int) -> str:
        """Get best_frequency_band for an HC channel."""
        if not self.freq_hc:
            return ""
        for c in self.freq_hc:
            ch_idx = int(c.get("channel_index", -1))
            if ch_idx == ch:
                return c.get("best_frequency_band", "")
        return ""

    def get_electrode_name(self, ch: int) -> str:
        if not self.channels:
            return f"Ch{ch}"
        for c in self.channels:
            if c["channel"] == ch:
                return c.get("electrode_name", f"Ch{ch}")
        return f"Ch{ch}"

    def get_roi(self, ch: int) -> str:
        if not self.channels:
            return "Unknown"
        for c in self.channels:
            if c["channel"] == ch:
                return c.get("roi", "Unknown")
        return "Unknown"


# ══════════════════════════════════════════════════════════════════════
# A. High-confidence channel overlap
# ══════════════════════════════════════════════════════════════════════

def compute_hc_overlap(runs: list[RunData], out_dir: Path):
    """Which channels are HC in which runs."""
    available = [r for r in runs if r.exists and r.hc_channels]
    all_hc = set()
    for r in available:
        all_hc |= r.get_hc_set()

    rows = []
    for ch in sorted(all_hc):
        row = {"channel": ch}
        # Get name/roi from first available run
        for r in available:
            name = r.get_electrode_name(ch)
            roi = r.get_roi(ch)
            if name != f"Ch{ch}":
                break
        row["electrode_name"] = name
        row["roi"] = roi

        appeared_in = []
        for r in available:
            present = ch in r.get_hc_set()
            row[f"hc_in_{r.name}"] = "YES" if present else ""
            if present:
                appeared_in.append(r.name)

        row["appeared_in_n_runs"] = len(appeared_in)
        row["appeared_in_all"] = "YES" if len(appeared_in) == len(available) else ""
        rows.append(row)

    rows.sort(key=lambda x: -x["appeared_in_n_runs"])
    save_csv(rows, out_dir / "high_confidence_overlap.csv")
    print(f"  high_confidence_overlap.csv: {len(rows)} channels appeared HC in at least one run")
    return rows


# ══════════════════════════════════════════════════════════════════════
# B. Robust channel overlap
# ══════════════════════════════════════════════════════════════════════

def compute_robust_overlap(runs: list[RunData], out_dir: Path):
    """Which channels are robust in which runs."""
    available = [r for r in runs if r.exists and r.robust_channels]
    all_robust = set()
    for r in available:
        all_robust |= r.get_robust_set()

    rows = []
    for ch in sorted(all_robust):
        row = {"channel": ch}
        for r in available:
            name = r.get_electrode_name(ch)
            roi = r.get_roi(ch)
            if name != f"Ch{ch}":
                break
        row["electrode_name"] = name
        row["roi"] = roi

        appeared_in = []
        for r in available:
            present = ch in r.get_robust_set()
            row[f"robust_in_{r.name}"] = "YES" if present else ""
            if present:
                appeared_in.append(r.name)

        row["appeared_in_n_runs"] = len(appeared_in)
        row["appeared_in_all"] = "YES" if len(appeared_in) == len(available) else ""
        rows.append(row)

    rows.sort(key=lambda x: -x["appeared_in_n_runs"])
    save_csv(rows, out_dir / "robust_overlap.csv")
    print(f"  robust_overlap.csv: {len(rows)} channels appeared robust in at least one run")
    return rows


# ══════════════════════════════════════════════════════════════════════
# C. Rank correlation
# ══════════════════════════════════════════════════════════════════════

def compute_rank_correlation(runs: list[RunData], out_dir: Path):
    """Spearman correlation of channel importance ranks between runs."""
    from scipy.stats import spearmanr

    available = [r for r in runs if r.exists and r.channels]
    if len(available) < 2:
        print("  rank_correlation: need at least 2 runs")
        return []

    rows = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            r1, r2 = available[i], available[j]
            ranks1 = r1.get_channel_ranks()
            ranks2 = r2.get_channel_ranks()
            common_chs = sorted(set(ranks1.keys()) & set(ranks2.keys()))
            if len(common_chs) < 3:
                continue
            v1 = [ranks1[ch] for ch in common_chs]
            v2 = [ranks2[ch] for ch in common_chs]
            rho, p_val = spearmanr(v1, v2)
            rows.append({
                "run_a": r1.name,
                "run_b": r2.name,
                "n_a": r1.n_windows,
                "n_b": r2.n_windows,
                "n_channels": len(common_chs),
                "spearman_rho": round(float(rho), 4),
                "p_value": float(p_val),
            })
            print(f"  Rank correlation {r1.name} vs {r2.name}: rho={rho:.4f} (p={p_val:.2e})")

    save_csv(rows, out_dir / "rank_correlation_across_runs.csv")
    return rows


# ══════════════════════════════════════════════════════════════════════
# D. Channel stability across runs (sign + rank)
# ══════════════════════════════════════════════════════════════════════

def compute_channel_stability(runs: list[RunData], out_dir: Path):
    """For each channel, track rank and contribution_type across runs."""
    available = [r for r in runs if r.exists and r.channels]
    if not available:
        return []

    all_chs = set()
    for r in available:
        all_chs |= set(r.get_channel_ranks().keys())

    rows = []
    for ch in sorted(all_chs):
        row = {"channel": ch}
        for r in available:
            name = r.get_electrode_name(ch)
            roi = r.get_roi(ch)
            if name != f"Ch{ch}":
                break
        row["electrode_name"] = name
        row["roi"] = roi

        ranks = []
        types = []
        for r in available:
            rank_map = r.get_channel_ranks()
            rank = rank_map.get(ch)
            ct = r.get_contribution_type(ch)
            row[f"rank_{r.name}"] = rank if rank is not None else ""
            row[f"type_{r.name}"] = ct if ct != "unknown" else ""
            if rank is not None:
                ranks.append(rank)
            if ct and ct != "unknown":
                types.append(ct)

        row["mean_rank"] = round(np.mean(ranks), 1) if ranks else ""
        row["rank_std"] = round(np.std(ranks), 1) if len(ranks) > 1 else ""

        # Sign consensus
        if types:
            from collections import Counter
            type_counts = Counter(types)
            most_common = type_counts.most_common(1)[0]
            if most_common[1] == len(types):
                row["type_consensus"] = most_common[0]
            elif most_common[1] >= len(types) * 0.75:
                row["type_consensus"] = f"mostly_{most_common[0]}"
            else:
                row["type_consensus"] = "inconsistent"
        else:
            row["type_consensus"] = ""

        rows.append(row)

    rows.sort(key=lambda x: float(x["mean_rank"]) if x["mean_rank"] != "" else 999)
    save_csv(rows, out_dir / "channel_stability_across_runs.csv")
    print(f"  channel_stability_across_runs.csv: {len(rows)} channels")
    return rows


# ══════════════════════════════════════════════════════════════════════
# E. Frequency stability
# ══════════════════════════════════════════════════════════════════════

def compute_frequency_stability(runs: list[RunData], out_dir: Path):
    """For HC channels in multiple runs, compare best_frequency_band."""
    available = [r for r in runs if r.exists and r.freq_hc]
    if len(available) < 2:
        print("  frequency_stability: need at least 2 runs with HC frequency data")
        return []

    all_hc = set()
    for r in available:
        all_hc |= r.get_hc_set()

    rows = []
    for ch in sorted(all_hc):
        bands_seen = []
        row = {"channel": ch}
        for r in available:
            name = r.get_electrode_name(ch)
            roi = r.get_roi(ch)
            if name != f"Ch{ch}":
                break
        row["electrode_name"] = name
        row["roi"] = roi

        for r in available:
            band = r.get_best_freq_band(ch)
            row[f"band_{r.name}"] = band
            if band:
                bands_seen.append(band)

        if bands_seen:
            from collections import Counter
            band_counts = Counter(bands_seen)
            most_common = band_counts.most_common(1)[0]
            row["band_consensus"] = most_common[0]
            row["band_stable"] = "YES" if most_common[1] == len(bands_seen) else "NO"
            row["n_runs_with_data"] = len(bands_seen)
        else:
            row["band_consensus"] = ""
            row["band_stable"] = ""
            row["n_runs_with_data"] = 0

        rows.append(row)

    save_csv(rows, out_dir / "frequency_stability_across_runs.csv")
    print(f"  frequency_stability_across_runs.csv: {len(rows)} channels")
    return rows


# ══════════════════════════════════════════════════════════════════════
# F. Final recommended channels
# ══════════════════════════════════════════════════════════════════════

def compute_final_recommendations(runs: list[RunData], out_dir: Path, min_runs: int = 2):
    """Recommend channels based on cross-run stability."""
    available = [r for r in runs if r.exists and r.channels]
    if len(available) < 2:
        print("  final_recommendations: need at least 2 available runs")
        return []

    n_runs = len(available)
    all_chs = set()
    for r in available:
        all_chs |= set(r.get_channel_ranks().keys())

    rows = []
    for ch in sorted(all_chs):
        # Count appearances
        hc_runs = [r.name for r in available if ch in r.get_hc_set()]
        robust_runs = [r.name for r in available if ch in r.get_robust_set()]

        # Get name/roi
        for r in available:
            name = r.get_electrode_name(ch)
            roi = r.get_roi(ch)
            if name != f"Ch{ch}":
                break

        # Mean rank
        ranks = []
        for r in available:
            rank_map = r.get_channel_ranks()
            if ch in rank_map:
                ranks.append(rank_map[ch])
        mean_rank = np.mean(ranks) if ranks else 999
        rank_std = np.std(ranks) if len(ranks) > 1 else 0

        # Contribution type consensus
        types = [r.get_contribution_type(ch) for r in available
                 if r.get_contribution_type(ch) != "unknown"]
        if types:
            from collections import Counter
            tc = Counter(types)
            most_common_type = tc.most_common(1)[0][0]
            if tc.most_common(1)[0][1] == len(types):
                type_consensus = most_common_type
            else:
                type_consensus = f"mixed ({', '.join(f'{t}:{c}' for t, c in tc.most_common())})"
        else:
            type_consensus = "unknown"

        # Best frequency band consensus
        bands = [r.get_best_freq_band(ch) for r in available if r.get_best_freq_band(ch)]
        if bands:
            from collections import Counter
            bc = Counter(bands)
            band_consensus = bc.most_common(1)[0][0]
        else:
            band_consensus = ""

        # Determine recommendation level
        n_hc = len(hc_runs)
        n_robust = len(robust_runs)

        # Check if HC in full run
        hc_in_full = any("full" in rn.lower() for rn in hc_runs)

        if n_hc >= 3 or (n_hc >= 2 and hc_in_full):
            level = "Strong"
        elif n_hc >= min_runs:
            level = "Moderate"
        elif n_robust == n_runs and mean_rank <= 20:
            level = "Exploratory"
        else:
            continue  # Don't recommend

        # Interpretation note
        notes = []
        if n_hc == n_runs:
            notes.append(f"HC in all {n_runs} runs")
        elif n_hc > 0:
            notes.append(f"HC in {n_hc}/{n_runs} runs")
        if n_robust == n_runs:
            notes.append("robust in all runs")
        if rank_std > 10:
            notes.append("high rank variability")
        if "mixed" in type_consensus:
            notes.append("CAUTION: sign changes across runs")
        if hc_in_full:
            notes.append("confirmed at full dataset")

        rows.append({
            "channel_index": ch,
            "electrode_name": name,
            "roi": roi,
            "appeared_high_confidence_runs": n_hc,
            "appeared_robust_runs": n_robust,
            "high_confidence_run_names": "; ".join(hc_runs),
            "robust_run_names": "; ".join(robust_runs),
            "mean_rank": round(mean_rank, 1),
            "rank_std": round(rank_std, 1),
            "contribution_type_consensus": type_consensus,
            "best_frequency_band_consensus": band_consensus,
            "recommendation_level": level,
            "interpretation_note": "; ".join(notes),
        })

    # Sort by level then rank
    level_order = {"Strong": 0, "Moderate": 1, "Exploratory": 2}
    rows.sort(key=lambda x: (level_order.get(x["recommendation_level"], 9),
                             float(x["mean_rank"])))

    save_csv(rows, out_dir / "final_recommended_channels.csv")
    print(f"  final_recommended_channels.csv: {len(rows)} recommended channels")
    for r in rows:
        print(f"    [{r['recommendation_level']:12s}] {r['electrode_name']:>6} "
              f"(Ch{r['channel_index']:2d}) HC={r['appeared_high_confidence_runs']}/{n_runs} "
              f"robust={r['appeared_robust_runs']}/{n_runs} "
              f"rank={r['mean_rank']}±{r['rank_std']}")
    return rows


# ══════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════

def plot_hc_overlap(runs: list[RunData], hc_overlap: list[dict], out_dir: Path):
    """Plot high-confidence channel overlap across runs."""
    available = [r for r in runs if r.exists and r.hc_channels]
    if len(available) < 2 or not hc_overlap:
        return

    # Binary matrix: channels x runs
    channels = [row["channel"] for row in hc_overlap]
    ch_names = [f"{row['electrode_name']} (Ch{row['channel']})" for row in hc_overlap]
    matrix = np.zeros((len(channels), len(available)))
    for ci, row in enumerate(hc_overlap):
        for ri, r in enumerate(available):
            if row.get(f"hc_in_{r.name}") == "YES":
                matrix[ci, ri] = 1

    fig, ax = plt.subplots(figsize=(max(6, len(available) * 2), max(4, len(channels) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([f"{r.name}\n(N={r.n_windows})" for r in available], fontsize=8)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(ch_names, fontsize=8)
    ax.set_title("High-Confidence Channel Overlap Across Runs")

    for ci in range(len(channels)):
        for ri in range(len(available)):
            ax.text(ri, ci, "✓" if matrix[ci, ri] else "",
                    ha="center", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "high_confidence_overlap_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_rank_stability(runs: list[RunData], stability: list[dict], out_dir: Path):
    """Plot rank evolution for top channels across runs."""
    available = [r for r in runs if r.exists and r.channels]
    if len(available) < 2 or not stability:
        return

    # Select top 20 by mean rank
    top_rows = [r for r in stability if r.get("mean_rank", 999) != "" and float(r["mean_rank"]) <= 20]
    if not top_rows:
        top_rows = stability[:20]

    fig, ax = plt.subplots(figsize=(max(8, len(available) * 2.5), 8))
    x_positions = range(len(available))

    for row in top_rows:
        ranks = []
        for r in available:
            rk = row.get(f"rank_{r.name}", "")
            ranks.append(int(rk) if rk != "" else None)

        # Plot only if we have at least 2 data points
        valid = [(x, rk) for x, rk in zip(x_positions, ranks) if rk is not None]
        if len(valid) < 2:
            continue
        xs, ys = zip(*valid)
        label = f"{row['electrode_name']} (Ch{row['channel']})"
        ax.plot(xs, ys, "o-", markersize=5, label=label, alpha=0.7)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([f"{r.name}\n(N={r.n_windows})" for r in available], fontsize=9)
    ax.set_ylabel("Importance Rank (lower = more important)")
    ax.set_title("Channel Rank Stability Across Runs (Top 20)")
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="upper right", ncol=2, bbox_to_anchor=(1.35, 1))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "channel_rank_stability_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_frequency_stability(runs: list[RunData], freq_stab: list[dict], out_dir: Path):
    """Plot frequency band assignment stability."""
    available = [r for r in runs if r.exists and r.freq_hc]
    if len(available) < 2 or not freq_stab:
        return

    band_map = {"delta": 0, "theta": 1, "alpha": 2, "beta": 3}
    band_colors = {"delta": "#1565c0", "theta": "#2e7d32", "alpha": "#f57f17", "beta": "#d32f2f"}

    fig, ax = plt.subplots(figsize=(max(6, len(available) * 2), max(4, len(freq_stab) * 0.5)))

    for ci, row in enumerate(freq_stab):
        for ri, r in enumerate(available):
            band = row.get(f"band_{r.name}", "")
            if band in band_map:
                color = band_colors[band]
                ax.scatter(ri, ci, c=color, s=100, marker="s", edgecolors="k",
                          linewidths=0.5, zorder=5)

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([f"{r.name}\n(N={r.n_windows})" for r in available], fontsize=8)
    ax.set_yticks(range(len(freq_stab)))
    ax.set_yticklabels([f"{r['electrode_name']} (Ch{r['channel']})" for r in freq_stab], fontsize=8)
    ax.set_title("Frequency Band Assignment Across Runs")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=b) for b, c in band_colors.items()]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "frequency_stability_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Comparison report
# ══════════════════════════════════════════════════════════════════════

def write_comparison_report(runs: list[RunData], hc_overlap, robust_overlap,
                             rank_corr, ch_stability, freq_stab, recommendations,
                             out_dir: Path):
    """Write the validation_comparison_report.txt."""
    available = [r for r in runs if r.exists]
    missing = [r for r in runs if not r.exists]

    L = []
    L.append("=" * 80)
    L.append("VALIDATION COMPARISON REPORT")
    L.append("Cross-Run Stability Analysis for Focused XAI Channel Importance")
    L.append("=" * 80)
    L.append("")
    L.append(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
    L.append(f"Runs compared: {len(available)}")
    if missing:
        L.append(f"Missing runs: {len(missing)}")
        for r in missing:
            L.append(f"  - {r.name} (not found at {r.path})")
    L.append("")

    # Run summary
    L.append("Run summary:")
    for r in available:
        n_hc = len(r.get_hc_set())
        n_robust = len(r.get_robust_set())
        L.append(f"  {r.name:40s}: N={r.n_windows:>5}, HC={n_hc:>2}, robust={n_robust:>2}")
    L.append("")

    # IMPORTANT CAVEAT
    L.append("=" * 80)
    L.append("INTERPRETATION GUIDELINES")
    L.append("=" * 80)
    L.append("")
    L.append("- The N=200 result is PRELIMINARY. It may contain sample-dependent channels.")
    L.append("- Larger-N and full-dataset runs are VALIDATION checks, not independent")
    L.append("  experiments. Channels should not be considered final unless they are")
    L.append("  stable across runs.")
    L.append("- If a channel is strong at N=200 but disappears at N=500 or full, it may")
    L.append("  be SAMPLE-DEPENDENT and should be interpreted cautiously.")
    L.append("- If a channel appears only at larger N, it may have been UNDERPOWERED")
    L.append("  in smaller runs.")
    L.append("- If contribution type (facilitatory/suppressive) changes sign across runs,")
    L.append("  interpret that channel CAUTIOUSLY.")
    L.append("")

    # A. HC overlap
    L.append("=" * 80)
    L.append("A. HIGH-CONFIDENCE CHANNEL OVERLAP")
    L.append("=" * 80)
    L.append("")
    if hc_overlap:
        in_all = [r for r in hc_overlap if r.get("appeared_in_all") == "YES"]
        L.append(f"Channels appearing HC in at least one run: {len(hc_overlap)}")
        L.append(f"Channels appearing HC in ALL available runs: {len(in_all)}")
        if in_all:
            L.append("")
            L.append("Channels HC in all runs:")
            for r in in_all:
                L.append(f"  {r['electrode_name']:>6} (Ch{r['channel']:>2}) — {r['roi']}")
        L.append("")

        only_small = [r for r in hc_overlap
                      if r["appeared_in_n_runs"] == 1
                      and any(r.get(f"hc_in_{rv.name}") == "YES"
                              for rv in available if rv.n_windows <= 200)]
        if only_small:
            L.append(f"Channels HC only in smallest run (potentially sample-dependent): {len(only_small)}")
            for r in only_small:
                L.append(f"  {r['electrode_name']:>6} (Ch{r['channel']:>2})")
            L.append("")
    else:
        L.append("No high-confidence data available for comparison.")
        L.append("")

    # B. Robust overlap
    L.append("=" * 80)
    L.append("B. ROBUST CHANNEL OVERLAP")
    L.append("=" * 80)
    L.append("")
    if robust_overlap:
        in_all = [r for r in robust_overlap if r.get("appeared_in_all") == "YES"]
        L.append(f"Channels appearing robust in at least one run: {len(robust_overlap)}")
        L.append(f"Channels appearing robust in ALL available runs: {len(in_all)}")
        L.append("")
    else:
        L.append("No robust channel data available for comparison.")
        L.append("")

    # C. Rank correlation
    L.append("=" * 80)
    L.append("C. RANK CORRELATION ACROSS RUNS")
    L.append("=" * 80)
    L.append("")
    if rank_corr:
        for rc in rank_corr:
            L.append(f"  {rc['run_a']:30s} vs {rc['run_b']:30s}: "
                     f"rho={rc['spearman_rho']:.4f} (p={rc['p_value']:.2e})")
        L.append("")
        avg_rho = np.mean([rc["spearman_rho"] for rc in rank_corr])
        L.append(f"  Average Spearman rho: {avg_rho:.4f}")
        if avg_rho > 0.8:
            L.append("  Interpretation: Channel rankings are HIGHLY STABLE across runs.")
        elif avg_rho > 0.6:
            L.append("  Interpretation: Channel rankings are MODERATELY STABLE across runs.")
        else:
            L.append("  Interpretation: Channel rankings show CONSIDERABLE VARIABILITY across runs.")
        L.append("")
    else:
        L.append("Insufficient runs for rank correlation comparison.")
        L.append("")

    # D. Sign consistency
    L.append("=" * 80)
    L.append("D. SIGN CONSISTENCY")
    L.append("=" * 80)
    L.append("")
    if ch_stability:
        inconsistent = [r for r in ch_stability if r.get("type_consensus", "").startswith("inconsistent")]
        n_stable = sum(1 for r in ch_stability
                       if r.get("type_consensus", "") in ("facilitatory", "suppressive"))
        L.append(f"  Channels with consistent sign: {n_stable}/{len(ch_stability)}")
        L.append(f"  Channels with inconsistent sign: {len(inconsistent)}/{len(ch_stability)}")
        if inconsistent:
            L.append("")
            L.append("  Channels with sign changes (interpret cautiously):")
            for r in inconsistent[:10]:
                L.append(f"    {r.get('electrode_name', ''):>6} (Ch{r['channel']:>2})")
        L.append("")
    else:
        L.append("No stability data available.")
        L.append("")

    # E. Frequency stability
    L.append("=" * 80)
    L.append("E. FREQUENCY BAND STABILITY")
    L.append("=" * 80)
    L.append("")
    if freq_stab:
        stable = [r for r in freq_stab if r.get("band_stable") == "YES"]
        L.append(f"  HC channels with frequency data: {len(freq_stab)}")
        L.append(f"  Channels with stable best band: {len(stable)}/{len(freq_stab)}")
        if stable:
            L.append("")
            for r in stable:
                L.append(f"    {r['electrode_name']:>6}: {r['band_consensus']} (stable across {r['n_runs_with_data']} runs)")
        L.append("")
    else:
        L.append("No frequency stability data available.")
        L.append("")

    # F. Recommendations
    L.append("=" * 80)
    L.append("F. FINAL RECOMMENDED CHANNELS")
    L.append("=" * 80)
    L.append("")
    if recommendations:
        strong = [r for r in recommendations if r["recommendation_level"] == "Strong"]
        moderate = [r for r in recommendations if r["recommendation_level"] == "Moderate"]
        exploratory = [r for r in recommendations if r["recommendation_level"] == "Exploratory"]

        L.append(f"  Strong recommendations: {len(strong)}")
        L.append(f"  Moderate recommendations: {len(moderate)}")
        L.append(f"  Exploratory: {len(exploratory)}")
        L.append("")

        L.append("Recommendation criteria:")
        L.append("  Strong: HC in ≥3 runs, OR HC in full + one other run")
        L.append("  Moderate: HC in ≥2 runs")
        L.append("  Exploratory: robust in all runs + top-20 in ≥2 runs")
        L.append("")

        if strong:
            L.append("STRONG:")
            for r in strong:
                L.append(f"  {r['electrode_name']:>6} (Ch{r['channel_index']:>2}) "
                         f"{r['roi']:>16} | {r['contribution_type_consensus']:>12} | "
                         f"rank={r['mean_rank']}±{r['rank_std']} | "
                         f"band={r['best_frequency_band_consensus']}")
            L.append("")

        if moderate:
            L.append("MODERATE:")
            for r in moderate:
                L.append(f"  {r['electrode_name']:>6} (Ch{r['channel_index']:>2}) "
                         f"{r['roi']:>16} | {r['contribution_type_consensus']:>12} | "
                         f"rank={r['mean_rank']}±{r['rank_std']} | "
                         f"band={r['best_frequency_band_consensus']}")
            L.append("")

        if exploratory:
            L.append("EXPLORATORY:")
            for r in exploratory:
                L.append(f"  {r['electrode_name']:>6} (Ch{r['channel_index']:>2}) "
                         f"{r['roi']:>16} | {r['contribution_type_consensus']:>12} | "
                         f"rank={r['mean_rank']}±{r['rank_std']}")
            L.append("")
    else:
        L.append("No recommendations generated (need at least 2 runs).")
        L.append("")

    # Output files
    L.append("=" * 80)
    L.append("OUTPUT FILES")
    L.append("=" * 80)
    L.append("")
    L.append(f"Output directory: {out_dir}")
    L.append("  channel_stability_across_runs.csv")
    L.append("  high_confidence_overlap.csv")
    L.append("  robust_overlap.csv")
    L.append("  rank_correlation_across_runs.csv")
    L.append("  frequency_stability_across_runs.csv")
    L.append("  final_recommended_channels.csv")
    L.append("  high_confidence_overlap_plot.png")
    L.append("  channel_rank_stability_plot.png")
    L.append("  frequency_stability_plot.png")
    L.append("  validation_comparison_report.txt")
    L.append("")
    L.append("=" * 80)
    L.append("END OF REPORT")
    L.append("=" * 80)

    report_text = "\n".join(L)
    (out_dir / "validation_comparison_report.txt").write_text(report_text, encoding="utf-8")
    print(f"  Report saved ({len(L)} lines)")
    return report_text


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VALIDATION COMPARISON: Cross-Run Stability Analysis")
    print("=" * 70)

    # Load runs
    runs = []
    for run_path in args.runs:
        # Handle relative paths from project root
        p = Path(run_path)
        if not p.is_absolute():
            p = ROOT / run_path
        rd = RunData(str(p))
        runs.append(rd)
        status = "OK" if rd.exists else "MISSING"
        print(f"  [{status:7s}] {rd.label:40s} ({rd.path})")

    available = [r for r in runs if r.exists]
    missing = [r for r in runs if not r.exists]

    if len(available) < 1:
        print("\nERROR: No available run folders found.")
        sys.exit(1)

    if missing:
        print(f"\nWARNING: {len(missing)} run folder(s) not found. Comparison will use available runs only.")

    # Compute metrics
    print(f"\nComparing {len(available)} runs...")

    print("\n[A] High-confidence overlap...")
    hc_overlap = compute_hc_overlap(runs, out_dir)

    print("\n[B] Robust overlap...")
    robust_overlap = compute_robust_overlap(runs, out_dir)

    print("\n[C] Rank correlation...")
    rank_corr = compute_rank_correlation(runs, out_dir)

    print("\n[D] Channel stability...")
    ch_stability = compute_channel_stability(runs, out_dir)

    print("\n[E] Frequency stability...")
    freq_stab = compute_frequency_stability(runs, out_dir)

    print("\n[F] Final recommendations...")
    recommendations = compute_final_recommendations(runs, out_dir, args.high_confidence_min_runs)

    # Plots
    print("\n[G] Generating plots...")
    plot_hc_overlap(runs, hc_overlap, out_dir)
    plot_rank_stability(runs, ch_stability, out_dir)
    plot_frequency_stability(runs, freq_stab, out_dir)

    # Report
    print("\n[H] Writing comparison report...")
    write_comparison_report(runs, hc_overlap, robust_overlap, rank_corr,
                            ch_stability, freq_stab, recommendations, out_dir)

    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"  Available runs: {len(available)}")
    print(f"  Recommended channels: {len(recommendations)}")
    print(f"  All outputs in: {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
