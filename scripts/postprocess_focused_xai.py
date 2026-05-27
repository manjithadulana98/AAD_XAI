"""Post-processing for focused XAI: adds high-confidence selection, montage validation,
scalp plots, supervisor summary, and updated report.

This script reads the existing xai_results_focused/ outputs and generates additional
files without re-running the expensive computations.

Usage:
    python scripts/postprocess_focused_xai.py
    python scripts/postprocess_focused_xai.py --output-dir xai_results_focused
    python scripts/postprocess_focused_xai.py --high-confidence-top-k 15
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
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Post-process focused XAI results")
    p.add_argument("--output-dir", type=str, default=str(ROOT / "xai_results_focused"))
    p.add_argument("--montage-file", type=str,
                   default=str(ROOT / "config" / "dtu_channel_montage.csv"))
    p.add_argument("--high-confidence-top-k", type=int, default=20)
    p.add_argument("--high-confidence-stability-threshold", type=float, default=0.60)
    p.add_argument("--high-confidence-require-both-fdr", type=bool, default=True)
    p.add_argument("--high-confidence-max-ig-rank", type=int, default=20)
    p.add_argument("--high-confidence-use-ig", type=bool, default=True)
    p.add_argument("--strict-montage-validation", action="store_true")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Montage validation
# ══════════════════════════════════════════════════════════════════════

def validate_montage(montage_path: str, expected_n_channels: int = 64) -> dict:
    """Validate the montage CSV file.

    Returns a dict with validation results.
    """
    results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "montage_path": montage_path,
        "n_channels": 0,
        "n_rois": 0,
        "has_coordinates": False,
    }

    path = Path(montage_path)
    if not path.is_file():
        results["valid"] = False
        results["errors"].append(f"Montage file not found: {montage_path}")
        return results

    # Read file
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for row in reader:
                rows.append(row)
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Failed to read montage file: {e}")
        return results

    # Check required columns
    required = {"channel_index", "electrode_name", "roi", "x", "y", "z"}
    missing_cols = required - set(headers)
    if missing_cols:
        results["valid"] = False
        results["errors"].append(f"Missing required columns: {missing_cols}")
        return results

    results["n_channels"] = len(rows)

    # Check channel indices
    indices = []
    for row in rows:
        try:
            idx = int(row["channel_index"])
            indices.append(idx)
        except ValueError:
            results["errors"].append(f"Non-integer channel_index: {row['channel_index']}")
            results["valid"] = False

    if results["valid"]:
        if len(set(indices)) != len(indices):
            results["errors"].append("Duplicate channel indices found")
            results["valid"] = False

        expected_set = set(range(expected_n_channels))
        actual_set = set(indices)
        if actual_set != expected_set:
            if min(indices) == 1 and max(indices) == expected_n_channels:
                results["warnings"].append(
                    "Channel indices appear to be 1-based (1 to 64) instead of 0-based (0 to 63). "
                    "The pipeline uses 0-based indexing.")
            else:
                missing = expected_set - actual_set
                extra = actual_set - expected_set
                if missing:
                    results["warnings"].append(f"Missing channel indices: {sorted(missing)[:10]}...")
                if extra:
                    results["warnings"].append(f"Extra channel indices: {sorted(extra)[:10]}...")

        if len(rows) != expected_n_channels:
            results["warnings"].append(
                f"Montage has {len(rows)} rows but expected {expected_n_channels} channels")

    # Check electrode names and ROIs
    empty_names = [i for i, r in enumerate(rows) if not r.get("electrode_name", "").strip()]
    empty_rois = [i for i, r in enumerate(rows) if not r.get("roi", "").strip()]
    if empty_names:
        results["warnings"].append(f"{len(empty_names)} channels have empty electrode_name")
    if empty_rois:
        results["warnings"].append(f"{len(empty_rois)} channels have empty roi")

    # Check coordinates
    coords_valid = True
    all_zero = True
    for row in rows:
        try:
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            if x != 0 or y != 0 or z != 0:
                all_zero = False
        except (ValueError, TypeError):
            coords_valid = False
            break

    if not coords_valid:
        results["warnings"].append("Some coordinates are non-numeric")
    elif all_zero:
        results["warnings"].append("All coordinates are zero — scalp plots may not be meaningful")
    else:
        results["has_coordinates"] = True

    # Count ROIs
    rois = set(r["roi"] for r in rows if r.get("roi", "").strip())
    results["n_rois"] = len(rois)
    results["rois"] = sorted(rois)

    if results["errors"]:
        results["valid"] = False

    return results


def write_montage_validation_report(validation: dict, out_dir: Path):
    """Save montage validation results."""
    # JSON
    save_json(validation, out_dir / "montage_validation.json")

    # TXT
    lines = []
    lines.append("MONTAGE VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Montage file: {validation['montage_path']}")
    lines.append(f"Overall status: {'PASSED' if validation['valid'] else 'FAILED'}")
    lines.append(f"Channels: {validation['n_channels']}")
    lines.append(f"ROIs: {validation['n_rois']}")
    lines.append(f"Has coordinates: {validation['has_coordinates']}")
    lines.append("")

    if validation["errors"]:
        lines.append("ERRORS:")
        for e in validation["errors"]:
            lines.append(f"  ✗ {e}")
        lines.append("")

    if validation["warnings"]:
        lines.append("WARNINGS:")
        for w in validation["warnings"]:
            lines.append(f"  ⚠ {w}")
        lines.append("")

    if not validation["errors"] and not validation["warnings"]:
        lines.append("No issues found.")
        lines.append("")

    (out_dir / "montage_validation.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Montage validation saved to {out_dir / 'montage_validation.txt'}")


# ══════════════════════════════════════════════════════════════════════
# High-confidence channel selection
# ══════════════════════════════════════════════════════════════════════

def select_high_confidence_channels(combined_channels: list, args) -> list:
    """Apply strict high-confidence criteria to combined channel importance data.

    Returns the combined_channels list with added high_confidence fields.
    """
    top_k = args.high_confidence_top_k
    stab_thresh = args.high_confidence_stability_threshold
    require_both_fdr = args.high_confidence_require_both_fdr
    max_ig_rank = args.high_confidence_max_ig_rank
    use_ig = args.high_confidence_use_ig

    for c in combined_channels:
        reasons = []
        failed = []

        # Criterion 1: same sign
        same_sign = c["contribution_type"] in ("facilitatory", "suppressive")
        if same_sign:
            reasons.append("same sign")
        else:
            failed.append("mixed sign between occlusion and permutation")

        # Criterion 2: both FDR-significant
        both_fdr = c["occ_fdr_significant"] and c["perm_fdr_significant"]
        if require_both_fdr:
            if both_fdr:
                reasons.append("both FDR-significant")
            else:
                if not c["occ_fdr_significant"]:
                    failed.append("occlusion not FDR-significant")
                if not c["perm_fdr_significant"]:
                    failed.append("permutation not FDR-significant")
        else:
            at_least_one = c["occ_fdr_significant"] or c["perm_fdr_significant"]
            if at_least_one:
                reasons.append("at least one FDR-significant")
            else:
                failed.append("neither method FDR-significant")

        # Criterion 3: stability
        stab_frac = c.get("subject_stability_frac", 0.0)
        if stab_frac >= stab_thresh:
            reasons.append(f"stability {c['subject_stability']}")
        else:
            failed.append(f"stability below threshold ({c['subject_stability']} < {stab_thresh:.0%})")

        # Criterion 4: top-K by combined importance
        rank = c["rank"]
        if rank <= top_k:
            reasons.append(f"top-{top_k}")
        else:
            failed.append(f"rank {rank} > top-{top_k}")

        # Criterion 5 (optional): IG rank
        ig_pass = True
        if use_ig and c.get("ig_rank", 65) <= max_ig_rank:
            reasons.append(f"IG rank {c['ig_rank']}")
        elif use_ig:
            # IG is optional/supporting - don't make it a hard fail
            pass

        # High confidence = all hard criteria pass
        is_hc = same_sign and (both_fdr if require_both_fdr else (c["occ_fdr_significant"] or c["perm_fdr_significant"])) and (stab_frac >= stab_thresh) and (rank <= top_k)

        c["high_confidence"] = is_hc
        c["high_confidence_reason"] = "; ".join(reasons) if is_hc else ""
        c["high_confidence_failed_criteria"] = "; ".join(failed) if not is_hc else ""
        c["combined_abs_importance_rank"] = rank
        c["both_fdr_significant"] = both_fdr
        c["same_sign_occ_perm"] = same_sign
        c["stability_fraction"] = stab_frac

    return combined_channels


def save_high_confidence_channels(combined_channels: list, out_dir: Path):
    """Save high_confidence_channels.csv."""
    hc = [c for c in combined_channels if c["high_confidence"]]
    hc.sort(key=lambda x: x["rank"])

    csv_fields = [
        "rank", "channel", "electrode_name", "roi",
        "occ_score", "occ_ci_lo", "occ_ci_hi", "occ_p_value", "occ_fdr_p_value",
        "occ_fdr_significant",
        "perm_score", "perm_ci_lo", "perm_ci_hi", "perm_p_value", "perm_fdr_p_value",
        "perm_fdr_significant",
        "ig_rank", "ig_importance",
        "contribution_type", "subject_stability", "stability_fraction",
        "high_confidence", "high_confidence_reason", "high_confidence_failed_criteria",
        "combined_abs_importance_rank", "both_fdr_significant", "same_sign_occ_perm",
        "robust_significant", "combined_score",
    ]
    with open(out_dir / "high_confidence_channels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for c in hc:
            w.writerow({k: c.get(k, "") for k in csv_fields})

    print(f"  Saved high_confidence_channels.csv ({len(hc)} channels)")
    return hc


# ══════════════════════════════════════════════════════════════════════
# Supervisor summary
# ══════════════════════════════════════════════════════════════════════

def build_supervisor_channel_summary(hc_channels: list, freq_lookup: dict, out_dir: Path):
    """Build supervisor_channel_summary.csv for high-confidence channels."""
    rows = []
    for i, c in enumerate(hc_channels):
        ch = c["channel"]
        freq_data = freq_lookup.get(ch, {})

        # Short interpretation
        if c["contribution_type"] == "facilitatory":
            interp = ("High-confidence facilitatory channel; removing or permuting this "
                      "channel reduces attended decoding confidence.")
        elif c["contribution_type"] == "suppressive":
            interp = ("High-confidence suppressive channel; removing or permuting this "
                      "channel improves attended decoding confidence, suggesting noisy "
                      "or competing information.")
        else:
            interp = "Mixed contribution; interpret cautiously."

        rows.append({
            "rank": i + 1,
            "channel_index": ch,
            "electrode_name": c["electrode_name"],
            "roi": c["roi"],
            "contribution_type": c["contribution_type"],
            "occlusion_delta_p": c["occ_score"],
            "occlusion_ci_low": c["occ_ci_lo"],
            "occlusion_ci_high": c["occ_ci_hi"],
            "occ_fdr_p_value": c["occ_fdr_p_value"],
            "permutation_delta_p": c["perm_score"],
            "permutation_ci_low": c["perm_ci_lo"],
            "permutation_ci_high": c["perm_ci_hi"],
            "perm_fdr_p_value": c["perm_fdr_p_value"],
            "subject_stability": c["subject_stability"],
            "stability_fraction": c["stability_fraction"],
            "ig_rank": c["ig_rank"],
            "best_frequency_band": freq_data.get("most_important_band", ""),
            "delta_score": freq_data.get("delta_dp", ""),
            "theta_score": freq_data.get("theta_dp", ""),
            "alpha_score": freq_data.get("alpha_dp", ""),
            "beta_score": freq_data.get("beta_dp", ""),
            "short_interpretation": interp,
        })

    csv_fields = list(rows[0].keys()) if rows else []
    with open(out_dir / "supervisor_channel_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"  Saved supervisor_channel_summary.csv ({len(rows)} high-confidence channels)")
    return rows


# ══════════════════════════════════════════════════════════════════════
# Frequency by high-confidence channels
# ══════════════════════════════════════════════════════════════════════

def compute_frequency_by_high_confidence(hc_channels: list, freq_by_channel: list,
                                          montage_data: dict, out_dir: Path):
    """Filter frequency data for high-confidence channels and save."""
    freq_lookup = {r["channel"]: r for r in freq_by_channel}
    hc_chs = set(c["channel"] for c in hc_channels)

    # Per-channel
    hc_freq = []
    for c in hc_channels:
        ch = c["channel"]
        fd = freq_lookup.get(ch)
        if not fd:
            continue
        # Frequency interpretation
        best = fd.get("most_important_band", "")
        best_dp = fd.get(f"{best}_dp", 0) if best else 0
        if best_dp > 0:
            freq_interp = f"Removing {best} band hurts decoding — {best} supports the model"
        elif best_dp < 0:
            freq_interp = f"Removing {best} band helps decoding — {best} may carry noise"
        else:
            freq_interp = "No dominant frequency effect"

        hc_freq.append({
            "channel_index": ch,
            "electrode_name": c["electrode_name"],
            "roi": c["roi"],
            "contribution_type": c["contribution_type"],
            "best_frequency_band": best,
            "delta_score": fd.get("delta_dp", ""),
            "theta_score": fd.get("theta_dp", ""),
            "alpha_score": fd.get("alpha_dp", ""),
            "beta_score": fd.get("beta_dp", ""),
            "frequency_interpretation": freq_interp,
        })

    if hc_freq:
        csv_fields = list(hc_freq[0].keys())
        with open(out_dir / "frequency_by_high_confidence_channels.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            w.writeheader()
            for r in hc_freq:
                w.writerow(r)

    # Per-ROI (only ROIs with high-confidence channels)
    rois = montage_data.get("rois", {})
    roi_freq = []
    for roi_name, chs in rois.items():
        roi_hc = [ch for ch in chs if ch in hc_chs]
        if not roi_hc:
            continue
        # Average frequency scores from hc channels in this ROI
        bands = ["delta", "theta", "alpha", "beta"]
        band_means = {}
        for b in bands:
            vals = [freq_lookup[ch].get(f"{b}_dp", 0) for ch in roi_hc if ch in freq_lookup]
            band_means[b] = float(np.mean(vals)) if vals else 0.0

        best_band = max(bands, key=lambda b: abs(band_means[b]))
        roi_freq.append({
            "roi": roi_name,
            "n_high_confidence_channels": len(roi_hc),
            "best_frequency_band": best_band,
            "delta_score": band_means["delta"],
            "theta_score": band_means["theta"],
            "alpha_score": band_means["alpha"],
            "beta_score": band_means["beta"],
        })

    if roi_freq:
        csv_fields = list(roi_freq[0].keys())
        with open(out_dir / "frequency_by_high_confidence_roi.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            w.writeheader()
            for r in roi_freq:
                w.writerow(r)

    # Heatmap
    if hc_freq:
        bands = ["delta", "theta", "alpha", "beta"]
        matrix = np.array([[float(r.get(f"{b}_score", 0) or 0) for b in bands] for r in hc_freq])
        fig, ax = plt.subplots(figsize=(8, max(4, len(hc_freq) * 0.4)))
        vmax = np.abs(matrix).max() if matrix.size > 0 else 1
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(4))
        ax.set_xticklabels(bands)
        ax.set_yticks(range(len(hc_freq)))
        ax.set_yticklabels([f"{r['electrode_name']} ({r['roi'][:5]})" for r in hc_freq], fontsize=8)
        ax.set_title("Frequency Band ΔP — High-Confidence Channels")
        plt.colorbar(im, ax=ax, label="ΔP", shrink=0.8)
        plt.tight_layout()
        plt.savefig(out_dir / "frequency_high_confidence_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved frequency_by_high_confidence_channels.csv ({len(hc_freq)} channels)")
    print(f"  Saved frequency_by_high_confidence_roi.csv ({len(roi_freq)} ROIs)")
    return hc_freq, roi_freq


# ══════════════════════════════════════════════════════════════════════
# Scalp plots
# ══════════════════════════════════════════════════════════════════════

def load_montage_coords(montage_path: str) -> dict | None:
    """Load x, y coordinates from montage for scalp plotting."""
    path = Path(montage_path)
    if not path.is_file():
        return None
    coords = {}
    names = {}
    rois = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["channel_index"])
                x = float(row["x"])
                y = float(row["y"])
                coords[idx] = (x, y)
                names[idx] = row["electrode_name"]
                rois[idx] = row["roi"]
            except (ValueError, KeyError):
                continue
    if not coords:
        return None
    # Check not all zero
    xs = [v[0] for v in coords.values()]
    ys = [v[1] for v in coords.values()]
    if all(x == 0 for x in xs) and all(y == 0 for y in ys):
        return None
    return {"coords": coords, "names": names, "rois": rois}


def _draw_head_outline(ax):
    """Draw a simple head outline for scalp plots."""
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 0.1
    ax.plot(r * np.cos(theta), r * np.sin(theta), "k-", linewidth=1, alpha=0.3)
    # nose
    ax.plot([0, -0.005, 0.005, 0], [r, r + 0.012, r + 0.012, r], "k-", linewidth=1, alpha=0.3)


def plot_scalp_occlusion_importance(combined_channels, montage_coords, out_dir):
    """2D scatter scalp plot colored by occlusion importance."""
    coords = montage_coords["coords"]
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_head_outline(ax)

    xs = [coords[c["channel"]][0] for c in combined_channels if c["channel"] in coords]
    ys = [coords[c["channel"]][1] for c in combined_channels if c["channel"] in coords]
    vals = [c["occ_score"] for c in combined_channels if c["channel"] in coords]
    vmax = max(abs(v) for v in vals) if vals else 1

    sc = ax.scatter(xs, ys, c=vals, cmap="RdBu_r", s=120, vmin=-vmax, vmax=vmax,
                    edgecolors="k", linewidths=0.5, zorder=5)
    for c in combined_channels[:10]:
        if c["channel"] in coords:
            x, y = coords[c["channel"]]
            ax.annotate(c["electrode_name"], (x, y), fontsize=6, ha="center", va="bottom",
                       xytext=(0, 6), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="Occlusion ΔP", shrink=0.7)
    ax.set_title("Scalp: Channel Occlusion Importance")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "scalp_occlusion_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_scalp_permutation_importance(combined_channels, montage_coords, out_dir):
    """2D scatter scalp plot colored by permutation importance."""
    coords = montage_coords["coords"]
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_head_outline(ax)

    xs = [coords[c["channel"]][0] for c in combined_channels if c["channel"] in coords]
    ys = [coords[c["channel"]][1] for c in combined_channels if c["channel"] in coords]
    vals = [c["perm_score"] for c in combined_channels if c["channel"] in coords]
    vmax = max(abs(v) for v in vals) if vals else 1

    sc = ax.scatter(xs, ys, c=vals, cmap="RdBu_r", s=120, vmin=-vmax, vmax=vmax,
                    edgecolors="k", linewidths=0.5, zorder=5)
    for c in combined_channels[:10]:
        if c["channel"] in coords:
            x, y = coords[c["channel"]]
            ax.annotate(c["electrode_name"], (x, y), fontsize=6, ha="center", va="bottom",
                       xytext=(0, 6), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="Permutation ΔP", shrink=0.7)
    ax.set_title("Scalp: Channel Permutation Importance")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "scalp_permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_scalp_high_confidence(combined_channels, hc_channels, montage_coords, out_dir):
    """Scalp plot highlighting high-confidence channels."""
    coords = montage_coords["coords"]
    hc_set = set(c["channel"] for c in hc_channels)

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_head_outline(ax)

    for c in combined_channels:
        ch = c["channel"]
        if ch not in coords:
            continue
        x, y = coords[ch]
        if ch in hc_set:
            if c["contribution_type"] == "facilitatory":
                color = "#2e7d32"
                marker = "^"
            else:
                color = "#c62828"
                marker = "v"
            ax.scatter(x, y, c=color, s=180, marker=marker, edgecolors="k",
                      linewidths=1.5, zorder=10)
            ax.annotate(c["electrode_name"], (x, y), fontsize=7, ha="center",
                       va="bottom", xytext=(0, 8), textcoords="offset points", fontweight="bold")
        else:
            ax.scatter(x, y, c="#bdbdbd", s=60, marker="o", edgecolors="gray",
                      linewidths=0.5, zorder=3, alpha=0.5)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2e7d32",
               markersize=12, label="High-conf facilitatory"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#c62828",
               markersize=12, label="High-conf suppressive"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bdbdbd",
               markersize=8, label="Other channels"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower left")
    ax.set_title("Scalp: High-Confidence Channels")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "scalp_high_confidence_channels.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_scalp_facilitatory_suppressive(combined_channels, montage_coords, out_dir):
    """Scalp plot showing facilitatory vs suppressive for all robust channels."""
    coords = montage_coords["coords"]
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_head_outline(ax)

    for c in combined_channels:
        ch = c["channel"]
        if ch not in coords:
            continue
        x, y = coords[ch]
        if not c["robust_significant"]:
            ax.scatter(x, y, c="#e0e0e0", s=40, marker="o", edgecolors="gray",
                      linewidths=0.3, zorder=2, alpha=0.4)
        elif c["contribution_type"] == "facilitatory":
            size = min(300, max(60, abs(c["occ_score"]) * 30000))
            ax.scatter(x, y, c="#2e7d32", s=size, marker="o", edgecolors="k",
                      linewidths=0.8, zorder=5, alpha=0.7)
        elif c["contribution_type"] == "suppressive":
            size = min(300, max(60, abs(c["occ_score"]) * 30000))
            ax.scatter(x, y, c="#c62828", s=size, marker="o", edgecolors="k",
                      linewidths=0.8, zorder=5, alpha=0.7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2e7d32", label="Facilitatory (robust)"),
        Patch(facecolor="#c62828", label="Suppressive (robust)"),
        Patch(facecolor="#e0e0e0", label="Not robust"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower left")
    ax.set_title("Scalp: Facilitatory vs Suppressive (robust channels)")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "scalp_facilitatory_suppressive.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_scalp_best_frequency_band(combined_channels, freq_by_channel, montage_coords, out_dir):
    """Scalp plot colored by best frequency band."""
    coords = montage_coords["coords"]
    freq_lookup = {r["channel"]: r for r in freq_by_channel}
    band_colors = {"delta": "#1565c0", "theta": "#2e7d32", "alpha": "#f57f17", "beta": "#d32f2f"}

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_head_outline(ax)

    for c in combined_channels:
        ch = c["channel"]
        if ch not in coords:
            continue
        x, y = coords[ch]
        fd = freq_lookup.get(ch)
        if fd and c["robust_significant"]:
            band = fd.get("most_important_band", "delta")
            color = band_colors.get(band, "#bdbdbd")
            ax.scatter(x, y, c=color, s=120, marker="o", edgecolors="k",
                      linewidths=0.8, zorder=5, alpha=0.8)
            ax.annotate(c["electrode_name"], (x, y), fontsize=5, ha="center",
                       va="bottom", xytext=(0, 5), textcoords="offset points")
        else:
            ax.scatter(x, y, c="#e0e0e0", s=30, marker="o", edgecolors="gray",
                      linewidths=0.3, zorder=2, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=b) for b, c in band_colors.items()]
    legend_elements.append(Patch(facecolor="#e0e0e0", label="Not analysed"))
    ax.legend(handles=legend_elements, fontsize=8, loc="lower left")
    ax.set_title("Scalp: Best Frequency Band (robust channels)")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "scalp_best_frequency_band.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_scalp_plots(combined_channels, hc_channels, freq_by_channel, montage_path, out_dir):
    """Generate all scalp plots if montage coordinates are available."""
    montage_coords = load_montage_coords(montage_path)
    if montage_coords is None:
        print("  WARNING: No valid montage coordinates — skipping scalp plots.")
        return False

    print("  Generating scalp plots...")
    plot_scalp_occlusion_importance(combined_channels, montage_coords, out_dir)
    plot_scalp_permutation_importance(combined_channels, montage_coords, out_dir)
    plot_scalp_high_confidence(combined_channels, hc_channels, montage_coords, out_dir)
    plot_scalp_facilitatory_suppressive(combined_channels, montage_coords, out_dir)
    if freq_by_channel:
        plot_scalp_best_frequency_band(combined_channels, freq_by_channel, montage_coords, out_dir)
    print("  Scalp plots saved.")
    return True


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
# Updated report
# ══════════════════════════════════════════════════════════════════════

def write_updated_report(out_dir: Path, args, combined_channels, hc_channels,
                         freq_by_channel, freq_by_roi, montage_validation,
                         hc_freq, hc_roi_freq, scalp_generated):
    """Rewrite FOCUSED_XAI_REPORT.txt with all new sections."""
    # Load existing data
    arch_info = load_json(out_dir / "architecture_summary.json")
    ablation = load_json(out_dir / "block_ablation.json")
    roi_results = load_json(out_dir / "roi_importance.json")
    run_config = load_json(out_dir / "run_config.json")

    now = datetime.now().strftime("%B %d, %Y")
    n_windows = run_config.get("n_windows", 200)
    n_boot = run_config.get("n_boot", 500)
    fdr_alpha = run_config.get("fdr_alpha", 0.05)
    stab_thresh = run_config.get("stability_threshold", 0.5)
    montage_source = run_config.get("montage_source", "index_fallback")
    montage_file = run_config.get("montage_file", "")

    n_robust = sum(1 for c in combined_channels if c.get("robust_significant"))
    n_hc = len(hc_channels)
    n_occ_fdr = sum(1 for c in combined_channels if c.get("occ_fdr_significant"))
    n_perm_fdr = sum(1 for c in combined_channels if c.get("perm_fdr_significant"))
    n_fac = sum(1 for c in combined_channels if c["contribution_type"] == "facilitatory")
    n_sup = sum(1 for c in combined_channels if c["contribution_type"] == "suppressive")
    n_mix = sum(1 for c in combined_channels if c["contribution_type"] == "mixed")
    n_occ_ci = sum(1 for c in combined_channels if c.get("occ_ci_significant"))
    n_perm_ci = sum(1 for c in combined_channels if c.get("perm_ci_significant"))

    occ_all = [c["occ_score"] for c in combined_channels]
    perm_all = [c["perm_score"] for c in combined_channels]
    r_occ_perm = float(np.corrcoef(occ_all, perm_all)[0, 1])

    block3_zero = ablation["block_3"]["zero_weights"]
    block3_perm = ablation["block_3"]["permute"]

    # Subject stability stats
    ch_matrix_path = out_dir / "subject_channel_matrix.npy"
    if ch_matrix_path.exists():
        ch_matrix = np.load(ch_matrix_path)
        if ch_matrix.shape[0] > 1:
            corr_mat = np.corrcoef(ch_matrix)
            triu = corr_mat[np.triu_indices(ch_matrix.shape[0], k=1)]
            subj_r_mean = float(triu.mean())
            subj_r_std = float(triu.std())
        else:
            subj_r_mean = subj_r_std = 0.0
    else:
        subj_r_mean = subj_r_std = 0.0

    L = []

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
    L.append(f"FDR significance level: α = {fdr_alpha}")
    L.append(f"Stability threshold (robust): {stab_thresh:.0%} of subjects must agree")
    L.append(f"High-confidence stability threshold: {args.high_confidence_stability_threshold:.0%}")
    L.append(f"High-confidence top-K: {args.high_confidence_top_k}")
    if montage_source == "montage_file":
        L.append(f"Electrode montage: {montage_file} (actual)")
    else:
        L.append(f"Electrode montage: INDEX-BASED FALLBACK (approximate)")
    L.append(f"Script: scripts/run_focused_xai.py + scripts/postprocess_focused_xai.py")
    L.append("")

    # A. OBJECTIVE
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
    L.append("  - Integrated Gradients (gradient-based attribution, corroboration only)")
    L.append("")
    L.append("Two-tier selection:")
    L.append("  - Robust channels: same sign + at least one FDR-sig + stability ≥ 50%")
    L.append(f"  - High-confidence channels: same sign + both FDR-sig + stability ≥ "
             f"{args.high_confidence_stability_threshold:.0%} + top-{args.high_confidence_top_k}")
    L.append("")

    # B. ARCHITECTURE
    L.append("=" * 80)
    L.append("B. ARCHITECTURE FINDING")
    L.append("=" * 80)
    L.append("")
    L.append(f"Total parameters: {arch_info['total_params']:,}")
    L.append(f"Shared parameters: {arch_info['shared_pct']*100:.1f}%")
    L.append("")
    L.append("Component breakdown:")
    for name, n in arch_info["components"].items():
        pct = int(n) / arch_info["total_params"] * 100
        L.append(f"  {name:25s}: {int(n):>10,} ({pct:.1f}%)")
    L.append("")
    L.append("Block ablation (key result):")
    L.append(f"  Block 3 zero_weights:  ΔP = {block3_zero['delta_p_mean']:+.5f}, "
             f"ΔAcc = {block3_zero['delta_acc']:+.3f}")
    L.append(f"  Block 3 permute:       ΔP = {block3_perm['delta_p_mean']:+.5f}, "
             f"ΔAcc = {block3_perm['delta_acc']:+.3f}")
    L.append("")
    L.append("KEY FINDING: Block 3 (final iteration) dominates the decision.")
    L.append("")

    # C. CHANNEL IMPORTANCE METHOD
    L.append("=" * 80)
    L.append("C. CHANNEL IMPORTANCE METHOD")
    L.append("=" * 80)
    L.append("")
    L.append(f"Occlusion–Permutation correlation: r = {r_occ_perm:.3f}")
    L.append(f"CI-based occlusion significance: {n_occ_ci}/64")
    L.append(f"CI-based permutation significance: {n_perm_ci}/64")
    L.append("")

    # D. MULTIPLE-COMPARISON CORRECTION
    L.append("=" * 80)
    L.append("D. MULTIPLE-COMPARISON CORRECTION")
    L.append("=" * 80)
    L.append("")
    L.append(f"FDR threshold: α = {fdr_alpha}")
    L.append(f"Occlusion FDR-significant:   {n_occ_fdr}/64")
    L.append(f"Permutation FDR-significant:  {n_perm_fdr}/64")
    L.append(f"Channels meeting robust criteria: {n_robust}/64")
    L.append("")

    # E. ROBUST CHANNELS
    L.append("=" * 80)
    L.append("E. ROBUST IMPORTANT CHANNELS")
    L.append("=" * 80)
    L.append("")
    L.append(f"{n_robust} channels meet robust criteria (same sign + ≥1 FDR-sig + stability ≥ {stab_thresh:.0%}).")
    L.append("See final_important_channels.csv for the full list.")
    L.append("")

    # F. HIGH-CONFIDENCE SUBSET
    L.append("=" * 80)
    L.append("F. HIGH-CONFIDENCE CHANNEL SUBSET")
    L.append("=" * 80)
    L.append("")
    L.append(f"Strict criteria applied to identify a smaller, supervisor-ready subset:")
    L.append(f"  1. Occlusion and permutation agree in sign")
    L.append(f"  2. Both occlusion AND permutation are FDR-significant")
    L.append(f"  3. Subject stability ≥ {args.high_confidence_stability_threshold:.0%}")
    L.append(f"  4. Channel rank ≤ {args.high_confidence_top_k} by combined importance")
    L.append("")
    L.append(f"High-confidence channels: {n_hc}/64")
    L.append("")
    if hc_channels:
        header = (f"{'Rank':>4} | {'Ch':>3} | {'Name':>6} | {'ROI':>16} | "
                  f"{'Occ ΔP':>9} | {'Perm ΔP':>9} | {'Type':>12} | {'Stable':>7}")
        L.append(header)
        L.append("-" * len(header))
        for i, c in enumerate(hc_channels):
            L.append(
                f"{i+1:>4} | {c['channel']:>3} | {c['electrode_name']:>6} | "
                f"{c['roi']:>16} | {c['occ_score']:>+9.5f} | {c['perm_score']:>+9.5f} | "
                f"{c['contribution_type']:>12} | {c['subject_stability']:>7}"
            )
        L.append("")
    else:
        L.append("No channels met all high-confidence criteria with current thresholds.")
        L.append("Consider relaxing --high-confidence-top-k or --high-confidence-stability-threshold.")
        L.append("")

    # G. FACILITATORY vs SUPPRESSIVE
    L.append("=" * 80)
    L.append("G. FACILITATORY vs SUPPRESSIVE INTERPRETATION")
    L.append("=" * 80)
    L.append("")
    L.append(f"Of 64 channels: {n_fac} facilitatory, {n_sup} suppressive, {n_mix} mixed")
    L.append("")
    hc_fac = [c for c in hc_channels if c["contribution_type"] == "facilitatory"]
    hc_sup = [c for c in hc_channels if c["contribution_type"] == "suppressive"]
    if hc_fac:
        L.append("High-confidence facilitatory channels:")
        for c in hc_fac:
            L.append(f"  {c['electrode_name']:>6} (Ch{c['channel']:2d}): Occ={c['occ_score']:+.5f}, Perm={c['perm_score']:+.5f}")
    if hc_sup:
        L.append("")
        L.append("High-confidence suppressive channels:")
        for c in hc_sup:
            L.append(f"  {c['electrode_name']:>6} (Ch{c['channel']:2d}): Occ={c['occ_score']:+.5f}, Perm={c['perm_score']:+.5f}")
    L.append("")

    # H. ROI INTERPRETATION
    L.append("=" * 80)
    L.append("H. ROI-LEVEL INTERPRETATION")
    L.append("=" * 80)
    L.append("")
    if montage_source == "montage_file":
        L.append(f"ROI mapping: from montage file ({montage_file})")
        L.append("NOTE: Verify against official DTU documentation before publication.")
    else:
        L.append("WARNING: ROI mapping is INDEX-BASED and approximate.")
    L.append("")
    header_roi = (f"{'ROI':>20} | {'Occ ΔP':>9} | {'Sig?':>5} | {'Perm ΔP':>9} | {'#FDR':>5} | {'#Robust':>7}")
    L.append(header_roi)
    L.append("-" * len(header_roi))
    for r in roi_results:
        L.append(f"{r['roi']:>20} | {r['occ_mean_dp']:>+9.5f} | "
                 f"{'YES' if r['occ_significant'] else 'no':>5} | "
                 f"{r['perm_mean_dp']:>+9.5f} | "
                 f"{r['n_fdr_significant_channels']:>5} | {r['n_robust_channels']:>7}")
    L.append("")

    # I. FREQUENCY (high-confidence)
    L.append("=" * 80)
    L.append("I. FREQUENCY CONTRIBUTION OF HIGH-CONFIDENCE CHANNELS")
    L.append("=" * 80)
    L.append("")
    if hc_freq:
        bands_header = (f"{'Name':>6} | {'Ch':>3} | {'ROI':>16} | {'Type':>12} | "
                       f"{'Best Band':>10} | {'δ':>8} | {'θ':>8} | {'α':>8} | {'β':>8}")
        L.append(bands_header)
        L.append("-" * len(bands_header))
        for r in hc_freq:
            L.append(f"{r['electrode_name']:>6} | {r['channel_index']:>3} | {r['roi']:>16} | "
                     f"{r['contribution_type']:>12} | {r['best_frequency_band']:>10} | "
                     f"{float(r.get('delta_score', 0) or 0):>+8.5f} | "
                     f"{float(r.get('theta_score', 0) or 0):>+8.5f} | "
                     f"{float(r.get('alpha_score', 0) or 0):>+8.5f} | "
                     f"{float(r.get('beta_score', 0) or 0):>+8.5f}")
        L.append("")
    else:
        L.append("No high-confidence channels with frequency data available.")
        L.append("")

    # J. SCALP VISUALISATION
    L.append("=" * 80)
    L.append("J. SCALP / TOPOGRAPHIC VISUALISATION")
    L.append("=" * 80)
    L.append("")
    if scalp_generated:
        L.append("Scalp plots generated using 2D electrode coordinates from montage file:")
        L.append("  scalp_occlusion_importance.png")
        L.append("  scalp_permutation_importance.png")
        L.append("  scalp_high_confidence_channels.png")
        L.append("  scalp_facilitatory_suppressive.png")
        L.append("  scalp_best_frequency_band.png")
    else:
        L.append("Scalp plots NOT generated (missing or invalid montage coordinates).")
    L.append("")

    # K. MONTAGE VALIDATION
    L.append("=" * 80)
    L.append("K. MONTAGE VALIDATION")
    L.append("=" * 80)
    L.append("")
    if montage_validation["valid"]:
        L.append(f"Montage validation: PASSED")
    else:
        L.append(f"Montage validation: FAILED")
    L.append(f"  File: {montage_validation['montage_path']}")
    L.append(f"  Channels: {montage_validation['n_channels']}")
    L.append(f"  ROIs: {montage_validation['n_rois']}")
    L.append(f"  Has coordinates: {montage_validation['has_coordinates']}")
    if montage_validation["warnings"]:
        L.append("  Warnings:")
        for w in montage_validation["warnings"]:
            L.append(f"    - {w}")
    if montage_validation["errors"]:
        L.append("  Errors:")
        for e in montage_validation["errors"]:
            L.append(f"    - {e}")
    L.append("")
    L.append("IMPORTANT: Electrode-level neuroscience claims require verification of the")
    L.append("channel order against the official DTU dataset documentation. The montage file")
    L.append("used here is a best-effort mapping and must be independently confirmed.")
    L.append("")

    # L. LIMITATIONS
    L.append("=" * 80)
    L.append("L. LIMITATIONS")
    L.append("=" * 80)
    L.append("")
    L.append(f"1. SAMPLE SIZE: N={n_windows}. Recommend N≥500 for publication.")
    L.append(f"2. UNATTENDED ENVELOPE: Circular time-shift proxy, not true competing speaker.")
    L.append(f"3. MONTAGE: Must be verified against official DTU documentation.")
    L.append(f"4. IG: Zero baseline; supporting evidence only.")
    L.append(f"5. STABILITY: Limited windows per subject.")
    L.append(f"6. CROSS-SUBJECT: Mean pairwise r = {subj_r_mean:.3f} ± {subj_r_std:.3f}.")
    L.append(f"7. FDR: p-value resolution limited to 1/{n_boot}.")
    L.append("")

    # M. NEXT STEPS
    L.append("=" * 80)
    L.append("M. NEXT STEPS")
    L.append("=" * 80)
    L.append("")
    L.append("1. Verify electrode montage against official DTU dataset documentation.")
    L.append("2. Rerun with larger N for publication-quality results:")
    L.append("")
    L.append("   Suggested commands:")
    L.append("   python scripts/run_focused_xai.py --max-samples 500 --n-boot 2000 \\")
    L.append("     --ig-samples 50 --windows-per-subject 20 \\")
    L.append("     --montage-file config/dtu_channel_montage.csv \\")
    L.append("     --output-dir xai_results_focused_n500")
    L.append("")
    L.append("   python scripts/run_focused_xai.py --max-samples 1000 --n-boot 2000 \\")
    L.append("     --ig-samples 100 --windows-per-subject 30 \\")
    L.append("     --montage-file config/dtu_channel_montage.csv \\")
    L.append("     --output-dir xai_results_focused_n1000")
    L.append("")
    L.append("3. After rerun, apply post-processing:")
    L.append("   python scripts/postprocess_focused_xai.py --output-dir xai_results_focused_n500")
    L.append("")
    L.append("4. Add topographic scalp maps with verified coordinates.")
    L.append("5. Validate findings with TRF baseline decoder.")
    L.append("")

    # N. OUTPUT FILES
    L.append("=" * 80)
    L.append("N. OUTPUT FILES")
    L.append("=" * 80)
    L.append("")
    L.append(f"Output directory: {out_dir}")
    L.append("")
    L.append("  channel_importance.csv              — Full 64-channel table with FDR + HC flags")
    L.append("  high_confidence_channels.csv        — High-confidence channels only")
    L.append("  supervisor_channel_summary.csv      — Compact summary for supervisor review")
    L.append("  final_important_channels.csv        — Robust channels (broader set)")
    L.append("  frequency_by_high_confidence_channels.csv")
    L.append("  frequency_by_high_confidence_roi.csv")
    L.append("  frequency_high_confidence_heatmap.png")
    L.append("  scalp_*.png                         — Topographic scalp plots")
    L.append("  montage_validation.json / .txt")
    L.append("  FOCUSED_XAI_REPORT.txt              — This report")
    L.append("  run_config.json                     — Full configuration")
    L.append("")
    L.append("=" * 80)
    L.append("END OF REPORT")
    L.append("=" * 80)

    report_text = "\n".join(L)
    (out_dir / "FOCUSED_XAI_REPORT.txt").write_text(report_text, encoding="utf-8")
    print(f"  Report saved ({len(L)} lines)")
    return report_text


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)

    print("=" * 70)
    print("POST-PROCESSING FOCUSED XAI RESULTS")
    print(f"  Output dir: {out_dir}")
    print(f"  Montage: {args.montage_file}")
    print(f"  High-confidence top-K: {args.high_confidence_top_k}")
    print(f"  High-confidence stability: {args.high_confidence_stability_threshold:.0%}")
    print(f"  Require both FDR: {args.high_confidence_require_both_fdr}")
    print("=" * 70)

    # Load existing results
    ci_path = out_dir / "channel_importance.json"
    if not ci_path.exists():
        print(f"ERROR: {ci_path} not found. Run scripts/run_focused_xai.py first.")
        sys.exit(1)

    ci_data = load_json(ci_path)
    combined_channels = ci_data["channels"]

    freq_path = out_dir / "frequency_analysis.json"
    freq_by_channel = []
    freq_by_roi = []
    if freq_path.exists():
        freq_data = load_json(freq_path)
        freq_by_channel = freq_data.get("frequency_by_channel", [])
        freq_by_roi = freq_data.get("frequency_by_roi", [])

    # Load montage info
    montage_data = {"rois": {}}
    if Path(args.montage_file).is_file():
        with open(args.montage_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rois = OrderedDict()
            for row in reader:
                roi = row["roi"]
                idx = int(row["channel_index"])
                rois.setdefault(roi, []).append(idx)
            montage_data["rois"] = rois

    # 1. Montage validation
    print("\n[1] Montage validation...")
    montage_validation = validate_montage(args.montage_file)
    write_montage_validation_report(montage_validation, out_dir)
    if not montage_validation["valid"] and args.strict_montage_validation:
        print("  ERROR: Montage validation failed in strict mode. Aborting.")
        sys.exit(1)

    # 2. High-confidence selection
    print("\n[2] High-confidence channel selection...")
    combined_channels = select_high_confidence_channels(combined_channels, args)
    hc_channels = save_high_confidence_channels(combined_channels, out_dir)

    # Also update channel_importance.csv with new columns
    csv_fields = [
        "rank", "channel", "electrode_name", "roi", "roi_mapping_source",
        "occ_score", "occ_ci_lo", "occ_ci_hi", "occ_p_value", "occ_fdr_p_value",
        "occ_fdr_significant", "occ_ci_significant",
        "perm_score", "perm_ci_lo", "perm_ci_hi", "perm_p_value", "perm_fdr_p_value",
        "perm_fdr_significant", "perm_ci_significant",
        "ig_rank", "ig_importance",
        "contribution_type", "subject_stability", "stability_fraction",
        "robust_significant", "combined_score",
        "high_confidence", "high_confidence_reason", "high_confidence_failed_criteria",
        "combined_abs_importance_rank", "both_fdr_significant", "same_sign_occ_perm",
    ]
    combined_sorted = sorted(combined_channels, key=lambda x: x.get("rank", 999))
    with open(out_dir / "channel_importance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for c in combined_sorted:
            w.writerow({k: c.get(k, "") for k in csv_fields})
    print(f"  Updated channel_importance.csv with high_confidence columns")

    # 3. Supervisor summary
    print("\n[3] Building supervisor summary...")
    freq_lookup = {r["channel"]: r for r in freq_by_channel}
    supervisor_rows = build_supervisor_channel_summary(hc_channels, freq_lookup, out_dir)

    # 4. Frequency by high-confidence
    print("\n[4] Frequency analysis for high-confidence channels...")
    hc_freq, hc_roi_freq = compute_frequency_by_high_confidence(
        hc_channels, freq_by_channel, montage_data, out_dir)

    # 5. Scalp plots
    print("\n[5] Scalp plots...")
    scalp_generated = save_scalp_plots(
        combined_sorted, hc_channels, freq_by_channel, args.montage_file, out_dir)

    # 6. Updated report
    print("\n[6] Writing updated report...")
    write_updated_report(out_dir, args, combined_sorted, hc_channels,
                         freq_by_channel, freq_by_roi, montage_validation,
                         hc_freq, hc_roi_freq, scalp_generated)

    print("\n" + "=" * 70)
    print("POST-PROCESSING COMPLETE")
    print(f"  High-confidence channels: {len(hc_channels)}/64")
    print(f"  Scalp plots: {'generated' if scalp_generated else 'skipped'}")
    print(f"  All outputs in: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
