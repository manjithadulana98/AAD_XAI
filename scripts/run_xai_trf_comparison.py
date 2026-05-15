"""TRF vs VLAAI XAI Comparison.

Runs the model-agnostic XAI sections (C, D, H, I, J) on a freshly trained
TRF decoder, then loads existing VLAAI results and generates a side-by-side
comparison report.

Usage:
    python scripts/run_xai_trf_comparison.py
    python scripts/run_xai_trf_comparison.py --max-samples 200 --n-boot 1000
    python scripts/run_xai_trf_comparison.py --sections C D
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# Constants (same as run_xai_comprehensive.py)
# ══════════════════════════════════════════════════════════════════════
FS = 64
ROIS = OrderedDict([
    ("Frontal",        list(range(0, 12))),
    ("Fronto-Central", list(range(12, 18))),
    ("Central",        list(range(18, 30))),
    ("Temporal",       list(range(30, 42))),
    ("Parietal",       list(range(42, 54))),
    ("Occipital",      list(range(54, 64))),
])

ALL_SECTIONS = list("CDHIJ")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="TRF vs VLAAI XAI Comparison")
    p.add_argument("--data-dir", type=str,
                    default=str(ROOT / "external" / "vlaai" / "evaluation_datasets" / "DTU"))
    p.add_argument("--vlaai-results", type=str, default=str(ROOT / "xai_results"),
                    help="Directory with existing VLAAI XAI results.")
    p.add_argument("--output-dir", type=str, default=str(ROOT / "xai_results_trf_comparison"))
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--windows-per-subject", type=int, default=50)
    p.add_argument("--trf-tmin", type=float, default=0.0)
    p.add_argument("--trf-tmax", type=float, default=0.5)
    p.add_argument("--trf-alpha", type=float, default=100.0)
    p.add_argument("--train-ratio", type=float, default=0.7,
                    help="Fraction of windows to use for TRF training (rest for XAI).")
    p.add_argument("--sections", nargs="*", default=ALL_SECTIONS,
                    help="Sections to run (C,D,H,I,J). Default: all.")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ══════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════
def make_output_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "root": base,
        "trf_xai": base / "trf_xai",
        "channel_occlusion": base / "trf_xai" / "C_channel_occlusion",
        "temporal_occlusion": base / "trf_xai" / "D_temporal_occlusion",
        "subject_wise": base / "trf_xai" / "H_subject_wise",
        "frequency_band": base / "trf_xai" / "I_frequency_band",
        "correct_incorrect": base / "trf_xai" / "J_correct_incorrect",
        "comparison": base / "comparison",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
# Shared analysis utilities (same as run_xai_comprehensive.py)
# ══════════════════════════════════════════════════════════════════════
def get_attended_prob(decision, eeg, att, unatt):
    """P(attended) for a batch — works for both VLAAI and TRF wrappers."""
    decision.set_envelopes(att, unatt)
    with torch.no_grad():
        logits = decision(eeg)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95,
                  seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, len(values), size=len(values))
        means[b] = values[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(values.mean()), float(lo), float(hi)


# ══════════════════════════════════════════════════════════════════════
# TRF training
# ══════════════════════════════════════════════════════════════════════
def train_trf(ds, train_indices, tmin, tmax, alpha, sfreq=64.0):
    """Train a TRFDecoder on the given dataset windows."""
    from aad_xai.models import TRFDecoder

    print(f"  Training TRF on {len(train_indices)} windows (tmin={tmin}, tmax={tmax}, alpha={alpha})...")

    # Concatenate training windows into long signals
    eeg_parts, env_parts = [], []
    for idx in train_indices:
        eeg_w, att_w, _, _ = ds[idx]
        eeg_parts.append(eeg_w.numpy())     # (T, 64)
        env_parts.append(att_w.numpy()[:, 0])  # (T,)

    eeg_concat = np.concatenate(eeg_parts, axis=0)  # (N*T, 64)
    env_concat = np.concatenate(env_parts, axis=0)   # (N*T,)

    trf = TRFDecoder(tmin_s=tmin, tmax_s=tmax, alpha=alpha)
    trf.fit(eeg_concat.T, env_concat, sfreq)  # fit expects (n_channels, n_times)

    # Quick validation
    pred = trf.predict(eeg_concat.T)
    n = min(len(pred), len(env_concat))
    r = np.corrcoef(pred[:n], env_concat[:n])[0, 1]
    print(f"  TRF training correlation: r = {r:.4f}")

    return trf


# ══════════════════════════════════════════════════════════════════════
# SECTION C — Channel / ROI occlusion
# ══════════════════════════════════════════════════════════════════════
def section_c_trf(decision, eeg, att, unatt, n_boot, seed, dirs):
    print("\n" + "=" * 70)
    print("SECTION C (TRF): CHANNEL / ROI OCCLUSION WITH BOOTSTRAP CIs")
    print("=" * 70)
    out = dirs["channel_occlusion"]
    N = eeg.shape[0]

    base_probs = get_attended_prob(decision, eeg, att, unatt)

    channel_drops_pw = np.zeros((N, 64))
    for ch in range(64):
        eeg_m = eeg.clone()
        eeg_m[:, :, ch] = 0.0
        m_probs = get_attended_prob(decision, eeg_m, att, unatt)
        channel_drops_pw[:, ch] = base_probs - m_probs
        if (ch + 1) % 16 == 0:
            print(f"    ... channel {ch + 1}/64 done")

    ch_results = []
    for ch in range(64):
        mean, lo, hi = bootstrap_ci(channel_drops_pw[:, ch], n_boot, seed=seed)
        ch_results.append({"channel": ch, "mean_dp": mean, "ci_lo": lo, "ci_hi": hi})

    roi_results = []
    for roi_name, chs in ROIS.items():
        roi_drops = channel_drops_pw[:, chs].mean(axis=1)
        mean, lo, hi = bootstrap_ci(roi_drops, n_boot, seed=seed)
        roi_results.append({"roi": roi_name, "mean_dp": mean, "ci_lo": lo, "ci_hi": hi, "channels": chs})

    save_json({"channels": ch_results, "rois": roi_results}, out / "channel_occlusion.json")
    np.save(out / "channel_drops_perwindow.npy", channel_drops_pw)

    with open(out / "channel_occlusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "mean_dp", "ci_lo", "ci_hi"])
        w.writeheader()
        w.writerows(ch_results)

    sorted_ch = sorted(ch_results, key=lambda x: abs(x["mean_dp"]), reverse=True)
    print("  Top-15 channels by |ΔP|:")
    for r in sorted_ch[:15]:
        print(f"    Ch {r['channel']:2d}: {r['mean_dp']:+.5f} [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]")

    print("\n  ROI summary:")
    for r in roi_results:
        print(f"    {r['roi']:20s}: ΔP = {r['mean_dp']:+.5f} [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]")

    return ch_results, roi_results


# ══════════════════════════════════════════════════════════════════════
# SECTION D — Temporal occlusion
# ══════════════════════════════════════════════════════════════════════
def section_d_trf(decision, eeg, att, unatt, n_boot, seed, dirs):
    print("\n" + "=" * 70)
    print("SECTION D (TRF): TEMPORAL OCCLUSION WITH CIs + SIGNIFICANCE TESTS")
    print("=" * 70)
    out = dirs["temporal_occlusion"]
    N = eeg.shape[0]

    base_probs = get_attended_prob(decision, eeg, att, unatt)

    win_t, hop_t = 64, 32
    T = eeg.shape[1]
    starts = list(range(0, T - win_t + 1, hop_t))
    n_starts = len(starts)

    temporal_pw = np.zeros((N, n_starts))
    for si, s in enumerate(starts):
        eeg_m = eeg.clone()
        eeg_m[:, s:s + win_t, :] = 0.0
        m_probs = get_attended_prob(decision, eeg_m, att, unatt)
        temporal_pw[:, si] = base_probs - m_probs

    t_results = []
    for si, s in enumerate(starts):
        mean, lo, hi = bootstrap_ci(temporal_pw[:, si], n_boot, seed=seed)
        t_results.append({
            "start_sample": int(s), "start_sec": s / FS,
            "end_sec": (s + win_t) / FS,
            "mean_dp": mean, "ci_lo": lo, "ci_hi": hi,
        })

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

    sorted_t = sorted(t_results, key=lambda x: abs(x["mean_dp"]), reverse=True)
    print("  Temporal sensitivity (1s mask, 0.5s hop):")
    for r in sorted_t[:8]:
        print(f"    t={r['start_sec']:.1f}-{r['end_sec']:.1f}s: "
              f"ΔP = {r['mean_dp']:+.5f} [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]")
    print(f"\n  Early vs Late: early={sig_test['early_mean']:+.5f}, "
          f"late={sig_test['late_mean']:+.5f}, p={sig_test['p_value']:.4f}")

    return t_results, sig_test


# ══════════════════════════════════════════════════════════════════════
# SECTION H — Subject-wise consistency
# ══════════════════════════════════════════════════════════════════════
def section_h_trf(decision, ds, n_boot, seed, dirs, windows_per_subject=50):
    print("\n" + "=" * 70)
    print("SECTION H (TRF): SUBJECT-WISE CONSISTENCY & CLUSTERING")
    print("=" * 70)
    out = dirs["subject_wise"]

    all_subject_ids = ds.subject_ids
    unique_subjects = sorted(set(all_subject_ids))
    print(f"  {len(unique_subjects)} subjects found: {unique_subjects}")

    if len(unique_subjects) < 2:
        print("  Need >= 2 subjects. Skipping.")
        return None, None

    subj_ch_profiles = {}
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

    subj_list = sorted(subj_ch_profiles.keys())
    ch_matrix = np.array([subj_ch_profiles[s] for s in subj_list])
    corr_matrix = np.corrcoef(ch_matrix)
    triu_idx = np.triu_indices(len(subj_list), k=1)
    pairwise = corr_matrix[triu_idx]

    print(f"\n  Cross-subject consistency (pairwise r):")
    print(f"    Mean = {pairwise.mean():.3f} +/- {pairwise.std():.3f}")
    print(f"    Range = [{pairwise.min():.3f}, {pairwise.max():.3f}]")

    save_json({
        "subjects": subj_list,
        "pairwise_r_mean": float(pairwise.mean()),
        "pairwise_r_std": float(pairwise.std()),
        "roi_profiles": subj_roi_profiles,
    }, out / "subject_consistency.json")
    np.save(out / "subject_channel_matrix.npy", ch_matrix)
    np.save(out / "subject_corr_matrix.npy", corr_matrix)

    return subj_ch_profiles, subj_roi_profiles


# ══════════════════════════════════════════════════════════════════════
# SECTION I — Frequency-band analysis
# ══════════════════════════════════════════════════════════════════════
def section_i_trf(decision, eeg, att, unatt, n_boot, seed, dirs):
    print("\n" + "=" * 70)
    print("SECTION I (TRF): FREQUENCY-BAND ANALYSIS (BUTTERWORTH)")
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
    pad_samples = 64

    band_results = []
    band_roi_results = []

    for band_name, (lo, hi) in BANDS.items():
        nyq = FS / 2.0
        lo_n = max(lo / nyq, 0.01)
        hi_n = min(hi / nyq, 0.99)
        sos = butter(4, [lo_n, hi_n], btype="bandpass", output="sos")

        eeg_np = eeg.numpy()
        T = eeg_np.shape[1]

        band_content = np.zeros_like(eeg_np)
        for w in range(N):
            for ch in range(64):
                sig = eeg_np[w, :, ch]
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

        roi_row = {"band": band_name}
        for roi_name, chs in ROIS.items():
            eeg_roi_np = eeg_np.copy()
            for w in range(N):
                for ch_idx in chs:
                    sig = eeg_roi_np[w, :, ch_idx]
                    padded_sig = np.pad(sig, pad_samples, mode="reflect")
                    filtered_sig = sosfiltfilt(sos, padded_sig)
                    eeg_roi_np[w, :, ch_idx] -= filtered_sig[pad_samples:pad_samples + T]

            p_roi = get_attended_prob(decision, torch.from_numpy(eeg_roi_np.astype(np.float32)),
                                      att, unatt)
            roi_dp = (base_probs - p_roi).mean()
            roi_row[roi_name] = float(roi_dp)
        band_roi_results.append(roi_row)

        print(f"  {band_name:8s} ({lo:.1f}-{hi:.1f} Hz): "
              f"ΔP = {mean:+.5f} [{lo_ci:+.5f}, {hi_ci:+.5f}]")

    save_json({"bands": band_results, "band_roi": band_roi_results},
              out / "frequency_band.json")

    return band_results, band_roi_results


# ══════════════════════════════════════════════════════════════════════
# SECTION J — Correct vs incorrect
# ══════════════════════════════════════════════════════════════════════
def section_j_trf(decision, trf, eeg, att, unatt, n_boot, seed, dirs):
    print("\n" + "=" * 70)
    print("SECTION J (TRF): CORRECT vs INCORRECT (CORRELATION-BASED LABELS)")
    print("=" * 70)
    out = dirs["correct_incorrect"]
    N = eeg.shape[0]

    # TRF predictions
    eeg_np = eeg.numpy()
    corr_att = np.zeros(N)
    corr_unatt = np.zeros(N)
    for i in range(N):
        eeg_i = eeg_np[i].T  # (C, T)
        pred_env = trf.predict(eeg_i)
        att_i = att[i, :, 0].numpy()
        unatt_i = unatt[i, :, 0].numpy()
        n = min(len(pred_env), len(att_i))
        r_a = np.corrcoef(pred_env[:n], att_i[:n])[0, 1]
        r_u = np.corrcoef(pred_env[:n], unatt_i[:n])[0, 1]
        corr_att[i] = r_a if np.isfinite(r_a) else 0.0
        corr_unatt[i] = r_u if np.isfinite(r_u) else 0.0

    correct_mask = corr_att > corr_unatt
    n_correct = correct_mask.sum()
    n_incorrect = (~correct_mask).sum()
    accuracy = n_correct / N

    print(f"  Correct (r_att > r_unatt): {n_correct}/{N} = {accuracy:.1%}")
    print(f"  Mean r_att = {corr_att.mean():.4f}, Mean r_unatt = {corr_unatt.mean():.4f}")

    if n_correct < 5 or n_incorrect < 5:
        print("  Need >= 5 in each group. Skipping detailed analysis.")
        save_json({"n_correct": int(n_correct), "n_incorrect": int(n_incorrect),
                    "accuracy": float(accuracy)}, out / "correct_incorrect.json")
        return

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

    top10_correct = set(np.argsort(np.abs(group_ch_drops["correct"]))[-10:])
    top10_incorrect = set(np.argsort(np.abs(group_ch_drops["incorrect"]))[-10:])
    overlap = len(top10_correct & top10_incorrect)

    from scipy.stats import ttest_ind
    t_stat, t_pval = ttest_ind(
        np.abs(group_ch_drops["correct"]),
        np.abs(group_ch_drops["incorrect"]),
    )

    results = {
        "n_correct": int(n_correct), "n_incorrect": int(n_incorrect),
        "accuracy": float(accuracy),
        "mean_r_att": float(corr_att.mean()),
        "mean_r_unatt": float(corr_unatt.mean()),
        "top10_overlap": overlap,
        "channel_diff_t_stat": float(t_stat),
        "channel_diff_p_value": float(t_pval),
        "roi_correct": group_roi["correct"],
        "roi_incorrect": group_roi["incorrect"],
    }
    save_json(results, out / "correct_incorrect.json")

    print(f"  Top-10 channel overlap: {overlap}/10")
    print(f"  Channel importance difference: t={t_stat:.3f}, p={t_pval:.4f}")

    print("\n  ROI comparison:")
    for roi in ROIS:
        c = group_roi["correct"][roi]
        ic = group_roi["incorrect"][roi]
        print(f"    {roi:20s}: correct={c:.5f}, incorrect={ic:.5f}, diff={c-ic:+.5f}")


# ══════════════════════════════════════════════════════════════════════
# Comparison report generation
# ══════════════════════════════════════════════════════════════════════
def generate_comparison(vlaai_dir: Path, trf_dir: Path, out_dir: Path):
    """Load VLAAI and TRF results, generate side-by-side report + plots."""
    print("\n" + "=" * 70)
    print("GENERATING VLAAI vs TRF COMPARISON REPORT")
    print("=" * 70)

    report_lines = [
        "=" * 70,
        "VLAAI vs TRF — XAI COMPARISON REPORT",
        "=" * 70,
        "",
    ]

    # ── Section C: Channel / ROI Occlusion ────────────────────────────
    report_lines.extend(_compare_section_c(vlaai_dir, trf_dir, out_dir))

    # ── Section D: Temporal Occlusion ─────────────────────────────────
    report_lines.extend(_compare_section_d(vlaai_dir, trf_dir, out_dir))

    # ── Section H: Subject-wise ───────────────────────────────────────
    report_lines.extend(_compare_section_h(vlaai_dir, trf_dir, out_dir))

    # ── Section I: Frequency Bands ────────────────────────────────────
    report_lines.extend(_compare_section_i(vlaai_dir, trf_dir, out_dir))

    # ── Section J: Correct vs Incorrect ───────────────────────────────
    report_lines.extend(_compare_section_j(vlaai_dir, trf_dir, out_dir))

    report = "\n".join(report_lines)
    report_path = out_dir / "COMPARISON_REPORT.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Full report saved to {report_path}")
    print(report)


def _load_if_exists(path):
    if Path(path).exists():
        return load_json(path)
    return None


def _compare_section_c(vlaai_dir, trf_dir, out_dir):
    lines = ["", "=" * 60, "SECTION C: CHANNEL / ROI OCCLUSION COMPARISON", "=" * 60, ""]

    v = _load_if_exists(vlaai_dir / "C_channel_occlusion" / "channel_occlusion.json")
    t = _load_if_exists(trf_dir / "C_channel_occlusion" / "channel_occlusion.json")

    if not v or not t:
        lines.append("  [SKIPPED] Missing results for one or both models.")
        return lines

    # ROI comparison table
    lines.append("ROI Importance (|ΔP|):")
    lines.append(f"  {'ROI':20s}  {'VLAAI':>10s}  {'TRF':>10s}  {'Diff':>10s}")
    lines.append("  " + "-" * 55)
    v_rois = {r["roi"]: r for r in v["rois"]}
    t_rois = {r["roi"]: r for r in t["rois"]}
    for roi in ROIS:
        v_dp = abs(v_rois[roi]["mean_dp"]) if roi in v_rois else 0
        t_dp = abs(t_rois[roi]["mean_dp"]) if roi in t_rois else 0
        lines.append(f"  {roi:20s}  {v_dp:10.5f}  {t_dp:10.5f}  {v_dp - t_dp:+10.5f}")

    # Top-10 channel overlap
    v_ch = sorted(v["channels"], key=lambda x: abs(x["mean_dp"]), reverse=True)
    t_ch = sorted(t["channels"], key=lambda x: abs(x["mean_dp"]), reverse=True)
    v_top10 = set(r["channel"] for r in v_ch[:10])
    t_top10 = set(r["channel"] for r in t_ch[:10])
    overlap = v_top10 & t_top10
    lines.append(f"\n  Top-10 channel overlap: {len(overlap)}/10  {sorted(overlap)}")

    # Rank correlation
    v_ranks = np.argsort(np.argsort([abs(r["mean_dp"]) for r in v["channels"]]))[::-1]
    t_ranks = np.argsort(np.argsort([abs(r["mean_dp"]) for r in t["channels"]]))[::-1]
    from scipy.stats import spearmanr
    rho, p = spearmanr(v_ranks, t_ranks)
    lines.append(f"  Channel rank correlation (Spearman): rho={rho:.3f}, p={p:.4f}")

    # Comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Side-by-side channel bars
    ax = axes[0]
    v_means = [r["mean_dp"] for r in v["channels"]]
    t_means = [r["mean_dp"] for r in t["channels"]]
    x = np.arange(64)
    ax.bar(x - 0.2, v_means, 0.4, label="VLAAI", color="#1976d2", alpha=0.7)
    ax.bar(x + 0.2, t_means, 0.4, label="TRF", color="#d32f2f", alpha=0.7)
    ax.set_xlabel("Channel")
    ax.set_ylabel("ΔP(attended)")
    ax.set_title("Channel Occlusion: VLAAI vs TRF")
    ax.legend()
    ax.axhline(0, color="k", linewidth=0.5)

    # ROI comparison
    ax = axes[1]
    roi_names = list(ROIS.keys())
    v_roi_vals = [abs(v_rois.get(r, {}).get("mean_dp", 0)) for r in roi_names]
    t_roi_vals = [abs(t_rois.get(r, {}).get("mean_dp", 0)) for r in roi_names]
    x_r = np.arange(len(roi_names))
    ax.bar(x_r - 0.15, v_roi_vals, 0.3, label="VLAAI", color="#1976d2", alpha=0.8)
    ax.bar(x_r + 0.15, t_roi_vals, 0.3, label="TRF", color="#d32f2f", alpha=0.8)
    ax.set_xticks(x_r)
    ax.set_xticklabels(roi_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|ΔP|")
    ax.set_title("ROI Importance: VLAAI vs TRF")
    ax.legend()

    # Scatter: VLAAI vs TRF per channel
    ax = axes[2]
    ax.scatter(v_means, t_means, alpha=0.5, s=20, color="#6a1b9a")
    ax.set_xlabel("VLAAI ΔP")
    ax.set_ylabel("TRF ΔP")
    ax.set_title(f"Channel ΔP Correlation (ρ={rho:.3f})")
    lim = max(abs(min(v_means + t_means)), abs(max(v_means + t_means))) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.5)
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "comparison_channel_occlusion.png", dpi=150, bbox_inches="tight")
    plt.close()

    return lines


def _compare_section_d(vlaai_dir, trf_dir, out_dir):
    lines = ["", "=" * 60, "SECTION D: TEMPORAL OCCLUSION COMPARISON", "=" * 60, ""]

    v = _load_if_exists(vlaai_dir / "D_temporal_occlusion" / "temporal_occlusion.json")
    t = _load_if_exists(trf_dir / "D_temporal_occlusion" / "temporal_occlusion.json")

    if not v or not t:
        lines.append("  [SKIPPED] Missing results for one or both models.")
        return lines

    lines.append("Early vs Late comparison:")
    lines.append(f"  {'':15s}  {'Early ΔP':>12s}  {'Late ΔP':>12s}  {'p-value':>10s}")
    lines.append("  " + "-" * 55)
    for name, d in [("VLAAI", v), ("TRF", t)]:
        ev = d["early_vs_late"]
        lines.append(f"  {name:15s}  {ev['early_mean']:12.5f}  {ev['late_mean']:12.5f}  {ev['p_value']:10.4f}")

    # Plot temporal profiles
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    v_t = v["temporal"]
    t_t = t["temporal"]
    ax.plot([r["start_sec"] for r in v_t], [r["mean_dp"] for r in v_t],
            color="#1976d2", linewidth=2, label="VLAAI")
    ax.fill_between([r["start_sec"] for r in v_t],
                     [r["ci_lo"] for r in v_t], [r["ci_hi"] for r in v_t],
                     alpha=0.15, color="#1976d2")
    ax.plot([r["start_sec"] for r in t_t], [r["mean_dp"] for r in t_t],
            color="#d32f2f", linewidth=2, label="TRF")
    ax.fill_between([r["start_sec"] for r in t_t],
                     [r["ci_lo"] for r in t_t], [r["ci_hi"] for r in t_t],
                     alpha=0.15, color="#d32f2f")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔP(attended)")
    ax.set_title("Temporal Occlusion: VLAAI vs TRF")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_temporal_occlusion.png", dpi=150, bbox_inches="tight")
    plt.close()

    return lines


def _compare_section_h(vlaai_dir, trf_dir, out_dir):
    lines = ["", "=" * 60, "SECTION H: SUBJECT-WISE CONSISTENCY COMPARISON", "=" * 60, ""]

    v = _load_if_exists(vlaai_dir / "H_subject_wise" / "subject_consistency.json")
    t = _load_if_exists(trf_dir / "H_subject_wise" / "subject_consistency.json")

    if not v or not t:
        lines.append("  [SKIPPED] Missing results for one or both models.")
        return lines

    lines.append("Cross-subject consistency (pairwise correlation):")
    lines.append(f"  {'Model':10s}  {'Mean r':>10s}  {'Std r':>10s}")
    lines.append("  " + "-" * 35)
    lines.append(f"  {'VLAAI':10s}  {v['pairwise_r_mean']:10.3f}  {v['pairwise_r_std']:10.3f}")
    lines.append(f"  {'TRF':10s}  {t['pairwise_r_mean']:10.3f}  {t['pairwise_r_std']:10.3f}")

    # ROI profile comparison across subjects
    if "roi_profiles" in v and "roi_profiles" in t:
        common_subj = sorted(set(v.get("subjects", [])) & set(t.get("subjects", [])))
        if common_subj:
            lines.append(f"\n  Common subjects: {len(common_subj)}")
            lines.append(f"\n  Top ROI per subject:")
            lines.append(f"  {'Subject':8s}  {'VLAAI Top ROI':20s}  {'TRF Top ROI':20s}  {'Match':>5s}")
            lines.append("  " + "-" * 60)
            agree = 0
            for s in common_subj:
                v_roi = v["roi_profiles"].get(s, {})
                t_roi = t["roi_profiles"].get(s, {})
                v_top = max(v_roi, key=v_roi.get) if v_roi else "N/A"
                t_top = max(t_roi, key=t_roi.get) if t_roi else "N/A"
                match = "Yes" if v_top == t_top else "No"
                if v_top == t_top:
                    agree += 1
                lines.append(f"  {s:8s}  {v_top:20s}  {t_top:20s}  {match:>5s}")
            lines.append(f"\n  Top-ROI agreement: {agree}/{len(common_subj)} ({agree/len(common_subj)*100:.0f}%)")

    return lines


def _compare_section_i(vlaai_dir, trf_dir, out_dir):
    lines = ["", "=" * 60, "SECTION I: FREQUENCY-BAND COMPARISON", "=" * 60, ""]

    v = _load_if_exists(vlaai_dir / "I_frequency_band" / "frequency_band.json")
    t = _load_if_exists(trf_dir / "I_frequency_band" / "frequency_band.json")

    if not v or not t:
        lines.append("  [SKIPPED] Missing results for one or both models.")
        return lines

    lines.append("Band Importance (|ΔP|):")
    lines.append(f"  {'Band':10s}  {'VLAAI':>12s}  {'TRF':>12s}  {'Ratio V/T':>10s}")
    lines.append("  " + "-" * 50)
    v_bands = {b["band"]: b for b in v["bands"]}
    t_bands = {b["band"]: b for b in t["bands"]}

    for band in ["delta", "theta", "alpha", "beta"]:
        vb = abs(v_bands.get(band, {}).get("mean_dp", 0))
        tb = abs(t_bands.get(band, {}).get("mean_dp", 0))
        ratio = vb / tb if tb > 1e-8 else float("inf")
        lines.append(f"  {band:10s}  {vb:12.5f}  {tb:12.5f}  {ratio:10.2f}")

    # Ranking comparison
    v_order = sorted(v_bands, key=lambda b: abs(v_bands[b]["mean_dp"]), reverse=True)
    t_order = sorted(t_bands, key=lambda b: abs(t_bands[b]["mean_dp"]), reverse=True)
    lines.append(f"\n  VLAAI band ranking: {' > '.join(v_order)}")
    lines.append(f"  TRF band ranking:   {' > '.join(t_order)}")
    lines.append(f"  Rankings match: {'Yes' if v_order == t_order else 'No'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bands = ["delta", "theta", "alpha", "beta"]
    band_colors = {"delta": "#1565c0", "theta": "#2e7d32", "alpha": "#f57f17", "beta": "#d32f2f"}

    ax = axes[0]
    x = np.arange(len(bands))
    v_vals = [abs(v_bands.get(b, {}).get("mean_dp", 0)) for b in bands]
    t_vals = [abs(t_bands.get(b, {}).get("mean_dp", 0)) for b in bands]
    ax.bar(x - 0.15, v_vals, 0.3, label="VLAAI", color="#1976d2", alpha=0.8)
    ax.bar(x + 0.15, t_vals, 0.3, label="TRF", color="#d32f2f", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("|ΔP|")
    ax.set_title("Frequency Band Importance")
    ax.legend()

    # Band x ROI comparison
    ax = axes[1]
    if "band_roi" in v and "band_roi" in t:
        v_br = {r["band"]: r for r in v["band_roi"]}
        t_br = {r["band"]: r for r in t["band_roi"]}
        diff_matrix = np.zeros((len(bands), len(ROIS)))
        for bi, b in enumerate(bands):
            for ri, roi in enumerate(ROIS):
                v_val = abs(v_br.get(b, {}).get(roi, 0))
                t_val = abs(t_br.get(b, {}).get(roi, 0))
                diff_matrix[bi, ri] = v_val - t_val

        im = ax.imshow(diff_matrix, aspect="auto", cmap="RdBu_r",
                        vmin=-np.abs(diff_matrix).max(), vmax=np.abs(diff_matrix).max())
        ax.set_xticks(range(len(ROIS)))
        ax.set_xticklabels(list(ROIS.keys()), rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels(bands)
        ax.set_title("Band×ROI Difference (VLAAI − TRF)")
        plt.colorbar(im, ax=ax, label="ΔΔP")
    else:
        ax.text(0.5, 0.5, "No band×ROI data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(out_dir / "comparison_frequency_band.png", dpi=150, bbox_inches="tight")
    plt.close()

    return lines


def _compare_section_j(vlaai_dir, trf_dir, out_dir):
    lines = ["", "=" * 60, "SECTION J: CORRECT vs INCORRECT COMPARISON", "=" * 60, ""]

    v = _load_if_exists(vlaai_dir / "J_correct_incorrect" / "correct_incorrect.json")
    t = _load_if_exists(trf_dir / "J_correct_incorrect" / "correct_incorrect.json")

    if not v or not t:
        lines.append("  [SKIPPED] Missing results for one or both models.")
        return lines

    lines.append("Decoding accuracy (correlation-based):")
    lines.append(f"  VLAAI: {v.get('accuracy', 0):.1%} ({v.get('n_correct', 0)}/{v.get('n_correct', 0) + v.get('n_incorrect', 0)})")
    lines.append(f"  TRF:   {t.get('accuracy', 0):.1%} ({t.get('n_correct', 0)}/{t.get('n_correct', 0) + t.get('n_incorrect', 0)})")

    lines.append(f"\n  Top-10 channel overlap (correct trials):")
    lines.append(f"    VLAAI: {v.get('top10_overlap', 'N/A')}/10")
    lines.append(f"    TRF:   {t.get('top10_overlap', 'N/A')}/10")

    if "roi_correct" in v and "roi_correct" in t:
        lines.append(f"\n  ROI importance for CORRECT trials:")
        lines.append(f"  {'ROI':20s}  {'VLAAI':>10s}  {'TRF':>10s}")
        lines.append("  " + "-" * 45)
        for roi in ROIS:
            vc = v["roi_correct"].get(roi, 0)
            tc = t["roi_correct"].get(roi, 0)
            lines.append(f"  {roi:20s}  {vc:10.5f}  {tc:10.5f}")

    return lines


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    set_seed(args.seed)
    dirs = make_output_dirs(Path(args.output_dir))
    vlaai_dir = Path(args.vlaai_results)

    sections = [s.upper() for s in args.sections]

    print("=" * 70)
    print("TRF vs VLAAI XAI COMPARISON")
    print(f"  Seed: {args.seed}, Bootstrap: {args.n_boot}, Max samples: {args.max_samples}")
    print(f"  Sections: {', '.join(sections)}")
    print(f"  TRF params: tmin={args.trf_tmin}, tmax={args.trf_tmax}, alpha={args.trf_alpha}")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    # ── Load dataset ──────────────────────────────────────────────────
    print("\nLoading dataset...")
    from aad_xai.data.vlaai_dataset import VLAAIDTUDataset

    ds = VLAAIDTUDataset(
        data_dir=args.data_dir,
        window_length=320, hop=64,
        subjects=args.subjects,
    )
    N_total = len(ds)
    N = min(args.max_samples, N_total)
    print(f"  {N_total} total windows, using {N}")

    # ── Split: train TRF on first portion, XAI on rest ────────────────
    n_train = int(N * args.train_ratio)
    n_test = N - n_train
    train_indices = list(range(n_train))
    test_indices = list(range(n_train, N))
    print(f"  Train: {n_train} windows, XAI test: {n_test} windows")

    # ── Train TRF ─────────────────────────────────────────────────────
    trf = train_trf(ds, train_indices, args.trf_tmin, args.trf_tmax, args.trf_alpha)

    # ── Build test tensors ────────────────────────────────────────────
    print("  Building test tensors...")
    eeg_all = torch.stack([ds[i][0] for i in test_indices])
    att_all = torch.stack([ds[i][1] for i in test_indices])
    unatt_all = torch.stack([ds[i][2] for i in test_indices])
    subject_ids = ds.subject_ids[test_indices[0]:test_indices[-1] + 1]

    # ── Wrap TRF in decision interface ────────────────────────────────
    from aad_xai.models.trf_decision import TRFDecisionWrapper
    decision = TRFDecisionWrapper(trf)

    # Quick baseline accuracy
    base_probs = get_attended_prob(decision, eeg_all, att_all, unatt_all)
    base_acc = (base_probs > 0.5).mean()
    print(f"  TRF baseline accuracy: {base_acc:.1%}")
    print(f"  TRF mean P(attended): {base_probs.mean():.4f}")

    # Save TRF training config
    save_json({
        "seed": args.seed, "n_boot": args.n_boot, "max_samples": args.max_samples,
        "n_train": n_train, "n_test": n_test,
        "trf_tmin": args.trf_tmin, "trf_tmax": args.trf_tmax, "trf_alpha": args.trf_alpha,
        "trf_train_corr": float(np.corrcoef(
            trf.predict(torch.stack([ds[i][0] for i in train_indices[:10]]).numpy()[0].T),
            ds[0][1].numpy()[:, 0][:trf.predict(ds[0][0].numpy().T).shape[0]]
        )[0, 1]),
        "baseline_accuracy": float(base_acc),
        "baseline_mean_p": float(base_probs.mean()),
        "sections": sections,
    }, dirs["root"] / "trf_run_config.json")

    # ── Run sections ─────────────────────────────────────────────────
    if "C" in sections:
        decision.set_envelopes(att_all, unatt_all)
        section_c_trf(decision, eeg_all, att_all, unatt_all,
                      args.n_boot, args.seed, dirs)

    if "D" in sections:
        decision.set_envelopes(att_all, unatt_all)
        section_d_trf(decision, eeg_all, att_all, unatt_all,
                      args.n_boot, args.seed, dirs)

    if "H" in sections:
        section_h_trf(decision, ds, args.n_boot, args.seed, dirs,
                      windows_per_subject=args.windows_per_subject)

    if "I" in sections:
        decision.set_envelopes(att_all, unatt_all)
        section_i_trf(decision, eeg_all, att_all, unatt_all,
                      args.n_boot, args.seed, dirs)

    if "J" in sections:
        section_j_trf(decision, trf, eeg_all, att_all, unatt_all,
                      args.n_boot, args.seed, dirs)

    # ── Comparison report ────────────────────────────────────────────
    trf_xai_dir = dirs["trf_xai"]
    comparison_dir = dirs["comparison"]
    if vlaai_dir.exists():
        generate_comparison(vlaai_dir, trf_xai_dir, comparison_dir)
    else:
        print(f"\n  WARNING: VLAAI results not found at {vlaai_dir}")
        print("  Run scripts/run_xai_comprehensive.py first, then re-run this script.")

    print("\n" + "=" * 70)
    print("TRF vs VLAAI COMPARISON COMPLETE")
    print(f"  All results: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
