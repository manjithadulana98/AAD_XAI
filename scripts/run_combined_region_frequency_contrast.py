"""Combined-group contrast: important regions/frequency vs. everything else.

Additive, standalone script -- does not modify any existing pipeline file.
Answers a different question than the existing per-channel/per-ROI/per-band
tables: instead of averaging independently-measured single-channel effects,
this actually occludes (or band-removes) an entire group of channels
*simultaneously in one forward pass*, so it can surface synergy/redundancy
effects a simple average of individual effects can't.

Two contrasts, each run for both occlusion (zero) and, where applicable,
permutation (shuffle) manipulation styles:
  1. Core ROI group (Fronto-Central + Central + Temporal + Centro-Parietal,
     the top-4 ROIs from this project's cross-model analysis) vs. every
     other ROI, as one combined 64-channel-spanning group each.
  2. Theta band (the one frequency finding that replicated across both
     models without the delta short-window caveat) vs. alpha band (the
     least important band in both models), removed from all 64 channels
     at once -- not per-ROI as the existing frequency section does.

For each condition, per-subject mean delta-P is computed, then tested
with the same subject-level Wilcoxon + bootstrap-CI + Cohen's d approach
used throughout this project's Phase 5 statistical layer. A direct paired
contrast (important-group delta-P minus other-group delta-P, per subject)
is also tested against zero.
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FS = 64
CORE_ROIS = ["Fronto-Central", "Central", "Temporal", "Centro-Parietal"]
IMPORTANT_BAND = "theta"
OTHER_BAND = "alpha"
BANDS = OrderedDict([("delta", (0.5, 4.0)), ("theta", (4.0, 8.0)),
                     ("alpha", (8.0, 13.0)), ("beta", (13.0, 30.0))])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(ROOT / "data" / "vlaai_dtu_npz"))
    p.add_argument("--h5-path", type=str, default=str(ROOT / "models" / "vlaai.h5"))
    p.add_argument("--montage-file", type=str, default=str(ROOT / "config" / "dtu_channel_montage.csv"))
    p.add_argument("--output-dir", type=str, default=str(ROOT / "xai_results_combined_contrast"))
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def load_montage(montage_path):
    import csv
    ch_to_roi, rois = {}, OrderedDict()
    with open(montage_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["channel_index"])
            ch_to_roi[idx] = row["roi"]
    for idx, roi in sorted(ch_to_roi.items()):
        rois.setdefault(roi, []).append(idx)
    return ch_to_roi, rois


def get_attended_prob(decision, eeg, att, unatt):
    decision.set_envelopes(att, unatt)
    with torch.no_grad():
        logits = decision(eeg)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


def bootstrap_ci(values, n_boot, seed):
    rng = np.random.RandomState(seed)
    means = np.array([values[rng.randint(0, len(values), len(values))].mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(x):
    x = np.asarray(x)
    if np.allclose(x, 0):
        return 1.0
    _, p = wilcoxon(x)
    return float(p)


def cohens_d(x):
    x = np.asarray(x)
    std = np.std(x, ddof=1)
    return float(np.mean(x) / std) if std > 1e-12 else 0.0


def band_content(eeg_np, band_name):
    """Vectorized equivalent of the existing per-window/per-channel loop
    (same 4th-order Butterworth design, same reflect padding via sosfiltfilt's
    own padtype) -- filtering all windows and channels along the time axis
    in one call instead of N*64 individual Python-level calls, which doesn't
    scale to a full 8100-window run."""
    lo, hi = BANDS[band_name]
    nyq = FS / 2.0
    sos = butter(4, [max(lo / nyq, 0.01), min(hi / nyq, 0.99)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, eeg_np, axis=1, padtype="even", padlen=64)


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_top = np.random.RandomState(args.seed)

    print("=" * 70)
    print("COMBINED-GROUP CONTRAST: important regions/frequency vs. other")
    print("=" * 70)

    ch_to_roi, rois = load_montage(args.montage_file)
    core_chs = sorted({ch for roi in CORE_ROIS for ch in rois.get(roi, [])})
    other_chs = sorted(set(range(64)) - set(core_chs))
    print(f"Core ROI group ({'+'.join(CORE_ROIS)}): {len(core_chs)} channels")
    print(f"Other ROI group (remaining {len(rois) - len(CORE_ROIS)} ROIs): {len(other_chs)} channels")

    from aad_xai.data.vlaai_dataset import VLAAIDTUDataset
    ds = VLAAIDTUDataset(data_dir=args.data_dir, window_length=320, hop=64)
    total = len(ds)
    N = total if args.max_samples == -1 else min(args.max_samples, total)
    idxs = list(range(N))
    subject_ids = np.asarray([ds.subject_ids[i] for i in idxs])
    unique_subjects = sorted(set(subject_ids.tolist()))
    print(f"Dataset: {N}/{total} windows, {len(unique_subjects)} subjects")

    eeg = torch.stack([ds[i][0] for i in idxs]).to(device)
    att = torch.stack([ds[i][1] for i in idxs]).to(device)
    unatt = torch.stack([ds[i][2] for i in idxs]).to(device)

    from aad_xai.models import VLAAIPyTorch, AADDecisionEEGOnly
    model = VLAAIPyTorch.from_h5(args.h5_path)
    model.eval().to(device)
    decision = AADDecisionEEGOnly(model)
    decision.eval().to(device)

    base_probs = get_attended_prob(decision, eeg, att, unatt)
    print(f"Baseline mean P(attended): {base_probs.mean():.4f}")

    conditions = {}

    # --- Region groups: occlusion (zero) and permutation (independent shuffle per channel) ---
    for name, chs in [("core_roi", core_chs), ("other_roi", other_chs)]:
        eeg_occ = eeg.clone()
        eeg_occ[:, :, chs] = 0.0
        probs_occ = get_attended_prob(decision, eeg_occ, att, unatt)
        conditions[f"{name}_occ"] = base_probs - probs_occ
        print(f"  {name} occlusion done ({len(chs)} ch simultaneously)")

        eeg_perm = eeg.clone()
        for ch in chs:
            perm = torch.from_numpy(rng_top.permutation(N)).long().to(device)
            eeg_perm[:, :, ch] = eeg.index_select(0, perm)[:, :, ch]
        probs_perm = get_attended_prob(decision, eeg_perm, att, unatt)
        conditions[f"{name}_perm"] = base_probs - probs_perm
        print(f"  {name} permutation done ({len(chs)} ch simultaneously)")

    # --- Whole-brain band removal: theta (important) vs alpha (other) ---
    eeg_np = eeg.detach().cpu().numpy()
    for band in (IMPORTANT_BAND, OTHER_BAND):
        bc = band_content(eeg_np, band)
        eeg_band = eeg.clone()
        eeg_band -= torch.from_numpy(bc.astype(np.float32)).to(device)
        probs_band = get_attended_prob(decision, eeg_band, att, unatt)
        conditions[f"band_{band}"] = base_probs - probs_band
        print(f"  whole-brain {band}-band removal done (64 ch simultaneously)")

    decision.set_envelopes(att, unatt)  # restore

    # --- Per-subject aggregation + subject-level stats ---
    def subj_means(arr):
        return np.array([arr[subject_ids == s].mean() for s in unique_subjects])

    results = {}
    for key, arr in conditions.items():
        sv = subj_means(arr)
        mean, lo, hi = bootstrap_ci(sv, args.n_boot, args.seed)
        results[key] = {
            "subject_values": sv.tolist(), "mean_dp": mean, "ci_lo": lo, "ci_hi": hi,
            "cohens_d": cohens_d(sv), "wilcox_p": wilcoxon_p(sv),
        }
        print(f"  {key:14s} mean dP={mean:+.6f} [{lo:+.6f},{hi:+.6f}] d={results[key]['cohens_d']:+.3f} p={results[key]['wilcox_p']:.4f}")

    # --- Paired contrasts: important vs other, per subject ---
    contrasts = {}
    for metric in ("occ", "perm"):
        a = np.array(results[f"core_roi_{metric}"]["subject_values"])
        b = np.array(results[f"other_roi_{metric}"]["subject_values"])
        diff = a - b
        mean, lo, hi = bootstrap_ci(diff, args.n_boot, args.seed)
        contrasts[f"core_vs_other_{metric}"] = {
            "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
            "cohens_d": cohens_d(diff), "wilcox_p": wilcoxon_p(diff),
        }
    a = np.array(results[f"band_{IMPORTANT_BAND}"]["subject_values"])
    b = np.array(results[f"band_{OTHER_BAND}"]["subject_values"])
    diff = a - b
    mean, lo, hi = bootstrap_ci(diff, args.n_boot, args.seed)
    contrasts[f"{IMPORTANT_BAND}_vs_{OTHER_BAND}"] = {
        "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
        "cohens_d": cohens_d(diff), "wilcox_p": wilcoxon_p(diff),
    }

    print("\nPaired contrasts (important minus other, per subject):")
    for key, c in contrasts.items():
        print(f"  {key:22s} diff={c['mean_diff']:+.6f} [{c['ci_lo']:+.6f},{c['ci_hi']:+.6f}] d={c['cohens_d']:+.3f} p={c['wilcox_p']:.4f}")

    out = {
        "n_windows": N, "n_subjects": len(unique_subjects),
        "core_roi_channels": core_chs, "other_roi_channels": other_chs,
        "important_band": IMPORTANT_BAND, "other_band": OTHER_BAND,
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "subject_values"} for k, v in results.items()},
        "contrasts": contrasts,
    }
    with open(out_dir / "combined_contrast_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_dir / 'combined_contrast_results.json'}")


if __name__ == "__main__":
    main()
