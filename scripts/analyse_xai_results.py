"""Deep analysis of VLAAI XAI results — cross-method comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

OUT = Path("results_vlaai_xai")

# ── Load model + data ────────────────────────────────────────────────
from aad_xai.data.vlaai_dataset import VLAAIDTUDataset
from aad_xai.models import VLAAIPyTorch, AADDecisionEEGOnly

print("Loading model & data (all 18 subjects, 200 windows)...")
ds = VLAAIDTUDataset(
    "external/vlaai/evaluation_datasets/DTU",
    window_length=320, hop=64,
)
model = VLAAIPyTorch.from_h5("external/vlaai/pretrained_models/vlaai.h5")
model.eval()
decision = AADDecisionEEGOnly(model)

N = min(200, len(ds))
eeg_all = torch.stack([ds[i][0] for i in range(N)])
att_all = torch.stack([ds[i][1] for i in range(N)])
unatt_all = torch.stack([ds[i][2] for i in range(N)])

print(f"Dataset: {len(ds)} total windows, using {N}")

# ── 1. Channel-level Occlusion (all 64 channels, N windows) ─────────
print("\n=== 1. CHANNEL OCCLUSION (N=%d) ===" % N)
decision.set_envelopes(att_all, unatt_all)
with torch.no_grad():
    base_logits = decision(eeg_all)
    base_probs = torch.softmax(base_logits, dim=-1)  # (N, 2)
    base_att_prob = base_probs[:, 1].cpu().numpy()    # P(attended)

channel_drop = np.zeros(64)
for ch in range(64):
    eeg_masked = eeg_all.clone()
    eeg_masked[:, :, ch] = 0.0
    with torch.no_grad():
        m_logits = decision(eeg_masked)
        m_probs = torch.softmax(m_logits, dim=-1)[:, 1].cpu().numpy()
    channel_drop[ch] = (base_att_prob - m_probs).mean()

# Sort by importance
ch_order = np.argsort(np.abs(channel_drop))[::-1]
print("Top-15 channels by |ΔP(attended)|:")
for i, ch in enumerate(ch_order[:15]):
    d = channel_drop[ch]
    sign = "↓ hurts" if d > 0 else "↑ helps"
    print(f"  {i+1:2d}. Ch {ch:2d}: {d:+.5f}  ({sign} when removed)")

np.save(OUT / "channel_occlusion_full.npy", channel_drop)

# ── 2. Temporal Occlusion (slide 1-sec zero window) ─────────────────
print("\n=== 2. TEMPORAL OCCLUSION ===")
n_temp = min(50, N)
decision.set_envelopes(att_all[:n_temp], unatt_all[:n_temp])
with torch.no_grad():
    base_p = torch.softmax(decision(eeg_all[:n_temp]), dim=-1)[:, 1].cpu().numpy()

hop_t = 32  # 0.5 sec steps
win_t = 64  # 1 sec window
starts = list(range(0, 320 - win_t + 1, hop_t))
temporal_drop = np.zeros(len(starts))
for si, s in enumerate(starts):
    eeg_m = eeg_all[:n_temp].clone()
    eeg_m[:, s:s+win_t, :] = 0.0
    with torch.no_grad():
        m_p = torch.softmax(decision(eeg_m), dim=-1)[:, 1].cpu().numpy()
    temporal_drop[si] = (base_p - m_p).mean()

print("Temporal sensitivity (1-sec zero-mask, 0.5-sec hop):")
peak_t = np.argsort(np.abs(temporal_drop))[::-1]
for i in range(min(8, len(peak_t))):
    s = starts[peak_t[i]]
    sec_start = s / 64
    print(f"  t={sec_start:.1f}-{sec_start+1:.1f}s: ΔP={temporal_drop[peak_t[i]]:+.5f}")

np.save(OUT / "temporal_occlusion.npy", temporal_drop)

# ── 3. GradCAM per-layer attribution magnitude ──────────────────────
print("\n=== 3. GRADCAM PER-LAYER MAGNITUDES ===")
from aad_xai.xai import gradcam_all_blocks
n_gc = 10
decision.set_envelopes(att_all[:n_gc], unatt_all[:n_gc])
gc_all = gradcam_all_blocks(decision, eeg_all[:n_gc], target_class=1)

gc_means = {}
for name, attr in gc_all.items():
    arr = attr.detach().cpu().numpy()
    gc_means[name] = float(np.abs(arr).mean())

gc_sorted = sorted(gc_means.items(), key=lambda x: x[1], reverse=True)
print("Layer-wise GradCAM mean |activation|:")
for name, val in gc_sorted:
    bar = "█" * int(val * 200)
    print(f"  {name:50s}: {val:.5f} {bar}")

# ── 4. Per-block contribution analysis ───────────────────────────────
print("\n=== 4. PER-BLOCK CONTRIBUTION (skip-connection ablation) ===")
# VLAAI has 4 blocks each with shared extractor → per-block dense → shared output_context
# We can zero out each block_dense to measure per-block contribution
n_blk = min(50, N)
decision.set_envelopes(att_all[:n_blk], unatt_all[:n_blk])
with torch.no_grad():
    base_p = torch.softmax(decision(eeg_all[:n_blk]), dim=-1)[:, 1].cpu().numpy()

for bi in range(4):
    # Save original weights
    orig_w = model.block_denses[bi].weight.data.clone()
    orig_b = model.block_denses[bi].bias.data.clone()
    # Zero the block
    model.block_denses[bi].weight.data.zero_()
    model.block_denses[bi].bias.data.zero_()
    with torch.no_grad():
        abl_p = torch.softmax(decision(eeg_all[:n_blk]), dim=-1)[:, 1].cpu().numpy()
    drop = (base_p - abl_p).mean()
    # Restore
    model.block_denses[bi].weight.data = orig_w
    model.block_denses[bi].bias.data = orig_b
    print(f"  Block {bi}: ΔP(attended) = {drop:+.5f} when ablated")

# ── 5. EEG region grouping (neuroscience-standard 10-20 ROIs) ───────
print("\n=== 5. ROI-LEVEL ANALYSIS ===")
# Standard 64-ch EEG approximate groupings
ROIS = OrderedDict([
    ("Frontal (0-11)",       list(range(0, 12))),
    ("Fronto-Central (12-17)", list(range(12, 18))),
    ("Central (18-29)",      list(range(18, 30))),
    ("Temporal (30-41)",     list(range(30, 42))),
    ("Parietal (42-53)",     list(range(42, 54))),
    ("Occipital (54-63)",    list(range(54, 64))),
])

print("Channel occlusion aggregated by ROI:")
for roi_name, channels in ROIS.items():
    mean_drop = channel_drop[channels].mean()
    abs_mean = np.abs(channel_drop[channels]).mean()
    max_ch = channels[np.argmax(np.abs(channel_drop[channels]))]
    print(f"  {roi_name:30s}: mean ΔP={mean_drop:+.5f}, |ΔP|={abs_mean:.5f}, top Ch={max_ch}")

# ── 6. Correlation analysis: channel importance vs probe accuracy ────
print("\n=== 6. CROSS-METHOD CONSISTENCY ===")
with open(OUT / "probe_results.json") as f:
    probes = json.load(f)

# Attention probe: summarize extractor layers (conv only)
att_probes = probes["attention"]
print("Attention probe accuracy by processing stage:")
stages = {
    "extractor.block0": np.mean([att_probes[f"extractor.blocks.0.{s}_0"] for s in ["conv","ln","act","pad"]]),
    "extractor.block1": np.mean([att_probes[f"extractor.blocks.1.{s}_1"] for s in ["conv","ln","act","pad"]]),
    "extractor.block2": np.mean([att_probes[f"extractor.blocks.2.{s}_2"] for s in ["conv","ln","act","pad"]]),
    "extractor.block3": np.mean([att_probes[f"extractor.blocks.3.{s}_3"] for s in ["conv","ln","act","pad"]]),
    "extractor.block4": np.mean([att_probes[f"extractor.blocks.4.{s}_4"] for s in ["conv","ln","act","pad"]]),
    "block_denses (mean)": np.mean([att_probes[f"block_denses.{i}"] for i in range(4)]),
    "output_context": att_probes["output_context"],
    "final_dense": att_probes["final_dense"],
}
for stage, acc in stages.items():
    bar = "█" * int((acc - 0.4) * 100)
    print(f"  {stage:25s}: {acc:.3f} {bar}")

# Auditory probe: amplitude by stage
aud_probes = probes["auditory"]
print("\nAmplitude probe accuracy by processing stage:")
aud_stages = {
    "extractor.block0": np.mean([aud_probes[f"extractor.blocks.0.{s}_0"]["amplitude"] for s in ["conv","ln","act","pad"]]),
    "extractor.block1": np.mean([aud_probes[f"extractor.blocks.1.{s}_1"]["amplitude"] for s in ["conv","ln","act","pad"]]),
    "extractor.block2": np.mean([aud_probes[f"extractor.blocks.2.{s}_2"]["amplitude"] for s in ["conv","ln","act","pad"]]),
    "extractor.block3": np.mean([aud_probes[f"extractor.blocks.3.{s}_3"]["amplitude"] for s in ["conv","ln","act","pad"]]),
    "extractor.block4": np.mean([aud_probes[f"extractor.blocks.4.{s}_4"]["amplitude"] for s in ["conv","ln","act","pad"]]),
    "block_denses (mean)": np.mean([aud_probes[f"block_denses.{i}"]["amplitude"] for i in range(4)]),
    "output_context": aud_probes["output_context"]["amplitude"],
    "final_dense": aud_probes["final_dense"]["amplitude"],
}
for stage, acc in aud_stages.items():
    bar = "█" * int((acc - 0.15) * 100)
    print(f"  {stage:25s}: {acc:.3f} {bar}")

# ── 7. Sanity check interpretation ──────────────────────────────────
print("\n=== 7. SANITY CHECK INTERPRETATION ===")
with open(OUT / "sanity_check.json") as f:
    sanity = json.load(f)
orig = sanity["__original__"]
print("Cascading randomization (from output → input):")
print(f"  Original IG norm:    {orig:.4f}")
for layer in ["final_dense", "output_context", "block_denses", "extractor"]:
    v = sanity[layer]
    ratio = v / orig
    passed = "✓ PASS" if ratio < 0.5 or layer == "extractor" else "✗ FAIL (not enough change)"
    if layer == "extractor" and ratio > 1.5:
        passed = "⚠ AMPLIFIED (expected: attributions change substantially)"
    print(f"  After randomizing {layer:20s}: {v:.4f} ({ratio:.0%} of original) {passed}")

# ── 8. Generate summary plots ────────────────────────────────────────
print("\n=== 8. GENERATING ANALYSIS PLOTS ===")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("VLAAI XAI Cross-Method Analysis", fontsize=14, fontweight="bold")

# Plot 1: Channel occlusion heatmap
ax = axes[0, 0]
sorted_ch = np.argsort(channel_drop)[::-1]
colors = ["#d32f2f" if channel_drop[c] > 0 else "#1976d2" for c in sorted_ch[:20]]
ax.barh(range(20), [channel_drop[c] for c in sorted_ch[:20]], color=colors)
ax.set_yticks(range(20))
ax.set_yticklabels([f"Ch {c}" for c in sorted_ch[:20]], fontsize=8)
ax.set_xlabel("ΔP(attended) when removed")
ax.set_title("Channel Occlusion (top 20)")
ax.axvline(0, color="k", linewidth=0.5)
ax.invert_yaxis()

# Plot 2: ROI-level
ax = axes[0, 1]
roi_names = list(ROIS.keys())
roi_vals = [channel_drop[ROIS[r]].mean() for r in roi_names]
roi_abs = [np.abs(channel_drop[ROIS[r]]).mean() for r in roi_names]
short_names = [r.split("(")[0].strip() for r in roi_names]
bars = ax.bar(range(len(roi_names)), roi_abs, color="#6a1b9a", alpha=0.7)
ax.set_xticks(range(len(roi_names)))
ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Mean |ΔP(attended)|")
ax.set_title("ROI Importance (occlusion)")

# Plot 3: Temporal sensitivity
ax = axes[0, 2]
t_seconds = [s/64 for s in starts]
ax.fill_between(t_seconds, temporal_drop, alpha=0.3, color="#1976d2")
ax.plot(t_seconds, temporal_drop, color="#1976d2", linewidth=2)
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("ΔP(attended)")
ax.set_title("Temporal Sensitivity (1s mask)")

# Plot 4: Attention probes by stage
ax = axes[1, 0]
stage_names = list(stages.keys())
stage_vals = list(stages.values())
short_stage = [s.replace("extractor.", "ext.").replace("block_denses", "dense") for s in stage_names]
colors_p = ["#2e7d32" if v > 0.55 else "#f57f17" if v > 0.5 else "#d32f2f" for v in stage_vals]
ax.barh(range(len(stage_vals)), stage_vals, color=colors_p)
ax.set_yticks(range(len(stage_vals)))
ax.set_yticklabels(short_stage, fontsize=8)
ax.axvline(0.5, color="k", linestyle="--", linewidth=1, label="chance")
ax.set_xlabel("Accuracy")
ax.set_title("Attention Probes by Stage")
ax.set_xlim(0.35, 0.7)
ax.invert_yaxis()

# Plot 5: Auditory amplitude probes
ax = axes[1, 1]
aud_vals = list(aud_stages.values())
colors_a = ["#2e7d32" if v > 0.35 else "#f57f17" if v > 0.25 else "#d32f2f" for v in aud_vals]
ax.barh(range(len(aud_vals)), aud_vals, color=colors_a)
ax.set_yticks(range(len(aud_vals)))
ax.set_yticklabels(short_stage, fontsize=8)
ax.set_xlabel("Accuracy")
ax.set_title("Auditory Amplitude Probes")
ax.set_xlim(0.1, 0.5)
ax.invert_yaxis()

# Plot 6: GradCAM per-layer
ax = axes[1, 2]
gc_names = [n.replace("decoder.", "") for n, _ in gc_sorted]
gc_vals = [v for _, v in gc_sorted]
ax.barh(range(len(gc_vals)), gc_vals, color="#e65100", alpha=0.8)
ax.set_yticks(range(len(gc_vals)))
ax.set_yticklabels(gc_names, fontsize=7)
ax.set_xlabel("Mean |GradCAM|")
ax.set_title("GradCAM per Conv1d Layer")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUT / "xai_cross_analysis.png", dpi=150, bbox_inches="tight")
print(f"  Saved {OUT / 'xai_cross_analysis.png'}")

# ── Channel topography ──────────────────────────────────────────────
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 3))
ax2.bar(range(64), channel_drop, color=["#d32f2f" if d > 0 else "#1976d2" for d in channel_drop], alpha=0.8)
ax2.axhline(0, color="k", linewidth=0.5)
ax2.set_xlabel("EEG Channel Index")
ax2.set_ylabel("ΔP(attended)")
ax2.set_title("Full Channel Occlusion Map (red=important for tracking, blue=suppressive)")
plt.tight_layout()
plt.savefig(OUT / "channel_topography.png", dpi=150, bbox_inches="tight")
print(f"  Saved {OUT / 'channel_topography.png'}")

print("\n✓ Core analysis complete.")

# ══════════════════════════════════════════════════════════════════════
# EXTENDED ANALYSES
# ══════════════════════════════════════════════════════════════════════
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# ── Helpers ──────────────────────────────────────────────────────────

def get_attended_probability(decision_model, eeg, env_att, env_unatt):
    """Compute P(attended) for a batch of windows."""
    decision_model.set_envelopes(env_att, env_unatt)
    with torch.no_grad():
        logits = decision_model(eeg)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


def compute_channel_occlusion(decision_model, eeg, env_att, env_unatt, base_prob=None):
    """Per-channel occlusion importance → array (64,)."""
    if base_prob is None:
        base_prob = get_attended_probability(decision_model, eeg, env_att, env_unatt)
    drops = np.zeros(64)
    decision_model.set_envelopes(env_att, env_unatt)
    for ch in range(64):
        eeg_m = eeg.clone()
        eeg_m[:, :, ch] = 0.0
        with torch.no_grad():
            m_p = torch.softmax(decision_model(eeg_m), dim=-1)[:, 1].cpu().numpy()
        drops[ch] = (base_prob - m_p).mean()
    return drops


def compute_temporal_occlusion(decision_model, eeg, env_att, env_unatt,
                                base_prob=None, win_t=64, hop_t=32):
    """Sliding-window temporal occlusion → (starts, drops)."""
    if base_prob is None:
        base_prob = get_attended_probability(decision_model, eeg, env_att, env_unatt)
    T = eeg.shape[1]
    starts = list(range(0, T - win_t + 1, hop_t))
    drops = np.zeros(len(starts))
    decision_model.set_envelopes(env_att, env_unatt)
    for si, s in enumerate(starts):
        eeg_m = eeg.clone()
        eeg_m[:, s:s + win_t, :] = 0.0
        with torch.no_grad():
            m_p = torch.softmax(decision_model(eeg_m), dim=-1)[:, 1].cpu().numpy()
        drops[si] = (base_prob - m_p).mean()
    return starts, drops


def compute_roi_importance(channel_drops, rois):
    """Aggregate channel drops per ROI → dict of mean |ΔP|."""
    return {name: float(np.abs(channel_drops[chs]).mean()) for name, chs in rois.items()}


# ── 9. Subject-wise Interpretability Consistency ─────────────────────
print("\n" + "=" * 70)
print("=== 9. SUBJECT-WISE INTERPRETABILITY CONSISTENCY ===")
print("=" * 70)

try:
    all_subject_ids = ds.subject_ids
    unique_subjects = sorted(set(all_subject_ids))
    print(f"  Found {len(unique_subjects)} subjects: {unique_subjects}")

    subj_channel_rows = []
    subj_temporal_rows = []
    subj_roi_rows = []

    for subj in unique_subjects:
        mask = np.array([sid == subj for sid in all_subject_ids])
        idxs = np.where(mask)[0]
        n_s = min(50, len(idxs))  # cap per subject
        idxs = idxs[:n_s]

        eeg_s = torch.stack([ds[i][0] for i in idxs])
        att_s = torch.stack([ds[i][1] for i in idxs])
        unatt_s = torch.stack([ds[i][2] for i in idxs])

        base_p = get_attended_probability(decision, eeg_s, att_s, unatt_s)

        # Channel occlusion
        ch_drops = compute_channel_occlusion(decision, eeg_s, att_s, unatt_s, base_p)
        row = {"subject": subj}
        for ch in range(64):
            row[f"ch_{ch}"] = ch_drops[ch]
        subj_channel_rows.append(row)

        # Temporal occlusion
        t_starts, t_drops = compute_temporal_occlusion(decision, eeg_s, att_s, unatt_s, base_p)
        t_row = {"subject": subj}
        for si, s in enumerate(t_starts):
            t_row[f"t_{s/64:.1f}s"] = t_drops[si]
        subj_temporal_rows.append(t_row)

        # ROI importance
        roi_imp = compute_roi_importance(ch_drops, ROIS)
        roi_row = {"subject": subj}
        roi_row.update(roi_imp)
        subj_roi_rows.append(roi_row)

        print(f"  {subj}: {n_s} windows, mean P(att)={base_p.mean():.3f}, "
              f"top ROI={max(roi_imp, key=roi_imp.get)}")

    # Save CSVs
    df_ch = pd.DataFrame(subj_channel_rows)
    df_ch.to_csv(OUT / "subject_channel_occlusion.csv", index=False)
    print(f"  Saved {OUT / 'subject_channel_occlusion.csv'}")

    df_temp = pd.DataFrame(subj_temporal_rows)
    df_temp.to_csv(OUT / "subject_temporal_occlusion.csv", index=False)
    print(f"  Saved {OUT / 'subject_temporal_occlusion.csv'}")

    df_roi = pd.DataFrame(subj_roi_rows)
    df_roi.to_csv(OUT / "subject_roi_importance.csv", index=False)
    print(f"  Saved {OUT / 'subject_roi_importance.csv'}")

    # Cross-subject consistency: pairwise correlation of channel importance
    ch_matrix = df_ch.drop(columns=["subject"]).values  # (n_subj, 64)
    n_subj = ch_matrix.shape[0]
    corr_matrix = np.corrcoef(ch_matrix)
    triu_idx = np.triu_indices(n_subj, k=1)
    pairwise_corrs = corr_matrix[triu_idx]
    print(f"\n  Cross-subject channel-importance consistency:")
    print(f"    Mean pairwise r = {pairwise_corrs.mean():.3f}")
    print(f"    Std  pairwise r = {pairwise_corrs.std():.3f}")
    print(f"    Min / Max r     = {pairwise_corrs.min():.3f} / {pairwise_corrs.max():.3f}")

    # Plot: Subject-wise ROI importance heatmap
    fig_s, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(14, 6))

    roi_matrix = df_roi.drop(columns=["subject"]).values  # (n_subj, n_rois)
    roi_short = [r.split("(")[0].strip() for r in ROIS.keys()]
    im = ax_s1.imshow(roi_matrix, aspect="auto", cmap="viridis")
    ax_s1.set_xticks(range(len(roi_short)))
    ax_s1.set_xticklabels(roi_short, rotation=30, ha="right", fontsize=8)
    ax_s1.set_yticks(range(n_subj))
    ax_s1.set_yticklabels(unique_subjects, fontsize=7)
    ax_s1.set_title("Subject × ROI Importance (|ΔP|)")
    ax_s1.set_xlabel("ROI")
    ax_s1.set_ylabel("Subject")
    plt.colorbar(im, ax=ax_s1)

    # Error bar plot: mean ± std across subjects per ROI
    roi_mean = roi_matrix.mean(axis=0)
    roi_std = roi_matrix.std(axis=0)
    ax_s2.bar(range(len(roi_short)), roi_mean, yerr=roi_std,
              color="#6a1b9a", alpha=0.7, capsize=4)
    ax_s2.set_xticks(range(len(roi_short)))
    ax_s2.set_xticklabels(roi_short, rotation=30, ha="right", fontsize=8)
    ax_s2.set_ylabel("Mean |ΔP(attended)| ± std")
    ax_s2.set_title("ROI Importance: Cross-Subject Mean ± Std")

    plt.tight_layout()
    plt.savefig(OUT / "subject_wise_analysis.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {OUT / 'subject_wise_analysis.png'}")

except Exception as e:
    print(f"  ⚠ Subject-wise analysis failed: {e}")

# ── 10. Frequency-Band Occlusion Analysis ────────────────────────────
print("\n" + "=" * 70)
print("=== 10. FREQUENCY-BAND OCCLUSION ANALYSIS ===")
print("=" * 70)

try:
    FS = 64  # sampling rate
    BANDS = OrderedDict([
        ("delta (0.5-4 Hz)",  (0.5, 4.0)),
        ("theta (4-8 Hz)",    (4.0, 8.0)),
        ("alpha (8-13 Hz)",   (8.0, 13.0)),
        ("beta (13-30 Hz)",   (13.0, 30.0)),
    ])

    n_fb = min(100, N)
    eeg_fb = eeg_all[:n_fb]
    att_fb = att_all[:n_fb]
    unatt_fb = unatt_all[:n_fb]
    base_p_fb = get_attended_probability(decision, eeg_fb, att_fb, unatt_fb)

    band_rows = []
    band_roi_rows = []

    for band_name, (lo, hi) in BANDS.items():
        # Design bandpass filter
        nyq = FS / 2.0
        lo_n = max(lo / nyq, 0.01)
        hi_n = min(hi / nyq, 0.99)
        sos = butter(4, [lo_n, hi_n], btype="bandpass", output="sos")

        # Extract band content from EEG, then subtract it
        eeg_np = eeg_fb.numpy()  # (n_fb, 320, 64)
        band_content = np.zeros_like(eeg_np)
        for ch in range(64):
            for w in range(n_fb):
                band_content[w, :, ch] = sosfiltfilt(sos, eeg_np[w, :, ch])

        eeg_no_band = torch.from_numpy((eeg_np - band_content).astype(np.float32))
        p_no_band = get_attended_probability(decision, eeg_no_band, att_fb, unatt_fb)
        band_drop = (base_p_fb - p_no_band).mean()

        band_rows.append({"band": band_name, "mean_delta_p": band_drop,
                          "abs_delta_p": abs(band_drop)})

        # Per-ROI × band analysis
        roi_row = {"band": band_name}
        for roi_name, channels in ROIS.items():
            eeg_roi_band = eeg_fb.clone().numpy()
            band_roi_content = np.zeros_like(eeg_roi_band)
            for ch in channels:
                for w in range(n_fb):
                    band_roi_content[w, :, ch] = sosfiltfilt(sos, eeg_roi_band[w, :, ch])
            eeg_no_roi_band = torch.from_numpy(
                (eeg_roi_band - band_roi_content).astype(np.float32))
            p_roi_band = get_attended_probability(decision, eeg_no_roi_band, att_fb, unatt_fb)
            roi_row[roi_name] = float((base_p_fb - p_roi_band).mean())

        band_roi_rows.append(roi_row)

        print(f"  {band_name:25s}: ΔP = {band_drop:+.5f}")

    # Save CSVs
    df_band = pd.DataFrame(band_rows)
    df_band.to_csv(OUT / "frequency_band_occlusion.csv", index=False)
    print(f"  Saved {OUT / 'frequency_band_occlusion.csv'}")

    df_band_roi = pd.DataFrame(band_roi_rows)
    df_band_roi.to_csv(OUT / "frequency_band_roi_occlusion.csv", index=False)
    print(f"  Saved {OUT / 'frequency_band_roi_occlusion.csv'}")

    # Plot: Band importance + Band × ROI heatmap
    fig_fb, (ax_fb1, ax_fb2) = plt.subplots(1, 2, figsize=(14, 5))

    band_names_short = [b.split("(")[0].strip() for b in BANDS.keys()]
    band_vals = [r["abs_delta_p"] for r in band_rows]
    colors_fb = ["#1565c0", "#2e7d32", "#f57f17", "#d32f2f"]
    ax_fb1.bar(range(len(band_names_short)), band_vals, color=colors_fb, alpha=0.8)
    ax_fb1.set_xticks(range(len(band_names_short)))
    ax_fb1.set_xticklabels(band_names_short, fontsize=9)
    ax_fb1.set_ylabel("|ΔP(attended)|")
    ax_fb1.set_title("Frequency Band Importance (whole-brain)")

    # Band × ROI heatmap
    roi_short_fb = [r.split("(")[0].strip() for r in ROIS.keys()]
    br_matrix = df_band_roi.drop(columns=["band"]).values  # (n_bands, n_rois)
    im_fb = ax_fb2.imshow(np.abs(br_matrix), aspect="auto", cmap="YlOrRd")
    ax_fb2.set_xticks(range(len(roi_short_fb)))
    ax_fb2.set_xticklabels(roi_short_fb, rotation=30, ha="right", fontsize=8)
    ax_fb2.set_yticks(range(len(band_names_short)))
    ax_fb2.set_yticklabels(band_names_short, fontsize=9)
    ax_fb2.set_title("Band × ROI Importance (|ΔP|)")
    plt.colorbar(im_fb, ax=ax_fb2)

    plt.tight_layout()
    plt.savefig(OUT / "frequency_band_analysis.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {OUT / 'frequency_band_analysis.png'}")

except Exception as e:
    print(f"  ⚠ Frequency-band analysis failed: {e}")

# ── 11. Correct vs Incorrect Prediction Explanations ─────────────────
print("\n" + "=" * 70)
print("=== 11. CORRECT vs INCORRECT PREDICTION ANALYSIS ===")
print("=" * 70)

try:
    n_ci = min(200, len(ds))
    eeg_ci = eeg_all[:n_ci]
    att_ci = att_all[:n_ci]
    unatt_ci = unatt_all[:n_ci]

    # Compute P(attended) for all windows
    p_att = get_attended_probability(decision, eeg_ci, att_ci, unatt_ci)

    # "Correct" = P(attended) > 0.5 (model confidently tracks attended speaker)
    # "Incorrect" = P(attended) <= 0.5 (model fails to identify attended)
    correct_mask = p_att > 0.5
    incorrect_mask = ~correct_mask

    n_correct = correct_mask.sum()
    n_incorrect = incorrect_mask.sum()
    print(f"  Correct: {n_correct} windows (P(att) > 0.5)")
    print(f"  Incorrect: {n_incorrect} windows (P(att) <= 0.5)")
    print(f"  Overall accuracy: {n_correct / n_ci:.1%}")

    if n_correct >= 5 and n_incorrect >= 5:
        # Channel occlusion for correct predictions
        c_idx = np.where(correct_mask)[0]
        c_n = min(100, len(c_idx))
        c_idx = c_idx[:c_n]
        eeg_c = eeg_ci[c_idx]
        att_c = att_ci[c_idx]
        unatt_c = unatt_ci[c_idx]
        base_p_c = get_attended_probability(decision, eeg_c, att_c, unatt_c)
        ch_drops_correct = compute_channel_occlusion(decision, eeg_c, att_c, unatt_c, base_p_c)

        # Channel occlusion for incorrect predictions
        i_idx = np.where(incorrect_mask)[0]
        i_n = min(100, len(i_idx))
        i_idx = i_idx[:i_n]
        eeg_i = eeg_ci[i_idx]
        att_i = att_ci[i_idx]
        unatt_i = unatt_ci[i_idx]
        base_p_i = get_attended_probability(decision, eeg_i, att_i, unatt_i)
        ch_drops_incorrect = compute_channel_occlusion(decision, eeg_i, att_i, unatt_i, base_p_i)

        # Temporal occlusion for both
        t_starts_c, t_drops_correct = compute_temporal_occlusion(
            decision, eeg_c, att_c, unatt_c, base_p_c)
        t_starts_i, t_drops_incorrect = compute_temporal_occlusion(
            decision, eeg_i, att_i, unatt_i, base_p_i)

        # Save CSVs
        ci_ch_rows = []
        for ch in range(64):
            ci_ch_rows.append({
                "channel": ch,
                "correct_delta_p": ch_drops_correct[ch],
                "incorrect_delta_p": ch_drops_incorrect[ch],
                "difference": ch_drops_correct[ch] - ch_drops_incorrect[ch],
            })
        df_ci_ch = pd.DataFrame(ci_ch_rows)
        df_ci_ch.to_csv(OUT / "correct_vs_incorrect_channel.csv", index=False)
        print(f"  Saved {OUT / 'correct_vs_incorrect_channel.csv'}")

        ci_t_rows = []
        for si, s in enumerate(t_starts_c):
            ci_t_rows.append({
                "time_s": s / 64,
                "correct_delta_p": t_drops_correct[si],
                "incorrect_delta_p": t_drops_incorrect[si],
            })
        df_ci_t = pd.DataFrame(ci_t_rows)
        df_ci_t.to_csv(OUT / "correct_vs_incorrect_temporal.csv", index=False)
        print(f"  Saved {OUT / 'correct_vs_incorrect_temporal.csv'}")

        # ROI comparison
        roi_correct = compute_roi_importance(ch_drops_correct, ROIS)
        roi_incorrect = compute_roi_importance(ch_drops_incorrect, ROIS)
        ci_roi_rows = []
        for roi_name in ROIS:
            ci_roi_rows.append({
                "ROI": roi_name,
                "correct_abs_delta_p": roi_correct[roi_name],
                "incorrect_abs_delta_p": roi_incorrect[roi_name],
                "difference": roi_correct[roi_name] - roi_incorrect[roi_name],
            })
        df_ci_roi = pd.DataFrame(ci_roi_rows)
        df_ci_roi.to_csv(OUT / "correct_vs_incorrect_roi.csv", index=False)
        print(f"  Saved {OUT / 'correct_vs_incorrect_roi.csv'}")

        # Print summary
        print("\n  Channel importance comparison (top-10 by |difference|):")
        diff_order = np.argsort(np.abs(ch_drops_correct - ch_drops_incorrect))[::-1]
        for i in range(min(10, len(diff_order))):
            ch = diff_order[i]
            print(f"    Ch {ch:2d}: correct={ch_drops_correct[ch]:+.5f}, "
                  f"incorrect={ch_drops_incorrect[ch]:+.5f}, "
                  f"diff={ch_drops_correct[ch] - ch_drops_incorrect[ch]:+.5f}")

        print("\n  ROI importance comparison:")
        for roi_name in ROIS:
            print(f"    {roi_name:30s}: correct={roi_correct[roi_name]:.5f}, "
                  f"incorrect={roi_incorrect[roi_name]:.5f}")

        # Plot: Correct vs Incorrect comparison
        fig_ci, axes_ci = plt.subplots(1, 3, figsize=(18, 5))

        # Channel comparison
        ax = axes_ci[0]
        top_ch = diff_order[:20]
        x_pos = np.arange(20)
        ax.barh(x_pos - 0.15, [ch_drops_correct[c] for c in top_ch],
                height=0.3, color="#2e7d32", alpha=0.8, label="Correct")
        ax.barh(x_pos + 0.15, [ch_drops_incorrect[c] for c in top_ch],
                height=0.3, color="#d32f2f", alpha=0.8, label="Incorrect")
        ax.set_yticks(x_pos)
        ax.set_yticklabels([f"Ch {c}" for c in top_ch], fontsize=7)
        ax.set_xlabel("ΔP(attended)")
        ax.set_title("Channel Importance: Correct vs Incorrect")
        ax.legend(fontsize=8)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.invert_yaxis()

        # Temporal comparison
        ax = axes_ci[1]
        t_sec = [s / 64 for s in t_starts_c]
        ax.plot(t_sec, t_drops_correct, color="#2e7d32", linewidth=2, label="Correct")
        ax.plot(t_sec, t_drops_incorrect, color="#d32f2f", linewidth=2, label="Incorrect")
        ax.fill_between(t_sec, t_drops_correct, alpha=0.15, color="#2e7d32")
        ax.fill_between(t_sec, t_drops_incorrect, alpha=0.15, color="#d32f2f")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ΔP(attended)")
        ax.set_title("Temporal Sensitivity: Correct vs Incorrect")
        ax.legend(fontsize=8)

        # ROI comparison
        ax = axes_ci[2]
        roi_short_ci = [r.split("(")[0].strip() for r in ROIS.keys()]
        x_roi = np.arange(len(roi_short_ci))
        vals_c = [roi_correct[r] for r in ROIS]
        vals_i = [roi_incorrect[r] for r in ROIS]
        ax.bar(x_roi - 0.15, vals_c, width=0.3, color="#2e7d32", alpha=0.8, label="Correct")
        ax.bar(x_roi + 0.15, vals_i, width=0.3, color="#d32f2f", alpha=0.8, label="Incorrect")
        ax.set_xticks(x_roi)
        ax.set_xticklabels(roi_short_ci, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Mean |ΔP(attended)|")
        ax.set_title("ROI Importance: Correct vs Incorrect")
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(OUT / "correct_vs_incorrect_analysis.png", dpi=150, bbox_inches="tight")
        print(f"  Saved {OUT / 'correct_vs_incorrect_analysis.png'}")
    else:
        print(f"  ⚠ Skipping: need ≥5 windows in each group "
              f"(correct={n_correct}, incorrect={n_incorrect})")

except Exception as e:
    print(f"  ⚠ Correct vs incorrect analysis failed: {e}")

# ── 12. Extended Summary ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== 12. GENERATING EXTENDED XAI SUMMARY ===")
print("=" * 70)

summary_lines = [
    "EXTENDED XAI ANALYSIS SUMMARY",
    "=" * 50,
    f"Dataset: {len(ds)} windows total, {N} used for core analysis",
    "",
]

# Subject-wise summary
try:
    summary_lines.append("--- SUBJECT-WISE CONSISTENCY ---")
    summary_lines.append(f"Subjects analysed: {len(unique_subjects)}")
    summary_lines.append(f"Cross-subject channel r: {pairwise_corrs.mean():.3f} ± {pairwise_corrs.std():.3f}")
    most_consistent_roi = roi_short[np.argmin(roi_matrix.std(axis=0))]
    most_variable_roi = roi_short[np.argmax(roi_matrix.std(axis=0))]
    summary_lines.append(f"Most consistent ROI across subjects: {most_consistent_roi}")
    summary_lines.append(f"Most variable ROI across subjects: {most_variable_roi}")
    summary_lines.append("")
except Exception:
    summary_lines.append("Subject-wise analysis: not available")
    summary_lines.append("")

# Frequency-band summary
try:
    summary_lines.append("--- FREQUENCY-BAND IMPORTANCE ---")
    for r in band_rows:
        summary_lines.append(f"  {r['band']:25s}: |ΔP| = {r['abs_delta_p']:.5f}")
    most_important_band = max(band_rows, key=lambda x: x["abs_delta_p"])["band"]
    summary_lines.append(f"Most important band: {most_important_band}")
    summary_lines.append("")
except Exception:
    summary_lines.append("Frequency-band analysis: not available")
    summary_lines.append("")

# Correct vs incorrect summary
try:
    summary_lines.append("--- CORRECT vs INCORRECT ---")
    summary_lines.append(f"Correct: {n_correct} windows, Incorrect: {n_incorrect} windows")
    summary_lines.append(f"Overall accuracy: {n_correct / n_ci:.1%}")
    top_diff_ch = diff_order[0]
    summary_lines.append(f"Largest channel importance difference: Ch {top_diff_ch} "
                         f"(diff={ch_drops_correct[top_diff_ch] - ch_drops_incorrect[top_diff_ch]:+.5f})")
    top_roi_diff = max(ROIS.keys(), key=lambda r: abs(roi_correct[r] - roi_incorrect[r]))
    summary_lines.append(f"Largest ROI importance difference: {top_roi_diff}")
    summary_lines.append("")
except Exception:
    summary_lines.append("Correct vs incorrect analysis: not available")
    summary_lines.append("")

summary_text = "\n".join(summary_lines)
(OUT / "extended_xai_summary.txt").write_text(summary_text, encoding="utf-8")
print(f"  Saved {OUT / 'extended_xai_summary.txt'}")
print(summary_text)

print("\n✓ Extended analysis complete.")
