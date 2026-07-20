# AAD XAI Research — Project Summary

**Author:** Manjitha K  
**Date:** June 2026  
**Repository:** https://github.com/manjithadulana98/AAD_XAI.git

---

## Overview

This project applies explainability methods (XAI) to EEG-based Auditory Attention Decoding (AAD) using the VLAAI deep learning model on the DTU EEG dataset. The goal is to identify which EEG channels and frequency bands drive the model's attention decoding decisions, and to assess how reliable those explanations are across subjects.

---

## 1. AADnet Model Training

- Trained the **AADnet** model on a **Google Cloud VM** using the DTU EEG dataset
- AADnet serves as a baseline/comparison decoder alongside the pretrained VLAAI model
- Training used VM GPU infrastructure to handle the compute requirements of EEG sequence modelling

---

## 2. VLAAI XAI Analysis Pipeline

### 2.1 Pipeline Overview

Built a full publication-grade XAI pipeline (`scripts/run_focused_xai.py`) for the pretrained VLAAI auditory attention decoder.

**Run configuration (full Kaggle run):**
- Dataset: DTU EEG, 18 subjects
- Analysis windows: N = 8,100 (450 per subject)
- Bootstrap iterations: 2,000
- Sign-flip permutations: 5,000
- FDR significance level: α = 0.05
- Hardware: Kaggle GPU T4 x2

---

### 2.2 Sections A–G: Window-Level Analysis

#### Section A — Architecture & Block Ablation

Systematically zeroed and permuted each of VLAAI's 4 recurrent blocks to identify which block drives the decoding decision.

| Block | Method | ΔP | ΔAcc |
|---|---|---|---|
| Block 0 | zero_weights | +0.00003 | −0.000 |
| Block 1 | zero_weights | +0.00100 | +0.006 |
| Block 2 | zero_weights | +0.00020 | +0.000 |
| **Block 3** | **zero_weights** | **+0.02652** | **+0.624** |
| **Block 3** | **permute** | **+0.02573** | **+0.139** |

**Finding:** Block 3 (final recurrent iteration) dominates the decoding decision. Blocks 0–2 can be ablated with negligible effect.

---

#### Section B–D — Channel Importance (Occlusion + Permutation + IG)

Three complementary methods applied to all 64 EEG channels:

- **Channel occlusion** — replace each channel with zero, measure ΔP(attended)
- **Channel permutation** — shuffle each channel across windows, measure ΔP
- **Integrated Gradients** — gradient-based attribution (supporting evidence only)
- BH-FDR correction applied to sign-flip permutation p-values

| Metric | Result |
|---|---|
| Occlusion–Permutation agreement | r = 0.947 |
| Occlusion FDR-significant channels | 42/64 |
| Permutation FDR-significant channels | 38/64 |
| **Robust significant channels** | **45/64** |

Robust criteria: both methods agree in sign + ≥1 FDR-significant + ≥50% subject stability.

**Top 15 channels:**

| Rank | Channel | ROI | Occ ΔP | Perm ΔP |
|---|---|---|---|---|
| 1 | C4 | Central | +0.00176 | +0.00368 |
| 2 | FC4 | Fronto-Central | +0.00174 | +0.00247 |
| 3 | Cz | Central | +0.00146 | +0.00264 |
| 4 | CP4 | Centro-Parietal | +0.00124 | +0.00287 |
| 5 | FT8 | Temporal | +0.00126 | +0.00188 |
| 6 | FC1 | Fronto-Central | +0.00106 | +0.00209 |
| 7 | FC5 | Fronto-Central | +0.00076 | +0.00143 |
| 8 | T8 | Temporal | +0.00069 | +0.00146 |
| 9 | PO4 | Parieto-Occipital | +0.00076 | +0.00124 |
| 10 | FT7 | Temporal | +0.00080 | +0.00104 |
| 11 | FC3 | Fronto-Central | +0.00069 | +0.00111 |
| 12 | CPz | Centro-Parietal | +0.00052 | +0.00131 |
| 13 | P1 | Parietal | +0.00063 | +0.00076 |
| 14 | C5 | Central | +0.00058 | +0.00071 |
| 15 | TP10 | Temporal | +0.00055 | +0.00065 |

**Channel types (45 robust):**
- Facilitatory (removing hurts decoding): 35 channels
- Suppressive (removing helps decoding): 10 channels — P8, AF4, CP6, P4, CP2, Fp1, AF3, M2, CP1, P6

---

#### Section E — ROI-Level Importance

| ROI | Occ ΔP | Significant | #FDR-sig | #Robust |
|---|---|---|---|---|
| Fronto-Central | +0.00072 | YES | 6 | 5 |
| Central | +0.00049 | YES | 4 | 3 |
| Temporal | +0.00042 | YES | 5 | 7 |
| Centro-Parietal | +0.00020 | YES | 6 | 6 |
| Parieto-Occipital | +0.00022 | YES | 4 | 4 |
| Parietal | +0.00008 | YES | 6 | 7 |
| Occipital | +0.00018 | YES | 2 | 3 |
| Frontal | +0.00003 | no | 8 | 9 |
| Mastoid | −0.00010 | YES (suppressive) | 1 | 1 |

---

#### Section F — Frequency-Band Analysis (Exploratory)

Per-channel and per-ROI frequency contribution via Butterworth bandpass filtering (delta / theta / alpha / beta).

| ROI | Best Band | Delta ΔP | Theta ΔP | Alpha ΔP | Beta ΔP |
|---|---|---|---|---|---|
| Fronto-Central | delta | +0.00201 | +0.00001 | +0.00067 | +0.00029 |
| Temporal | delta | +0.00233 | −0.00046 | +0.00025 | −0.00002 |
| Central | beta | +0.00066 | +0.00024 | +0.00054 | +0.00094 |
| Parietal | theta | +0.00025 | +0.00074 | +0.00011 | −0.00045 |
| Parieto-Occipital | theta | +0.00030 | +0.00139 | +0.00016 | +0.00019 |

> **Caution:** Delta-band results should be interpreted carefully — the analysis window is short and delta frequency resolution is limited.

---

#### Section G — Subject Specificity

| Metric | Result |
|---|---|
| N subjects | 18 |
| Mean inter-subject profile r | 0.037 ± 0.262 |
| Per-subject mean ρ vs group | 0.268 ± 0.178 |
| agree_frac vs combined importance | r = +0.313 |
| subj_std vs combined importance | r = +0.857 |

**Finding:** Very low inter-subject agreement (r = 0.037). Individual subjects rely on different channels. The group map should not be interpreted as a universal neural signature.

---

### 2.3 Section H: Subject-Level Statistical Validation (Publication-Grade)

Added to address the nested data problem (windows within subjects) and provide statistically defensible claims.

#### H.1 — Subject Channel Importance
- Per-subject per-channel mean occlusion and permutation ΔP computed
- Saved: `subject_channel_importance.csv` (1,152 rows = 18 subjects × 64 channels)

#### H.2 — Wilcoxon Tests + BH-FDR Across Subjects

| Test | Result |
|---|---|
| Occlusion FDR-significant (subject-level) | **0/64** |
| Permutation FDR-significant (subject-level) | **12/64** |
| Both methods FDR-significant | **0/64** |

#### H.3 — Channel Tiers

| Tier | Criteria | Count |
|---|---|---|
| Tier 1 — High-confidence | Both FDR-sig + same sign + ≥12/18 stability + top 20% effect | **0** |
| Tier 2 — Candidate | ≥1 FDR-sig + same sign + ≥10/18 stability | **12** |
| Tier 3 — Exploratory | Window-level robust only | 33 |

#### H.4 — Split-Half Reliability

| Metric | Value |
|---|---|
| Median Spearman ρ | **0.833** |
| 95% CI | [0.778, 0.887] |
| Iterations | 1,000 |
| Interpretation | HIGH — rankings are stable across subject samples |

#### H.5 — ROI Subject-Level Statistics
- Wilcoxon + BH-FDR applied at ROI level across subjects
- Saved: `subject_level_roi_stats.csv`

#### H.6 — ROI × Frequency Band Subject-Level Statistics
- FDR-significant ROI × band combinations: **5/36**
- Saved: `subject_level_roi_frequency_stats.csv`

---

## 3. Publication Figure Generation

`scripts/generate_publication_xai_figures.py` — 6 figures at 300 dpi (PNG + PDF):

| Figure | Content |
|---|---|
| fig1 | Top-15 channel importance — occlusion & permutation side-by-side |
| fig2 | Core top-15 vs remaining 30 robust channels |
| fig3 | ROI importance with 95% CI |
| fig4 | Facilitatory vs suppressive channels |
| fig5 | Subject × channel heatmap + pairwise Spearman ρ matrix |
| fig6 | Exploratory frequency contribution per band (delta hatched = caution) |

---

## 4. Key Claims

1. **Block 3 architectural dominance is the strongest finding.** Ablating the final recurrent block causes a 62.4% drop in accuracy (ΔP = +0.026), while Blocks 0–2 are negligible. This is robust, large-effect, and replicable.

2. **Channel importance rankings are reproducible across subject samples** (split-half ρ = 0.833, 95% CI [0.778, 0.887]), indicating a stable group-level pattern even though individual channel effects are small.

3. **The window-level count of 45/64 robust channels is a liberal estimate.** Windows are nested within subjects and are not statistically independent. The subject-level analysis is the correct unit: 0/64 channels survive occlusion FDR and 12/64 survive permutation FDR with n = 18 subjects.

4. **Individual channel effect sizes are small** (max ΔP ≈ 0.004 for the top channel, C4). The model's above-chance performance (accuracy = 0.646) is distributed across many channels rather than concentrated in a few.

5. **High inter-subject variability (r = 0.037) suggests the model may learn subject-specific rather than universal channel representations.** Group-level maps should be interpreted accordingly.

6. **Frequency-band findings are exploratory only** and should not be presented as primary results. Delta-band dominance across 28/45 channels is likely a frequency resolution artefact of the short analysis window.

---

## 5. Limitations

| # | Limitation |
|---|---|
| 1 | n = 18 subjects — underpowered for subject-level individual channel tests |
| 2 | Unattended envelope is a circular time-shift proxy, not a true competing speaker |
| 3 | Short analysis window limits delta-band frequency resolution |
| 4 | Electrode montage should be independently verified against DTU documentation |
| 5 | IG uses a zero baseline which may not be neurophysiologically meaningful |
| 6 | Window-level FDR inflates significance due to nested data structure |

---

## 6. Output Files

| File | Description |
|---|---|
| `FOCUSED_XAI_REPORT.txt` | Full window-level analysis report |
| `publication_summary.txt` | Conservative subject-level framing summary |
| `final_important_channels.csv` | 45 robust channels with all stats |
| `subject_channel_importance.csv` | Per-subject per-channel ΔP |
| `subject_level_channel_stats.csv` | Wilcoxon + FDR + CI + Cohen's d per channel |
| `high_confidence_channels.csv` | Tier-1 channels (none in current run) |
| `candidate_channels.csv` | Tier-2 candidate channels (12) |
| `split_half_reliability.csv` | Per-iteration split-half Spearman ρ |
| `subject_level_roi_stats.csv` | ROI Wilcoxon tests across subjects |
| `subject_level_roi_frequency_stats.csv` | ROI × band subject-level tests |
| `publication_figures/` | 6 publication-ready figures (PNG + PDF) |

---

## 7. Infrastructure

| Component | Detail |
|---|---|
| AADnet training | Google Cloud VM (GPU) |
| VLAAI XAI analysis | Kaggle GPU T4 x2 |
| Code repository | GitHub — `manjithadulana98/AAD_XAI` (master) |
| Notebook | `notebooks/kaggle_run_xai.ipynb` — full 8-step pipeline |
