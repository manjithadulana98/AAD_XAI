# AAD_XAI Methodology Reference

Reference document for the two auditory-attention-decoding (AAD) XAI pipelines in this repo:

- **VLAAI** — `scripts/run_focused_xai.py` (group-level model, 1 pretrained checkpoint, full statistical layer)
- **AADNet** — `notebooks/kaggle_run_xai_aadnet.py` (18 per-subject models × 8 folds, thinner statistical layer)

Both operate on the DTU dataset, a 64-channel EEG montage mapped to 9 ROIs via `config/dtu_channel_montage.csv` (Frontal, Fronto-Central, Temporal, Central, Centro-Parietal, Parietal, Parieto-Occipital, Occipital, Mastoid).

This document has three parts: **(1) methodology** — what each technique does and how the two pipelines implement it differently, **(2) results** — the actual numbers produced by a full run of each pipeline, and **(3) a decision framework** — what the evidence currently licenses you to claim, and what it doesn't yet.

---

## Part 1 — Methodology

### 0. Why explainability at all

Both models can already classify/score attended-vs-unattended speech per window. The XAI pipelines exist to answer a harder question: on what basis? Every technique below removes or perturbs a different piece of the input or the model and watches the prediction move — occlusion and permutation edit the input, integrated gradients traces the prediction back through calculus, architecture ablation edits the network itself, and the reliability/subject-specificity/sanity-check layers ask whether any of the above holds up under resampling or is even sensitive to the trained weights at all.

### 1. Occlusion

Each of the 64 channels is replaced (zero, or channel-mean for VLAAI) across the whole window, one at a time; `ΔP = P(attended)_original − P(attended)_occluded`. Positive ΔP = facilitatory (removing the channel hurt the model), negative = suppressive (model does better without it).

- **VLAAI**: `compute_channel_occlusion()`, `scripts/run_focused_xai.py:399`. Supports `--occlusion-mode {zero,mean}` (default zero). Bootstrap CI (`bootstrap_ci`, line 189, `n_boot=500`, resamples windows) and sign-flip p-value (`sign_flip_p_value`, line 202, `n_perm=5000`) computed per channel.
- **AADNet**: `subject_perturbations()`, `notebooks/kaggle_run_xai_aadnet.py:463-468`. Zero-only. No per-channel window-level CI/p-value — only the subject-level Wilcoxon test (§4) is computed. Run per (subject, fold) using that fold's own checkpoint, concatenated by window index.

### 2. Permutation

Instead of deleting a channel, its values are shuffled *across windows* — channel `ch` in window *i* takes channel `ch`'s values from a randomly chosen window *j*, preserving the channel's marginal distribution but breaking the window-specific pairing.

- **VLAAI**: `compute_channel_permutation()`, line 426. Shuffles across **any** sampled window (can span subjects, since one shared model).
- **AADNet**: lines 470-477. Shuffles only **within the same subject and fold** (`fold_perm_local`) — a difference forced by architecture (each fold has its own checkpoint; windows from other folds aren't valid substitutes), not a shortcut.

**Sign-flip permutation test** (identical algorithm both pipelines, `sign_flip_p_value`, VLAAI line 202 / AADNet line 329): observed `|mean(ΔP)|` compared against 5,000 null draws where every window's ΔP sign is flipped independently at random.

### 3. Integrated Gradients (IG)

Walks a straight line from a zero baseline to the real input, accumulating the gradient of the "attended" logit at each step, scaled by `(input − baseline)`. Mean absolute attribution over time collapses this to one score per channel. Treated as **corroborating evidence only** in both pipelines — never folded into the combined ranking.

- **VLAAI**: `compute_integrated_gradients_summary()`, line 451. Uses **captum**'s `IntegratedGradients`, 50 steps (`--ig-steps`), run on 30 windows total (`--ig-samples`, default `n_ig=30`), batched by 5 to avoid CPU OOM.
- **AADNet**: `subject_perturbations()`, lines 479-502. Hand-rolled Riemann-sum IG (no captum), 32 steps (`IG_STEPS`, line 85), ~30 windows **per subject** (`N_IG_WINDOWS`, line 84) — selection is deterministic: folds visited in ascending order, first `take` windows of each fold accumulated until 30 total (`ig_taken` counter, not randomly subsampled).

**Autograd note** (relevant to Part B/C of the Phase-1 sanity-check work): both VLAAI's captum-based IG and AADNet's hand-rolled `torch.autograd.grad` IG need an active gradient tape to compute attributions. `cascading_randomization` (§9) calls every `attr_fn` inside an outer `torch.no_grad()`. Rather than relying on an unverified assumption about whether captum internally re-enables gradients, both pipelines' IG `attr_fn` wrap their attribution call in an explicit local `with torch.enable_grad():` — cheap, always-correct, and removes the dependency on captum's internal implementation details (captum was not available in this environment to verify directly).

### 4. The significance layer

- **Bootstrap CI**: `bootstrap_ci()` (VLAAI line 189, AADNet line 318) — percentile bootstrap, resampling either windows or subjects depending on call site, 500-2000 iterations, 95% CI from the 2.5th/97.5th percentiles.
- **Wilcoxon signed-rank test**: subject is the unit — given e.g. 18 subjects' mean ΔP for one channel, is the median reliably non-zero? This is what feeds FDR correction.
- **Benjamini-Hochberg FDR**: `fdr_correction()` (VLAAI line 215, AADNet line 338) — standard BH step-up, applied at multiple scopes in VLAAI (64 channels, 9 ROIs, 4 top-K sizes, 36 ROI×band combos) and only once (64 channels) in AADNet.
- **Hierarchical FDR** (VLAAI only, `run_hierarchical_channel_fdr()`): two-stage — ROIs that clear FDR first (the "gate"), then only channels within those surviving ROIs get re-tested against a much smaller correction. AADNet has no equivalent.

### 5. ROI aggregation vs. ROI-group ablation — two different experiments

- **ROI aggregation** (both pipelines): the 64 single-channel occlusion results are averaged within each of the 9 regions. No new inference — pure post-hoc pooling of results already in hand.
- **ROI-group ablation** (VLAAI only, `run_roi_group_ablation()`, "Section E2"): every channel in a region is zeroed **simultaneously** in one forward pass — a genuinely different, larger-effect experiment ("much larger effects than single-channel occlusion because the combined signal loss is additive," per the code's own docstring). AADNet has no equivalent.
- **Cumulative top-K ablation** (VLAAI only, `run_topk_ablation()`, "Section E3"): top 5/10/15/20 channels by combined_score zeroed together, tracing how fast decoding degrades.

### 6. Frequency-band analysis

Both isolate delta/theta/alpha/beta with a 4th-order Butterworth filter, then **subtract** that band back out of the signal (ablative, not "keep only this band"). VLAAI uses `sosfiltfilt` with 64-sample reflect padding (more numerically stable for narrow low-frequency bands); AADNet uses `filtfilt` with no padding. VLAAI runs both per-channel and whole-ROI granularity (`--roi-frequency-mode`); AADNet only implements the whole-ROI version.

### 7. Architecture ablation

Edits the network, not the input.

| Model | Target | Mechanism |
|---|---|---|
| VLAAI | 4 recurrent iterations (`extractor`, `block_denses[0..3]`, `output_context`, `final_dense`) | zero the block's weights in place + restore, **or** permute its output activations via a forward hook |
| AADNet | `inception_1_eeg`, `inception_1_aud`, `fc1` | zero the module's output activation via a forward hook; weights never touched |

Both answers land on the same shape: one late-stage component is fully load-bearing (VLAAI's final recurrent block; AADNet's fusion layer `fc1`), the rest are largely redundant.

### 8. Cross-subject checks

- **Subject specificity** (VLAAI: `run_subject_specificity_analysis()`, "Section G"; AADNet: pairwise-only version): per-subject occlusion profile vs. group-mean profile, Spearman rank correlation (rank-based since only relative ordering matters, not ΔP magnitude). AADNet only computes pairwise subject-to-subject correlations, no comparison against a group-mean profile.
- **Split-half reliability**: subject pool split randomly in two 1,000 times; each half's mean channel-importance vector computed independently; Spearman r between the two halves, repeated 1,000 times, median + 95% CI reported. VLAAI ranks by `|occ|+|perm|` combined; AADNet by occlusion only.

### 9. Sanity checks (Adebayo et al., cascading parameter randomization) — Phase 1 of the validation ladder

**Already implemented generically** in `src/aad_xai/xai/sanity_checks.py`:
- `randomize_parameters(model)` (line 11): deep-copies the model, calls `.reset_parameters()` on every submodule that has one. Note: `nn.LayerNorm.reset_parameters()` just resets affine weight/bias to 1/0 — not a genuine randomization — so LayerNorm "randomization" is really an identity reset; only Conv1d/Linear are actually randomized.
- `cascading_randomization(model, attr_fn, x)` (line 23): deep-copies `model` once, computes `attr_fn(m, x)` as `"__original__"`, then iterates `model.named_children()` **in reverse** (last layer first), cumulatively resetting each child's parameters and recording `attr_fn(m, x)` after each step. Everything runs inside an outer `torch.no_grad()`.

Prior usage: `scripts/run_vlaai_xai.py:328-350` (passes the raw `VLAAIPyTorch` model — correct, gives 4 real cascade depths via its `named_children()`); `scripts/run_xai_comprehensive.py:789-906` (`section_g`, passes the `decision` wrapper instead — a design oversight, since the wrapper has only one named child and the cascade degenerates to a single step). **When wiring this into a new section, always pass the raw model (`VLAAIPyTorch`, or `model.model` for AADNet's `ExternalAADNet`), never the decision/wrapper object.**

A rank-correlation-based sanity check (Spearman ρ between the original channel-importance vector and each cascade depth's vector, for occlusion and IG separately) is the Phase-1 addition to both `run_focused_xai.py` and the AADNet notebook — see the plan file for the exact insertion points, output schema (`sanity_check_results.csv`: `[model, randomization_depth, method, spearman_rho, p_value]`), and the `torch.enable_grad()` fix required specifically for AADNet's hand-rolled IG. No label-shuffled checkpoint/training script exists anywhere in the repo for either model — this is documented as an explicit "not run" limitation, not silently skipped.

### 10. Synthesis (VLAAI only)

- **contribution_type**: `facilitatory` if occ>0 and perm>0; `suppressive` if both <0; otherwise `mixed`.
- **robust_significant**: same sign (not mixed) AND at least one method FDR-significant AND ≥50% of subjects agree in sign.
- **combined_score**: average of the independently z-scored `|occlusion|` and `|permutation|` effect sizes across the 64 channels — this, not raw p-values, is what ranks channels.

AADNet has none of this classification machinery (Phase 5 of the validation-ladder work backfills it).

### 11. Publication figures

`scripts/generate_publication_xai_figures.py` builds exclusively from the window-level `robust_significant` set. "Top 15" / "core vs. distributed" is a hard positional rank-15 cutoff by `combined_score`. Ships an explicit conservative-language convention: use "suggests," "is consistent with," "should be interpreted cautiously"; avoid "proves," "definitively shows."

---

## Part 2 — Results (from a full run of each pipeline)

### Baseline performance
| | AADNet | VLAAI |
|---|---|---|
| Metric | mean P(attended) accuracy | AAD accuracy (correlation-margin rule) |
| Value | 0.776 | 0.646 |
| n | 18 subjects × 8 folds | 8,100 windows, 18 subjects |

Not directly comparable — different metrics on different model designs. VLAAI's "unattended" reference is a circular time-shift proxy, not a true competing speaker.

### Channel-level significance — the number depends on the unit of analysis
| Test | Unit | n | Occ. sig. | Perm. sig. | Both |
|---|---|---|---|---|---|
| Window-level FDR (VLAAI) | 8,100 windows (non-independent) | 8100 | 42/64 | 38/64 | not tested |
| Subject-level Wilcoxon+FDR (VLAAI) | 18 subjects | 18 | **0/64** | 12/64 | **0/64** |
| Subject-level Wilcoxon+FDR (AADNet) | 18 subjects | 18 | 10/64 | 46/64 | not cross-tabulated |

At the *correct* statistical unit (18 independent subjects, not 8,100 correlated windows), VLAAI's occlusion channel-level significance collapses to zero; only permutation clears subject-level FDR (12 channels: C4, FC4, CP4, Cz, FC1, FT8, T8, FT7, FC3, C5, CP3, F6), and none clear both methods — `high_confidence_channels.csv` (the strictest tier) is empty.

### Core-ROI composite (VLAAI)
Pooling the 28 channels of the four strongest ROIs (Fronto-Central, Central, Temporal, Centro-Parietal) at the subject level (n=18):
- Occlusion: mean ΔP = 0.000446, Cohen's d = **1.19**, Wilcoxon p = 7.6×10⁻⁵
- Permutation: mean ΔP = 0.000943, Cohen's d = **1.70**, Wilcoxon p = 1.5×10⁻⁵

No single channel individually clears the strictest bar, but the composite region does — a large effect at the correct unit of analysis.

### ROI-group ablation (VLAAI, whole-region simultaneous zeroing)
Fronto-Central, Temporal, Central, Parieto-Occipital, Parietal, Occipital all FDR-significant; Frontal (p=0.56) and Centro-Parietal not significant as whole regions (internal sign cancellation); Mastoid significantly **negative** (suppressive).

### Cumulative top-K ablation (VLAAI)
K=5: ΔP=0.0105; K=10: 0.0156; K=15: 0.0211; K=20: 0.0214 — the K=15→20 step (+0.0003) is ~20× smaller than every prior step, empirically supporting a "top-15" cutoff rather than an arbitrary round number.

### Architecture ablation
| | AADNet (`fc1` zeroed) | VLAAI (Block 3 zeroed) | VLAAI (Block 3 permuted) |
|---|---|---|---|
| Accuracy | 0.776 → 0.000 (every fold) | 0.646 → 0.022 | 0.646 → 0.507 |

VLAAI's zero-weight Block-3 ablation drives accuracy **below chance** (0.022 vs. 0.5) — not just information loss, but a systematically inverted decision; permuting (rather than zeroing) only degrades to ≈chance. Blocks 0-2 are near-inert (ΔAcc between −0.0001 and +0.007).

### Reliability & cross-subject generalization
| | AADNet | VLAAI |
|---|---|---|
| Split-half reliability | median ρ = 0.333 [0.174, 0.488] | median r = 0.833 [0.778, 0.887] |
| Cross-subject profile agreement | mean pairwise ρ = +0.072 | mean subject-vs-group r = +0.037 ± 0.262 (range −0.06 to +0.56) |

---

## Part 3 — Decision framework

**The core distinction**: statistical significance, effect size, and correct unit of analysis are three separate things, and this result set exhibits failure modes in all three if not read carefully.

- Large N (8,100 windows) inflates significance, not importance — a ΔP of 0.0002 can clear p<0.05 at that N. `combined_score` intentionally ranks by z-scored effect size, not p-value, for this reason.
- Windows are not subjects — the window-level FDR pass treats correlated samples from 18 people as independent. Useful as an exploratory first pass; the subject-level test (n=18) is the one that should govern generalization claims.
- Aggregation (core-ROI composite) is a principled a-priori pooling, not p-hacking — it recovers power that individually underpowered channels lack.

### Confidence ladder
- **High confidence**: both models concentrate their decision in one late-stage, load-bearing component (VLAAI Block 3, AADNet `fc1`) with effect sizes far above any noise floor. Central/Fronto-Central/Temporal ROIs carry the most decodable information (large, subject-level-significant core-ROI composite in both models).
- **Moderate, usable with caveats**: the ~6-12 channel shortlist (Cz, FC4, CP4, CPz, TP10, P1, + the 12 subject-level tier-2 candidates) is the most defensible individual-electrode list, but zero channels clear both methods at the subject level. The window-level "45/64 robust" and "42/64 FDR-sig" statements are real but computed at a statistically liberal unit — fine as sensitivity analysis, risky as a headline claim.
- **Exploratory only**: per-channel/per-ROI frequency-band contributions — the pipeline's own authors flag delta-band effects as suspect given the short analysis window.
- **Not yet supportable**: any claim about a specific individual's channel importance, or any single-channel claim not backed by the core-ROI composite — cross-subject profile correlation is r ≈ 0.03-0.07 in both models.

### To raise a tier
1. Report channel-level claims at the subject level (12-channel tier-2 list), not the window level (42/64, 45/64).
2. Backfill AADNet's statistical layer (bootstrap CI, combined_score, robust/tiering, hierarchical FDR) to VLAAI parity so its channel-level counts can be cross-checked the same way (Phase 5 of the validation ladder).
3. Treat the core-ROI composite as the primary spatial claim; individual top-15 lists are supporting/descriptive only.
4. Run held-out-subject validation (Phase 4) before any claim about generalizing to new subjects — cross-subject consistency (r ≈ 0.03-0.07) is the weakest link in both pipelines and isn't yet directly tested.
5. Keep frequency-band findings out of headline claims until the analysis window is lengthened or an edge-effect correction is validated.

---

*Sources: `scripts/run_focused_xai.py`, `notebooks/kaggle_run_xai_aadnet.py`, `scripts/generate_publication_xai_figures.py`, `src/aad_xai/xai/sanity_checks.py`, `src/aad_xai/models/vlaai_pytorch.py`, `src/aad_xai/models/aadnet_external.py`, `external/AADNet/aadnet/EnvelopeAAD.py`, `config/dtu_channel_montage.csv`, and the corresponding `xai_results_focused`/`xai_results_aadnet` output directories from a full run of each pipeline. Line numbers verified against the repository state as of this writing; re-check if this document is read after further edits to the pipeline scripts.*
