# Pre-registration: external held-out test of StimulusGAT on the KU Leuven AAD dataset

**Status: FROZEN on 2026-11-15**, in the commit that set this line and added the notebooks and
the analysis script (section 9). No notebook had decoded attention on KUL before that commit.
Nothing in sections 2–8 changes after the runs start. Any deviation is reported in RESULTS_LOG.

## 1. Question

All model development so far used DTU. Every design choice (rounds 1-11) was selected on DTU
validation folds, so the DTU test numbers carry some selection optimism even though no choice
was made on DTU test data. KUL has never been used for any decision in this line of work.

The test asks: **does the round-9 StimulusGAT, applied unchanged to KUL with unseen subjects and
unseen stories, decode attention above chance and beat the same linear baseline it beats on
DTU?**

KUL is a harder transfer than another DTU split:

| | DTU (development) | KUL (held out) |
|---|---|---|
| Language | Danish | Dutch |
| Talkers | the same male and female talker in every trial | four male narrators |
| Label | attended talker (male / female) | attended ear (left / right) |
| Presentation | loudspeakers | insert earphones, dichotic ("dry") or HRTF-simulated ±90° |
| EOG channels | yes | none |
| Recording | DTU, BioSemi-64, 512 Hz | KU Leuven, BioSemi-64, released at 128 Hz after a 0.5 Hz high-pass |

## 2. Data

**Source**
- Zenodo record 4004271, version 2.0, CC-BY-NC-SA 4.0.
- Fetched by the private Kaggle kernel `kul-aad-dataset-fetch`. All 19 files matched Zenodo's published MD5 checksums.
- The fetch output was saved as the private Kaggle dataset `dulanamanjitha/kul-dataset`.
- The kernel `kul-aad-structure-check` (version 2, run on that dataset) re-checked the MD5s and recorded the design facts used below.
  - It also recorded a fingerprint of the 32 unpacked wav files: `853dd5093a3d88b2209fc72db741067f`.
  - Every decoding notebook asserts both the MD5s and this fingerprint before loading.
- Neither kernel decodes anything.

**Trials used: experiments 1 and 2 only.** That is 8 trials of about 6.5 min per subject, so about 52 min for each of the 16 subjects.

**Experiment 3 is excluded from training, validation and test.** It has 12 trials of 2 min each, and is excluded for two reasons:
- **Its audio is not new.** Its trials repeat the first 2 min of the experiment-1 parts.
- **It is all one track.** Every experiment-3 trial attends track 1. A rule that simply picks the track-1 narrator would score above chance on experiment-3 test data without using the EEG.

**The design of experiments 1 and 2 removes stimulus-only shortcuts.**
- Each subject hears every part twice, with the same two stories, once attending each.
- So a model that ignores the EEG scores exactly 50% over the two presentations.
- Any accuracy above chance has to come from the EEG–speech relation.

**Envelopes** come from the `_dry.wav` files for both conditions, as the dataset README instructs. Each is truncated to the length of its trial's EEG.

**Facts confirmed by the structure check (2026-11-15)**
- All 320 trials are (samples × 64) at 128 Hz. The channel labels (`FileHeader.Channels.Label`) equal our BioSemi-64 montage order, 64 of 64.
- In experiments 1 and 2, the `part` field equals the part in the stimulus file name (128 of 128 trials).
- Each subject hears each part twice, in the same condition:
  - experiment 1 attends track 1 and experiment 2 attends track 2;
  - the attended *ear* is the same in both presentations;
  - the two tracks swap ears between them.
- Every part is dry for 8 subjects and HRTF for the other 8.
- Experiment 3 always attends track 1. Its `part` field is the presentation index (1–12), and the audio played is rep_part((part − 1) mod 4 + 1).
- Each subject has 52.3 min of EEG in experiments 1 and 2.
- The EEG ends 4–5 s before the stimulus (73 s before it for part 4, which is 472 s long). As the README instructs, the audio is truncated to the EEG length.

## 3. Frozen configuration (round 9 on DTU; nothing tuned on KUL)

**EEG**
- Average reference over the 64 channels.
- Band-pass 1–31 Hz, resampled 128 → 64 Hz.
- Each window z-scored per channel.
- Euclidean alignment (EA) per subject, fitted separately inside the train, val and test sets (shrinkage 1e-3).
- No EOG regression (KUL has no EOG). This is the round-9 configuration, not round 11.

**Speech**
- Power-law gammatone sub-band envelope, 15 bands summed.
- Band-passed 1–8 Hz, at 64 Hz.

**Graph**
- Distance-based k = 6 nearest-neighbour electrode graph built from `config/aadnet_dtu_channel_montage.csv`, the BioSemi-64 layout used by both datasets.
- The KUL channel labels already match this file, 64 of 64. The notebook asserts the match again before training.

**Model**
- `StimulusGAT(film_pooling="corr", speech_kernel_sizes=(33, 33), freq_length_normalize=True)`.
- All other arguments at their defaults.

**Training**
- AdamW, learning rate 1e-3, weight decay 1e-4, batch 64, 25 epochs.
- Class-weighted cross-entropy.
- The checkpoint kept is the epoch with the best class-balanced 5 s validation accuracy.
- Training windows: 5 s with a 2.5 s hop.

**Evaluation**
- Non-overlapping test windows of 5 s, plus 1 s and 2 s windows.
- Decisions at 10, 20 and 40 s use the sign of the summed 5 s logit differences of consecutive windows within one trial.

**Ensemble**
- 5 seeds (42, 1, 2, 3, 4), averaging logit differences.
- This is the same headline model as the DTU round-9 result.

**Baseline: TRF + EA**
- Backward ridge decoder on 1–8 Hz EEG at 64 Hz, lags 0–400 ms, 15-band power-law envelope targets.
- Ridge alpha (grid 1 to 1e9) and the raw / standardised variant chosen per fold by validation correlation.
- EA per subject per split.
- Same folds and 5 s test windows as StimulusGAT.
- Decisions at 10, 20 and 40 s recompute the correlation over the whole window. This is the standard rule for a linear decoder, and on DTU it is the TRF's own rule.
- The summed-score rule is reported descriptively.

## 4. Split

**Folds**
- `subject_group_story_out` with the content group set to the part (1–4), read from the stimulus file name.
- Story folds: parts {1, 2} vs parts {3, 4}.
- 8 groups of 2 test subjects, 2 validation subjects per fold, seed 42.
- That gives 16 folds. Every experiment-1/2 trial is tested exactly once.
- In each fold: train on 12 subjects × the other half of the content; validate on 2 other subjects × the held-out half; test on the 2 test subjects × the held-out half.
- `assert_story_disjoint_fold_integrity` runs on every fold: no subject is shared between train, val and test, and no part is shared between train and val/test.

**Why these story folds**
- The README states that each story is split into two parts, and that consecutive presentations play "the next part of the two stories".
- So parts {1, 2} and parts {3, 4} hold different story pairs, and holding out a half holds out whole stories.
- If that reading of the README were wrong, the split would still be disjoint at the content level: no audio segment is ever shared between training and test.

## 5. Runs

- StimulusGAT: `notebooks/kaggle_run_kul_external_sgat.py`.
  - Two GPU kernels: part A (seeds 42, 1, 2) and part B (seeds 3, 4), about 6 GPU-h in total.
- TRF + EA: `notebooks/kaggle_run_kul_external_trf.py`, one CPU kernel.
- All three save, for every fold:
  - per-window test and validation outputs and the control outputs;
  - the fold metadata;
  - checkpoints (StimulusGAT) or weights (TRF);
  - each trial's design information.
- Analysis: `scripts/analyse_kul_external.py <part A dir> <part B dir> <TRF dir>`, run once on the downloaded outputs.

## 6. Confirmatory analysis

Subjects are the unit of analysis (n = 16). All tests are two-sided Wilcoxon signed-rank tests on per-subject accuracies unless stated.

**H1 (above chance).** At 5 s, the ensemble's per-subject accuracy is above 0.5 (one-sided Wilcoxon, p < 0.05).

**H2 (vs the linear baseline).** Ensemble minus TRF + EA at 5, 10, 20 and 40 s, with Holm correction over the 4 lengths. The outcome is decided by these rules, applied in order:
- **"mixed":** there are Holm-significant differences (p < 0.05) in both directions;
- **"better":** the difference is positive with Holm p < 0.05 at one or more lengths, and the mean difference is ≥ −0.5 points at every length;
- **"worse":** the difference is negative with Holm p < 0.05 at one or more lengths;
- **"no difference":** neither of the above.

**Controls.** Both are scored against the true labels.
- *Shifted envelope:* both candidate envelopes are rolled by 2.5 s inside each window.
- *Trial swap:* each test window is paired with the window at the same position (wrapping round) in the same subject's trial of the *other* held-out part, from the same experiment. The swapped audio therefore never overlaps the true audio.
  - The DTU swap is not reused unchanged. On KUL two trials share identical audio, so a swap could land on a time-aligned copy of the true stimulus.
- *Pass rule (same as DTU):* subject Wilcoxon vs 0.5 has p ≥ 0.05 at 5, 10, 20 and 40 s, and no fold reaches 0.6 at 5 s.

**Claim rule.** We claim that StimulusGAT generalises to KUL only if H1 holds, H2 is "better", and both controls pass. Any other outcome is reported as stated, as a limitation of the DTU result.

## 7. Pre-specified descriptive checks (no decision attached)

- **Narrator bias.** Per subject, the share of test windows where the model picks the track-1 narrator. Expected 0.5. Wilcoxon vs 0.5 reported.
- **Side bias.** The share of windows where the model picks the left candidate.
  - Expected 0.5: the same network scores both candidates, and each ear carries track 1 and track 2 equally often.
  - A clear deviation would point to a bug or to an asymmetry in the stimuli.
- **Accuracy by condition.** Dry vs HRTF, per subject.
- **Single-model results.** Mean single-seed accuracy at 1, 2, 5 and 10 s, and ensemble accuracy at 1 and 2 s.
- **DTU vs KUL.** The per-subject table, and the KUL ensemble-minus-TRF gap next to the DTU round-9 gap.

## 8. Conduct

- **One run of the frozen configuration.** If a kernel crashes, the bug is fixed and the identical configuration is re-run. Every deviation is listed in the RESULTS_LOG entry.
- **No retuning on KUL.** No configuration is changed after any KUL result is seen. Any later KUL analysis is labelled exploratory: for example, adding experiment 3, tuning on KUL, or an interpretability comparison.
- **Every outcome is reported**, in RESULTS_LOG and in the thesis.

## 9. Implementation, completed before the freeze (none of it touches the model)

1. **KUL loader** (`aad_xai.data.kul_dataset`).
   - `envelope_source="dry"` builds HRTF trials' envelopes from the dry files.
   - `trial_info` records, per trial: experiment, part, condition, attended ear and track, the files presented and the files used for envelopes, and the track on each ear.
   - Multi-band envelopes are truncated along the time axis.
   - The experiment filter now runs before EEG preprocessing.
   - Unit tests in `tests/test_kul_external.py`.
2. **Content groups** (`aad_xai.data.cv_splits`).
   - Added `kul_story_part` (part 1–4 for experiments 1 and 2; None for experiment 3) and `KUL_STORY_FOLDS = ((1, 2), (3, 4))`.
   - Fixed `_content_group` for experiment 3. The old `ceil(part / 3)` matched the audio played in only 64 of 192 trials; it is now `(part − 1) mod 4 + 1`.
     - This protocol does not use experiment 3, so the fix does not affect it.
     - None of the logged KUL results used this function with experiment 3.
   - Unit tests cover the 16-fold split: every trial tested once, 12 training subjects, held-out halves.
3. **KUL trial-swap permutation** (`aad_xai.data.negatives`): `same_stratum_other_content_partners` and `partner_trial_permutation`.
   - Unit tests cover the pairing, the wrap-round, and rejection of ambiguous or self pairings.
   - The notebooks also assert, for every fold, that no swapped window plays the same part.
4. **Notebooks** (section 5).
   - Data-check assertions: Zenodo MD5s, the stimuli fingerprint, channel order, 16 subjects × 8 trials, 64 channels, each part twice per subject with each track attended once, and dry envelopes.
   - Smoke-tested end to end on a fake 16-subject KUL release (1 epoch).
5. **Offline analysis script** (`scripts/analyse_kul_external.py`), implementing sections 6 and 7.
   - Tested in `tests/test_analyse_kul_external.py` on synthetic outputs in the notebooks' format:
     - a null case;
     - a strong model with a weak baseline;
     - leaky controls, which block the claim;
     - a stronger baseline;
     - the outcome-rule edge cases;
     - a tampered swap, which the integrity check catches.

## 10. Known limitations

- **EA is transductive.** It uses the test subject's unlabelled EEG from the held-out half (about 26 min). It is applied identically to StimulusGAT and the TRF.
- **No EOG on KUL**, so ocular activity cannot be regressed out. Gaze alone cannot select a candidate, for two reasons:
  - the same network scores both candidates, so the score cannot depend on side;
  - the dry envelopes carry no side information.

  The shifted-envelope control would expose any EEG-only cue. On DTU, removing EOG (round 11) did not lower accuracy.
- **16 subjects** give power for paired differences of roughly 2–3 points.
- **Narrators may repeat across story folds**, as on DTU, where the same two talkers read every story. Audio content is never shared.
