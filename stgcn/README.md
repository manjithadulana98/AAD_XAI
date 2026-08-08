# ST-GCN — Third Model (Progress & Phase Plan)

Reference: Wang, Cai & Li, "EEG-based Auditory Attention Detection with
Spatiotemporal Graph and Graph Convolutional Network," Interspeech 2023,
pp. 1144-1148, doi: 10.21437/Interspeech.2023-620.

ST-GCN is the thesis's third model, alongside the existing AADNet (18
per-subject models) and VLAAI (1 group-level model). Work proceeds in six
strict, independently-gated phases — each phase's results are reported and
reviewed before the next one starts, rather than pushing straight through.

This file is the single source of truth for phase status. Update it whenever
a phase starts, finishes, or its scope changes.

## Phase plan

| Phase | Scope | Status |
|---|---|---|
| 1 | Adjacency matrix construction | **Done** |
| 2 | GCN-only training (stages a+b, no temporal attention) | **Done — flagged, see below** |
| 3 | Temporal attention (stage c) | Not started |
| 4 | Window-length sweep | Not started |
| 5 | XAI wiring (occlusion/permutation, matching AADNet/VLAAI treatment) | Not started |
| 6 | Centrality analysis on the adjacency | Not started |

---

## Phase 1 — Adjacency matrix construction (done)

**Code:** `stgcn/adjacency.py`, `stgcn/channel_order_check.py`

Two pre-determined graph adjacency variants built from channel montage CSVs:

- **`build_adjacency_distance(montage, k)`** — inverse-Euclidean-distance
  weighted, k-nearest-neighbors (k=6 used in Phase 2).
- **`build_adjacency_roi(montage)`** — binary, 1 if two channels share an
  ROI, else 0. Block-diagonal by ROI under a channel reordering. **Not yet
  used in any training run** — only the distance variant has been tried.

Built twice, against two different channel montages, because the project has
two parallel "real" DTU pipelines with different channel orders:

- `config/dtu_channel_montage.csv` (VLAAI's real order) →
  `stgcn/outputs/phase1_adjacency/`
- `config/aadnet_dtu_channel_montage.csv` (AADNet's real, Fuglsang-64 order,
  verified 64/64 against `external/AADNet/aadnet/dataset.py`) →
  `stgcn/outputs/phase1_adjacency_aadnet/`

Each output directory contains `adjacency_distance.png`,
`adjacency_roi.png` (heatmaps), and `adjacency_degree_by_roi.csv`
(per-channel degree under both variants). Diagnostics passed: no zero
rows/columns in either matrix; neighboring channel triples show clearly
higher connectivity than distant pairs; Central/Fronto-Central channels show
higher degree than Mastoid in both variants.

Phase 2 trains against the **AADNet-montage, distance-k6** variant, since
Phase 2 sources its training windows from AADNet's raw pipeline (channel
order must match).

---

## Phase 2 — GCN-only training, stages (a)+(b) (done — flagged)

**Code:** `stgcn/model.py` (`GraphConvKW`, `STGCNGCNOnly`),
`notebooks/kaggle_train_stgcn_gcn_only.py/.ipynb`,
`notebooks/kaggle_diag_stgcn_overfit_check.py/.ipynb`

Full write-up: **`stgcn/outputs/phase2_gcn_only/PHASE2_REPORT.md`**

### What was built

- K-order (K=5) Chebyshev-style spectral graph convolution — an independent
  learnable coefficient per (kernel, basis-term) pair, chosen over a naive
  single-term Kipf-Welling layer specifically because that degenerates to
  one shared filter when the input feature dimension is 1 (see `model.py`'s
  module docstring for the full rejection rationale, including why a
  two-term basis was also tried and rejected).
- Trained on AADNet's real, leakage-safe subject-independent (SI) DTU
  pipeline (`DTUDataset.createSICrossValidation`, reused unmodified) at a
  1-second decision window, with `duplicate=False` (AADNet's built-in
  label-flip augmentation disabled — valid for its own dual-stream
  architecture, but would silently duplicate identical EEG windows under
  opposite labels for an EEG-only model).
- 2,632 trainable parameters (paper reference: ~2,930).
- 18 subjects × 8 SI folds = 144 fold-trainings, 40 epochs, batch 32,
  2,000 randomly-sampled training windows/epoch, seed 42.

### Two issues found and fixed en route

1. **CUDA "no kernel image available" (P100/sm_60)** — platform issue
   (Kaggle occasionally assigns old P100 GPUs incompatible with the
   pre-installed torch wheel). Mitigated by rewriting the graph-conv layer's
   `einsum` to `matmul`/reshape. Did not recur once a T4 was assigned.
2. **`RuntimeError: expected scalar type Float but found Double`** — a real
   bug: `DTUDataset` yields `float64` EEG tensors (raw DTU `.mat` arrays are
   double-precision) against a `float32` graph-conv basis buffer. Fixed by
   casting `eeg.to(DEVICE).float()` at the training-notebook call sites.

Both committed: `e1f5a69` (CUDA), `b15d7bb` (dtype).

### Result

| Metric | Value | Target |
|---|---|---|
| Mean test accuracy | **0.4041** (sd 0.1488) | 0.60–0.85 |
| Paper reference (full model, w/ temporal attention) | — | 0.731 ± 0.076 |
| Folds below chance | 101/144 (70%) | — |
| Subjects with per-fold mean ≥ 0.5 | **0/18** | — |

Trips the notebook's own stop condition (mean accuracy outside [0.60, 0.85]
→ report and debug, don't retune blindly). Raw per-fold and per-subject
numbers: `gcn_only_fold_accuracy.csv`, `gcn_only_per_subject_summary.csv`.
Representative-fold training curves (`gcn_only_training_curves.png`) show
train accuracy stuck at ~50-54% for all 40 epochs — the model doesn't fit
its own training data, not just failing to generalize.

### Overfit sanity check — ruled out one failure mode

Tested whether the model can memorize a small, fixed batch at all (standard
sanity check for "are gradients/wiring broken").

- **First attempt was invalid, self-caught:** took the first 32 windows of
  one subject's training split. Reached 93.8% "accuracy," but the batch was
  **32-zeros/0-ones** — DTU windows are drawn sequentially within a trial
  (one fixed attended speaker throughout), so "the first N windows" landed
  entirely inside one trial. Memorizing an all-one-class batch only requires
  learning a constant bias, not real discrimination — discarded.
- **Corrected version** (scan until 16 examples of each label are found):
  climbs cleanly from chance (epoch 0: 50%) to **100% train accuracy by
  epoch 100**, loss monotonically decreasing 0.95 → 0.036 through epoch 300.

**This rules out dead gradients / broken wiring / degenerate adjacency** —
the mechanical pipeline is sound.

### What this does *not* rule out — open before committing to Phase 3

The overfit check is a weak test: it only proves the model isn't completely
broken, not that its 0.404 mean reflects a real, well-understood ceiling.
Flagged but **not yet checked**:

1. **The paper's own ablation table has not been confirmed to show a
   GCN-only (no temporal attention) result near/below chance.** The Phase 2
   report's framing — "this is the expected shape of that ablation story" —
   is a plausible assumption, not a verified fact.
2. **1-second windows are unvalidated territory for this project's
   pipeline.** Every other result in this thesis (VLAAI, AADNet) runs at
   10s windows. Whether AADNet's own *full* architecture (inception
   branches included) can beat chance at 1s with this same
   `duplicate=False` pipeline has not been tested — if it also struggles,
   that says "1s is hard for everyone," strengthening Phase 2's reading; if
   it doesn't, that points at something specific to this submodel instead.
3. **Only ~2% of the training pool is seen per epoch** (2,000 sampled
   windows out of ~94k available, × 40 epochs = 80k total window-exposures)
   — a plain undertraining risk independent of architecture, not addressed
   by the overfit check (which never tests learning from the full
   distribution).
4. **Only the distance-k6 adjacency variant has been tried.** The ROI-based
   variant (`build_adjacency_roi`, built in Phase 1, never trained against)
   might behave differently — untested adjacency sensitivity.

**Recommended next check before Phase 3:** run AADNet's own full
architecture (or a trivial linear-on-envelope-power baseline) through this
same 1s / `duplicate=False` pipeline and see if it beats chance. That
isolates "1s is hard for everyone here" from "1s is hard specifically for
GCN-only" — and determines whether Phase 3 (temporal attention) is really
the right next lever, or whether the window-length/training-budget choices
need revisiting first.

---

## Phases 3–6 (not started)

Scope not yet finalized in detail — sketched at a high level in the original
plan, to be gated the same way as Phases 1–2 (propose → confirm → implement
→ report) when each one starts:

- **Phase 3 — Temporal attention (stage c).** Add the paper's temporal
  attention module over the graph-conv output's time axis, replacing or
  supplementing the current global-average-pool, before the FC head.
- **Phase 4 — Window-length sweep.** Re-run across multiple decision-window
  lengths (the project already sweeps 0.5–30s elsewhere) to characterize how
  accuracy scales with window length for this architecture.
- **Phase 5 — XAI wiring.** Occlusion/permutation analysis on the trained
  ST-GCN, matching the statistical treatment already applied to AADNet and
  VLAAI (`scripts/run_focused_xai.py`'s conventions — combined_score,
  window-level robust, split-half reliability, BH-FDR).
- **Phase 6 — Centrality analysis.** Graph-centrality analysis on the fixed
  adjacency itself (e.g., which channels/ROIs the graph structure makes
  structurally central), as an independent cross-check against the XAI
  channel-importance rankings from Phase 5 and from AADNet/VLAAI.

---

## Directory map

```
stgcn/
  README.md                          -- this file
  adjacency.py                       -- Phase 1: build_adjacency_distance, build_adjacency_roi
  channel_order_check.py             -- Phase 1: montage/channel-order verification
  model.py                           -- Phase 2: GraphConvKW, STGCNGCNOnly
  outputs/
    channel_order_check*.csv
    phase1_adjacency/                -- VLAAI montage (dtu_channel_montage.csv)
    phase1_adjacency_aadnet/         -- AADNet montage (aadnet_dtu_channel_montage.csv)
    phase2_gcn_only/
      PHASE2_REPORT.md               -- full Phase 2 write-up
      gcn_only_fold_accuracy.csv     -- raw 144-row per-fold results
      gcn_only_per_subject_summary.csv
      gcn_only_training_curves.png

notebooks/
  kaggle_train_stgcn_gcn_only.py/.ipynb        -- Phase 2 training
  kaggle_diag_stgcn_overfit_check.py/.ipynb    -- Phase 2 overfit sanity check
```
