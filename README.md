# AAD-XAI — Leakage-Safe EEG Auditory Attention Decoding + Interpretability

A clean, runnable research pipeline for **EEG Auditory Attention Decoding (AAD)** with:

- **Leakage-safe evaluation** — subject-independent splits with strict control of shared stimulus segments and overlapping windows across train/val/test boundaries.
- **Preprocessing** — EEG bandpass 1–8 Hz (envelope-tracking band) + downsample to 64 Hz.
- **Decision-window sweep** — 0.5 s, 1 s, 2 s, 5 s, 10 s, 30 s. Overlap allowed only *within* a split.
- **Three model families:**
  - **TRF (ridge)** — EEG → envelope reconstruction + attended vs unattended correlation margin.
  - **AADNet (CNN)** — end-to-end `Conv1d` architecture (channels × time → binary label).
  - **ST-GCN (CNN+GCN hybrid)** — hand-written graph convolution (no `torch_geometric` dependency).
- **Interpretability validation ladder:**
  - Integrated Gradients (via Captum) + LRP stub (plug in zennit/innvestigate).
  - Layer-wise linear probes (held-out accuracy vs layer index).
  - Sanity checks: single-layer and cascading parameter randomisation (Adebayo et al., 2018).
  - Deletion/insertion faithfulness curves.
  - EEG-structured perturbations: channel groups, lag ranges, band-limited attenuation.
- **Reproducibility** — multi-seed training (≥3), config-driven runs, JSON logging of splits/preprocessing/params, unit tests for leakage checks.
- **Synthetic smoke-test** — runs the full pipeline without any real datasets.

---

## Quick Start

### 1. Create environment and install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Download datasets (optional)

See [scripts/DOWNLOAD_LINKS.md](scripts/DOWNLOAD_LINKS.md) for official dataset sources:

| Dataset | Source |
|---------|--------|
| KU Leuven AAD | Zenodo / SparrKULee |
| DTU EEG+audio | Zenodo (COCOHA) |

Place extracted data in `data_raw/` (or any path you configure).

### 3. Run the synthetic smoke-test (no data needed)

```bash
# Quick 2-epoch smoke-test with CNN
python -m aad_xai.train --synthetic --model cnn --epochs 2 --seeds 1

# TRF baseline on synthetic data
python -m aad_xai.train --synthetic --model trf --seeds 1

# ST-GCN on synthetic data
python -m aad_xai.train --synthetic --model stgcn --epochs 3 --seeds 1
```

### 4. Full training (multi-seed)

```bash
# Train CNN, 3 seeds, 1-second windows, 30 epochs
python -m aad_xai.train --synthetic --model cnn --window 1.0 --epochs 30 --seeds 3

# Train TRF baseline
python -m aad_xai.train --synthetic --model trf --seeds 3
```

### 5. Evaluation: accuracy vs window length + confidence intervals

```bash
# Sweep over multiple window lengths, plot accuracy with 95% CI
python -m aad_xai.evaluate --synthetic --model cnn --seeds 3 --epochs 10 \
    --windows 0.5,1,2,5,10
```

Output: `eval_results/accuracy_vs_window.png` + `eval_results/accuracy_vs_window.json`.

### 6. XAI: Integrated Gradients on a trained model

```bash
python -m aad_xai.evaluate --synthetic --model cnn --xai ig --seeds 1 --epochs 5
```

### 7. Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
src/aad_xai/
├── __init__.py              # package root
├── config.py                # RunConfig (JSON-serialisable dataclasses)
├── train.py                 # training entry-point (TRF + deep models)
├── evaluate.py              # evaluation: window sweep + XAI
├── data/
│   ├── base.py              # Trial dataclass + BaseDataset ABC
│   ├── synthetic_dataset.py # SyntheticDataset (smoke-test)
│   ├── kul_dataset.py       # KU Leuven dataset loader (TODO: schema)
│   ├── dtu_dataset.py       # DTU dataset loader (TODO: schema)
│   ├── preprocessing.py     # bandpass + resample via MNE
│   ├── speech_features.py   # speech envelope extraction
│   ├── splits.py            # subject-independent splits + leakage checks
│   ├── windowing.py         # decision windows + cross-split assertions
│   └── torch_dataset.py     # WindowedEEGDataset (PyTorch Dataset)
├── models/
│   ├── aadnet.py            # AADNet (3-block CNN)
│   ├── stgcn.py             # ST-GCN (CNN + hand-written GraphConv)
│   └── trf_baseline.py      # TRF ridge decoder
├── utils/
│   ├── logging.py           # save_json, get_run_dir, log_run_metadata
│   ├── metrics.py           # accuracy, bootstrap_ci
│   └── seed.py              # seed_everything (numpy + torch + python)
└── xai/
    ├── integrated_gradients.py  # IG via Captum
    ├── lrp.py                   # LRP stub (plug in zennit)
    ├── faithfulness.py          # deletion + insertion curves
    ├── probes.py                # linear probes (held-out) + probe_all_layers
    ├── sanity_checks.py         # randomize_parameters + cascading_randomization
    └── perturbations.py         # band, lag, channel-group perturbations

tests/
├── test_splits.py           # subject split + leakage tests
├── test_windowing.py        # window edge-cases + cross-split overlap
├── test_synthetic.py        # SyntheticDataset + full pipeline smoke-test
├── test_models.py           # forward pass + gradient flow tests
└── test_xai.py              # IG, faithfulness, perturbations, LRP stub
```

---

## Configuration

All settings live in `RunConfig` (see [src/aad_xai/config.py](src/aad_xai/config.py)):

```python
from aad_xai.config import RunConfig

cfg = RunConfig()          # all defaults
cfg.to_json("my_run.json") # persist
cfg = RunConfig.from_json("my_run.json")  # reload
```

Pass `--config my_run.json` to `train.py` for fully config-driven runs.

---

## Notes

- **Dataset loaders** (`kul_dataset.py`, `dtu_dataset.py`) contain parsing scaffolds with TODO markers — adapt field names to match your specific data release.
- **LRP** is a documented stub (`xai/lrp.py`). Plug in [zennit](https://github.com/chr5tphr/zennit) or [innvestigate](https://github.com/albermax/innvestigate) when ready.
- The **graph adjacency** in `stgcn.py` uses a default ring topology. Replace with a real EEG-montage distance matrix for production use.
- **Captum** is used for Integrated Gradients; `matplotlib` for plotting.

