# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Is CSP the bottleneck in the FAConformer SI port? A deep-network-free probe
#
# The SI port pilot (subject 0, all 8 SI folds) showed train loss pinned
# near the combined-loss chance floor (2*ln(2)~=1.386) on every fold, and
# several checkpoints landing at/near epoch 1 -- the same symptom shape
# that needed real debugging in ST-GCN Phase 2. An overfit sanity check on
# the full ~629K-parameter FAConformer network wouldn't distinguish
# "features are junk but memorized anyway" from "features are
# informative" -- a network that size can memorize noise regardless of
# feature quality. This notebook instead probes the CSP-transformed
# features DIRECTLY with a trivial linear classifier (logistic
# regression), with no deep network involved at all.
#
# **CSP fitting scope, confirmed from the actual call site**
# (`kaggle_faconformer_si_port_pilot.py`, `build_faconformer_si_loaders`):
# ```python
# csp = CSP(n_components=63, reg='ledoit_wolf', log=None, cov_est='concat',
#           transform_into='csp_space', norm_trace=True)
# train_data = csp.fit_transform(train_data, train_label_sq)
# ```
# `train_data` at this call is ALL windows from the pooled ~762-trial SI
# training set spanning the 17 non-held-out subjects -- not per-subject,
# no subject grouping. `cov_est='concat'` means MNE concatenates every
# window belonging to each class along the time axis and computes ONE
# 64x64 channel-covariance matrix per class from that pooled data -- a
# single cross-subject covariance estimate per class, exactly as MNE's own
# `cov_est='concat'` semantics describe.
#
# **Two cases, same probe:**
# 1. **SI case** (subject 0, fold 0) -- CSP fit on the pooled, cross-subject
#    training set described above.
# 2. **SD positive control** -- CSP fit on subject 0's OWN ~54-trial pool
#    (FAConformer's native chronological 90/10 split, already validated in
#    `kaggle_faconformer_pilot.py`), evaluated on that same subject's
#    held-out 10%. If this probe shows real signal here but not in the SI
#    case, that's clean evidence the probe itself works and the SI case's
#    null result is about cross-subject CSP fitting specifically, not a
#    broken diagnostic.
#
# No deep network, no GPU-bound training loop -- CSP fit + a linear probe
# is cheap. Report findings before touching either candidate fix.
#
# **Kaggle setup requirements:** Internet enabled, GPU accelerator (unused
# by this notebook's own computation, but kept for setup consistency),
# `dulanamanjitha/aad-xai-artifacts` dataset attached. No Kaggle Secret needed.

# %% [markdown]
# ## 1. Clone repository + install dependencies (AADNet only -- this
#    notebook never touches FAConformer's model or FFT-band code)

# %%
import os
import subprocess
import sys

REPO_DIR = "/kaggle/working/AAD_XAI"

if not os.path.exists(REPO_DIR):
    subprocess.run(
        ["git", "clone", "https://github.com/manjithadulana98/AAD_XAI.git", REPO_DIR],
        check=True,
    )
else:
    print(f"Repository already cloned at {REPO_DIR}")

os.chdir(REPO_DIR)

subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run(["pip", "install", "-q", "-e", "."], check=True)

for extra in ("src", "external/AADNet"):
    p = os.path.join(REPO_DIR, extra)
    if p not in sys.path:
        sys.path.insert(0, p)

print("Setup done.")

# %% [markdown]
# ## 2. Load config, bulk-load all 18 subjects (needed for the SI pool)

# %%
import yaml
import math
import numpy as np

PILOT_SUBJECT_ID = 0
SEED = 42
TIME_LEN = 2.0
VAL_FRACTION = 0.2
FS = 64
WINDOW_LENGTH = math.ceil(FS * TIME_LEN)   # 128
OVERLAP = 0.5
CSP_COMP = 63

np.random.seed(SEED)

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'."

BASE_CONFIG_PATH = os.path.join(REPO_DIR, "external", "AADNet", "config", "config_AADNet_SI_DTU_kaggle.yml")
with open(BASE_CONFIG_PATH, encoding="utf-8") as f:
    raw_config = yaml.safe_load(f)
raw_config["dataset"]["folder"] = os.path.join(DTU_ROOT, "eeg_new") + "/"
raw_config["dataset"]["stimuli_path"] = os.path.join(DTU_ROOT, "Audio")

from utils.config import Config
aadnet_config = Config.load_config(raw_config)

from aadnet.dataset import DTUDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP

import time as time_module
_load_start = time_module.time()
DTUDataset.loadData(aadnet_config, subject=None)
print(f"Bulk-loaded 18 subjects in {time_module.time() - _load_start:.1f}s")

# %% [markdown]
# ## 3. Shared helpers -- windowing (both conventions), CSP fit/transform,
#    log-variance features, the trivial linear probe

# %%
def windows_for_trials(eeg_data_list, event_data_list, window_size, overlap, eeg_channel=64):
    """SI convention -- all windows for a list of trials, no internal split
    (verbatim from kaggle_faconformer_si_port_pilot.py)."""
    stride = int(window_size * (1 - overlap))
    all_windows, all_labels = [], []
    for trial_eeg, label in zip(eeg_data_list, event_data_list):
        for i in range(0, trial_eeg.shape[0] - window_size + 1, stride):
            all_windows.append(trial_eeg[i:i + window_size, :])
            all_labels.append(label)
    windows = np.stack(all_windows, axis=0).reshape(-1, window_size, eeg_channel)
    labels = np.array(all_labels)
    return windows, labels


def sliding_window_sd(eeg_datas, labels, window_size, overlap, eeg_channel=64):
    """SD convention -- FAConformer's own chronological 90/10 split per
    trial (verbatim from kaggle_faconformer_pilot.py / their own
    data_loader.py)."""
    stride = int(window_size * (1 - overlap))
    train_eeg, test_eeg, train_label, test_label = [], [], [], []
    for m in range(len(labels)):
        eeg = eeg_datas[m]
        label = labels[m]
        windows, new_label = [], []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            windows.append(eeg[i:i + window_size, :])
            new_label.append(label)
        train_eeg.append(np.array(windows)[:int(len(windows) * 0.9)])
        test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
        train_label.append(np.array(new_label)[:int(len(windows) * 0.9)])
        test_label.append(np.array(new_label)[int(len(windows) * 0.9):])
    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, eeg_channel)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
    train_label = np.concatenate(train_label)
    test_label = np.concatenate(test_label)
    return train_eeg, test_eeg, train_label, test_label


def fit_csp(train_windows_tmajor, train_labels):
    """train_windows_tmajor: (N, window_size, 64) time-major, matching
    windows_for_trials'/sliding_window_sd's own output convention."""
    X = train_windows_tmajor.transpose(0, 2, 1)   # -> (N, 64, window_size), same as the pilot scripts
    print(f"  CSP.fit_transform input shape: {X.shape}  (N_windows, channels, time_samples)")
    csp = CSP(n_components=CSP_COMP, reg='ledoit_wolf', log=None, cov_est='concat',
              transform_into='csp_space', norm_trace=True)
    Xt = csp.fit_transform(X, train_labels)
    return csp, Xt


def transform_csp(csp, windows_tmajor):
    X = windows_tmajor.transpose(0, 2, 1)
    return csp.transform(X)


def logvar_features(csp_windows):
    """Canonical CSP feature extraction (CSP -> log-variance -> linear
    classifier), not an arbitrary flatten -- the standard, fairest test of
    whether CSP's spatial filtering itself carries discriminative signal.
    csp_windows: (N, n_components, window_size)."""
    var = csp_windows.var(axis=2)
    return np.log(var + 1e-10)


def run_probe(train_feat, train_label, *eval_sets):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(train_feat, train_label)
    accs = [clf.score(train_feat, train_label)]
    for feat, label in eval_sets:
        accs.append(clf.score(feat, label))
    return accs


# %% [markdown]
# ## 4. SI case -- subject 0, fold 0 (the same fold the pilot's red flags
#    were observed on)

# %%
crossSIData = DTUDataset.createSICrossValidation(PILOT_SUBJECT_ID, aadnet_config)
(tr_eeg, tr_aud, tr_label), (te_eeg, te_aud, te_label) = crossSIData[0]
print(f"SI fold 0: {len(tr_eeg)} train trials (pre-val-split), {len(te_eeg)} test trials")

tr_eeg2, va_eeg, tr_label2, va_label = train_test_split(
    tr_eeg, tr_label, test_size=VAL_FRACTION, random_state=PILOT_SUBJECT_ID
)
print(f"  after val split: {len(tr_eeg2)} train trials, {len(va_eeg)} valid trials")

si_train_eeg = [t.T.astype(np.float64) for t in tr_eeg2]
si_valid_eeg = [t.T.astype(np.float64) for t in va_eeg]
si_test_eeg = [t.T.astype(np.float64) for t in te_eeg]
si_train_label = [int(l) for l in tr_label2]
si_valid_label = [int(l) for l in va_label]
si_test_label = [int(l) for l in te_label]

si_train_w, si_train_l = windows_for_trials(si_train_eeg, si_train_label, WINDOW_LENGTH, OVERLAP)
si_valid_w, si_valid_l = windows_for_trials(si_valid_eeg, si_valid_label, WINDOW_LENGTH, OVERLAP)
si_test_w, si_test_l = windows_for_trials(si_test_eeg, si_test_label, WINDOW_LENGTH, OVERLAP)
print(f"SI window counts: train={si_train_w.shape} valid={si_valid_w.shape} test={si_test_w.shape}")

si_csp, si_train_csp = fit_csp(si_train_w, si_train_l)
si_valid_csp = transform_csp(si_csp, si_valid_w)
si_test_csp = transform_csp(si_csp, si_test_w)

si_train_feat = logvar_features(si_train_csp)
si_valid_feat = logvar_features(si_valid_csp)
si_test_feat = logvar_features(si_test_csp)

si_train_acc, si_valid_acc, si_test_acc = run_probe(
    si_train_feat, si_train_l, (si_valid_feat, si_valid_l), (si_test_feat, si_test_l)
)
print(f"SI PROBE  | train_acc={si_train_acc:.4f}  valid_acc={si_valid_acc:.4f}  test_acc={si_test_acc:.4f}")

# %% [markdown]
# ## 5. SD positive control -- subject 0's own native FAConformer pipeline

# %%
sd_s_data = DTUDataset.all_data[PILOT_SUBJECT_ID]   # already bulk-loaded; loadData(subject=X) would wrongly no-op here
sd_eeg_list = [trial["eeg"].T.astype(np.float64) for trial in sd_s_data]
sd_label_list = [int(trial["label"]) for trial in sd_s_data]
print(f"SD subject {PILOT_SUBJECT_ID}: {len(sd_eeg_list)} trials")

sd_train_w, sd_test_w, sd_train_l, sd_test_l = sliding_window_sd(sd_eeg_list, sd_label_list, WINDOW_LENGTH, OVERLAP)
print(f"SD window counts: train={sd_train_w.shape} test={sd_test_w.shape}")

sd_csp, sd_train_csp = fit_csp(sd_train_w, sd_train_l)
sd_test_csp = transform_csp(sd_csp, sd_test_w)

sd_train_feat = logvar_features(sd_train_csp)
sd_test_feat = logvar_features(sd_test_csp)

sd_train_acc, sd_test_acc = run_probe(sd_train_feat, sd_train_l, (sd_test_feat, sd_test_l))
print(f"SD PROBE  | train_acc={sd_train_acc:.4f}  test_acc={sd_test_acc:.4f}")

# %% [markdown]
# ## 6. Side-by-side report

# %%
import json
from pathlib import Path

OUT_DIR = Path("/kaggle/working/faconformer_csp_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary = {
    "si_fold0": {
        "train_shape": list(si_train_w.shape), "valid_shape": list(si_valid_w.shape), "test_shape": list(si_test_w.shape),
        "train_acc": si_train_acc, "valid_acc": si_valid_acc, "test_acc": si_test_acc,
    },
    "sd_subject0": {
        "train_shape": list(sd_train_w.shape), "test_shape": list(sd_test_w.shape),
        "train_acc": sd_train_acc, "test_acc": sd_test_acc,
    },
}
print(json.dumps(summary, indent=2))

with open(OUT_DIR / "csp_probe_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Written to {OUT_DIR}")
