"""Quick debug: time each step for 1 subject, 1 fold."""
import sys, time, numpy as np
sys.path.insert(0, "src")
from aad_xai.data.vlaai_dataset import load_dtu_trials, window_data
from aad_xai.models.trf_baseline import lag_matrix
from sklearn.linear_model import Ridge

FS = 64
MAX_TRAIN = 10000
ALPHAS = [0.1, 1, 10, 100, 1000, 10000, 100000]

t0 = time.time()
trials = load_dtu_trials("external/vlaai/evaluation_datasets/DTU", subjects=["S1"])
print(f"1. Load data: {time.time()-t0:.2f}s")

trial_list = trials["S1"]
print(f"   S1 has {len(trial_list)} trials")

t0 = time.time()
trial_data = []
for trial in trial_list:
    eeg = trial["eeg"].astype(np.float32)
    env = trial["envelope"].astype(np.float32)
    eeg = (eeg - eeg.mean(0, keepdims=True)) / (eeg.std(0, keepdims=True) + 1e-8)
    env = (env - env.mean(0, keepdims=True)) / (env.std(0, keepdims=True) + 1e-8)
    env_unatt = np.roll(env, env.shape[0] // 2, axis=0)
    eeg_w = window_data(eeg, 320, 64)
    env_w = window_data(env, 320, 64)
    env_unatt_w = window_data(env_unatt, 320, 64)
    n = min(eeg_w.shape[0], env_w.shape[0], env_unatt_w.shape[0])
    trial_data.append({
        "eeg_raw": eeg, "env_raw": env.ravel(),
        "eeg_w": eeg_w[:n], "env_att_w": env_w[:n], "env_unatt_w": env_unatt_w[:n],
    })
    print(f"   trial: eeg_raw={eeg.shape}, windows={n}")
print(f"2. Prep trials: {time.time()-t0:.2f}s")

# 1 fold: train on trials 0-7, test on 8-9
train_set = list(range(8))
test_set = [8, 9]

t0 = time.time()
eeg_tr = np.concatenate([trial_data[t]["eeg_raw"] for t in train_set], axis=0)
env_tr = np.concatenate([trial_data[t]["env_raw"] for t in train_set], axis=0)
print(f"3. Concat train: {eeg_tr.shape}, {time.time()-t0:.2f}s")

t0 = time.time()
eeg_c = eeg_tr[:MAX_TRAIN].T
env_c = env_tr[:MAX_TRAIN]
lags = np.arange(0, int(0.5*FS)+1)
X = lag_matrix(eeg_c, lags)
print(f"4. Lag matrix: {X.shape}, {time.time()-t0:.2f}s")

t0 = time.time()
Xm, Xs = X.mean(0), X.std(0)
Xs[Xs < 1e-8] = 1
X = (X-Xm)/Xs
y = env_c[:X.shape[0]]
ym, ys = y.mean(), y.std() or 1
y = (y - ym)/ys

n_tr = int(X.shape[0]*0.8)
best_a, best_s = 100, -99
for a in ALPHAS:
    m = Ridge(alpha=a, solver="cholesky", fit_intercept=True)
    m.fit(X[:n_tr], y[:n_tr])
    p = m.predict(X[n_tr:])
    r = np.corrcoef(p, y[n_tr:])[0, 1] if p.std() > 1e-12 else 0
    if np.isfinite(r) and r > best_s:
        best_s, best_a = r, a

model = Ridge(alpha=best_a, solver="cholesky", fit_intercept=True)
model.fit(X, y)
print(f"5. Fit+tune: {time.time()-t0:.2f}s, alpha={best_a}")

# Test
t0 = time.time()
test_eeg = np.concatenate([trial_data[t]["eeg_w"] for t in test_set], axis=0)
test_att = np.concatenate([trial_data[t]["env_att_w"] for t in test_set], axis=0)
test_unatt = np.concatenate([trial_data[t]["env_unatt_w"] for t in test_set], axis=0)
print(f"6. Concat test windows: {test_eeg.shape}, {time.time()-t0:.2f}s")

t0 = time.time()
n_correct = 0
for i in range(len(test_eeg)):
    Xi = lag_matrix(test_eeg[i].T.astype(np.float32), lags)
    Xi = (Xi - Xm)/Xs
    pred = model.predict(Xi) * ys + ym
    att_i = test_att[i][:, 0]
    unatt_i = test_unatt[i][:, 0]
    n = min(len(pred), len(att_i))
    ra = np.corrcoef(pred[:n], att_i[:n])[0, 1]
    ru = np.corrcoef(pred[:n], unatt_i[:n])[0, 1]
    if (np.isfinite(ra) and np.isfinite(ru)) and ra > ru:
        n_correct += 1
acc = n_correct / len(test_eeg)
print(f"7. Evaluate {len(test_eeg)} windows: {time.time()-t0:.2f}s, acc={acc:.1%}")
