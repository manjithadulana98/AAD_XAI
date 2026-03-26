from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge


def lag_matrix(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Create lagged design matrix for multichannel EEG.

    x: (n_channels, n_times)
    returns: (n_times, n_channels * n_lags)  -- float32 to save memory.
    """
    n_ch, n_t = x.shape
    x = np.asarray(x, dtype=np.float32)
    X = np.zeros((n_t, n_ch * len(lags)), dtype=np.float32)
    for li, lag in enumerate(lags):
        sl = slice(li * n_ch, (li + 1) * n_ch)
        if lag == 0:
            X[:, sl] = x.T
        elif lag > 0:
            # Feature at time t uses EEG at t + lag (EEG typically lags the stimulus)
            X[: n_t - lag, sl] = x[:, lag:].T
        else:
            # lag < 0 uses EEG at t + lag (i.e., past EEG)
            X[-lag:, sl] = x[:, : n_t + lag].T
    return X


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns 0.0 instead of NaN for constant signals."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class TRFDecoder:
    """Ridge TRF decoder: EEG -> stimulus envelope (attended).

    Fits a multichannel temporal response function with lagged regression.
    Standardises EEG features and target envelopes **in-place** to avoid
    doubling memory when the lag matrix is large.
    """

    def __init__(self, tmin_s: float = 0.0, tmax_s: float = 0.5, alpha: float = 100.0):
        self.tmin_s = float(tmin_s)
        self.tmax_s = float(tmax_s)
        self.alpha = float(alpha)
        self.model = Ridge(alpha=self.alpha, fit_intercept=True, solver="lsqr", copy_X=False)
        self.best_alpha_: float | None = None
        self.lags_: np.ndarray | None = None
        self.sfreq_: float | None = None
        # Store scaling stats for transform / inverse_transform
        self._X_mean: np.ndarray | None = None
        self._X_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _z_inplace(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Z-score columns of *X* in-place. Returns (mean, std).

        Important: NumPy's default `X.std(axis=0)` materializes large temporaries
        for big design matrices. We compute mean/variance in streaming chunks to
        avoid allocating an extra matrix the size of X.
        """
        n_rows, n_cols = X.shape
        if n_rows == 0:
            mu = np.zeros((n_cols,), dtype=np.float32)
            std = np.ones((n_cols,), dtype=np.float32)
            return mu, std

        sum_ = np.zeros((n_cols,), dtype=np.float64)
        sumsq = np.zeros((n_cols,), dtype=np.float64)

        chunk = 4096
        for start in range(0, n_rows, chunk):
            end = min(start + chunk, n_rows)
            c = X[start:end]
            sum_ += c.sum(axis=0, dtype=np.float64)
            sumsq += (c * c).sum(axis=0, dtype=np.float64)

        mu64 = sum_ / float(n_rows)
        var64 = sumsq / float(n_rows) - mu64 * mu64
        var64[var64 < 1e-12] = 1.0
        std64 = np.sqrt(var64)

        mu = mu64.astype(np.float32, copy=False)
        std = std64.astype(np.float32, copy=False)

        X -= mu
        X /= std
        return mu, std

    # ------------------------------------------------------------------ #
    def fit(self, eeg: np.ndarray, env: np.ndarray, sfreq: float) -> "TRFDecoder":
        """Fit the TRF decoder.

        Parameters
        ----------
        eeg : (n_channels, n_times) -- concatenated training EEG
        env : (n_times,) -- concatenated attended envelopes
        sfreq : sampling frequency in Hz
        """
        eeg = np.asarray(eeg, dtype=np.float32)
        env = np.asarray(env, dtype=np.float32).reshape(-1)
        self.sfreq_ = float(sfreq)

        lags = np.arange(
            int(round(self.tmin_s * sfreq)),
            int(round(self.tmax_s * sfreq)) + 1,
        )
        self.lags_ = lags
        X = lag_matrix(eeg, lags)
        y = env[: X.shape[0]]

        # Standardise in-place (no copy)
        self._X_mean, self._X_std = self._z_inplace(X)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        y = (y - self._y_mean) / self._y_std

        self.model = Ridge(alpha=self.alpha, fit_intercept=True, solver="lsqr", copy_X=False)
        self.model.fit(X, y)
        self.best_alpha_ = self.alpha
        return self

    def fit_select_alpha(
        self,
        eeg_train: np.ndarray,
        env_train: np.ndarray,
        eeg_val: np.ndarray,
        env_val: np.ndarray,
        sfreq: float,
        *,
        alphas: np.ndarray | None = None,
    ) -> "TRFDecoder":
        """Fit TRF and select ridge alpha using a held-out validation set.

        Selection criterion: Pearson correlation between predicted and true
        attended envelope on the validation set.

        This is closer to common AAD baselines that tune the ridge parameter
        (often via CV); here we use the fold's provided validation split.
        """
        eeg_train = np.asarray(eeg_train, dtype=np.float32)
        env_train = np.asarray(env_train, dtype=np.float32).reshape(-1)
        eeg_val = np.asarray(eeg_val, dtype=np.float32)
        env_val = np.asarray(env_val, dtype=np.float32).reshape(-1)
        self.sfreq_ = float(sfreq)

        if alphas is None:
            alphas = np.logspace(-7, 7, 15, dtype=np.float64)
        else:
            alphas = np.asarray(alphas, dtype=np.float64).reshape(-1)
            if alphas.size == 0:
                raise ValueError("alphas must be non-empty")

        lags = np.arange(
            int(round(self.tmin_s * sfreq)),
            int(round(self.tmax_s * sfreq)) + 1,
        )
        self.lags_ = lags

        Xtr = lag_matrix(eeg_train, lags)
        ytr = env_train[: Xtr.shape[0]]
        Xva = lag_matrix(eeg_val, lags)
        yva = env_val[: Xva.shape[0]]

        # Standardise using TRAIN stats
        self._X_mean, self._X_std = self._z_inplace(Xtr)
        Xva -= self._X_mean
        Xva /= self._X_std

        self._y_mean = float(ytr.mean())
        self._y_std = float(ytr.std()) or 1.0
        ytr_sc = (ytr - self._y_mean) / self._y_std
        yva_sc = (yva - self._y_mean) / self._y_std

        best_alpha = float(alphas[0])
        best_score = -np.inf
        best_model: Ridge | None = None

        for a in alphas:
            mdl = Ridge(alpha=float(a), fit_intercept=True, solver="lsqr", copy_X=False)
            mdl.fit(Xtr, ytr_sc)
            pred = mdl.predict(Xva)
            score = _safe_corrcoef(pred, yva_sc)
            if score > best_score:
                best_score = score
                best_alpha = float(a)
                best_model = mdl

        assert best_model is not None
        self.alpha = best_alpha
        self.model = best_model
        self.best_alpha_ = best_alpha
        return self

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        """Predict envelope from EEG."""
        assert self.lags_ is not None and self.sfreq_ is not None
        X = lag_matrix(np.asarray(eeg, dtype=np.float32), self.lags_)
        # Apply same scaling in-place
        X -= self._X_mean
        X /= self._X_std
        y_sc = self.model.predict(X)
        return y_sc * self._y_std + self._y_mean

