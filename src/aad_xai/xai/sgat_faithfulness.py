"""Faithfulness tests for StimulusGAT explanations, across folds and in electrode space.

Levels 1-2 of the interpretability plan (sanity, faithfulness), for the four
explanation types the project requires:

- channel importance: occlusion, integrated gradients and attention, each
  scored by deleting (or keeping only) electrodes in rank order;
- edge importance: attention and attention x gradient, scored by removing the
  top-ranked graph edges from the attention mask;
- frequency-band contributions: attribution to the frequency encoder's
  band-power features, compared with the accuracy drop when each band is
  filtered out of the EEG.

Electrode space: Euclidean alignment maps each subject's EEG with W = R^{-1/2},
so a model "channel" mixes electrodes. ElectrodeSpaceScorer applies W (and the
per-window z-scoring the datasets use) inside the differentiable forward pass, so
deleting or attributing to an input channel means deleting or attributing to a
physical electrode. Graph nodes and edges stay in aligned space; W is the
symmetric whitening closest to the identity, and alignment_locality() reports
how close.

Integrated gradients: the model z-scores every window, so it is invariant to
the input's scale, and the usual all-zero baseline (a scaling path) carries no
information. The baseline here is a multivariate phase-randomised surrogate of
the same window: same spectrum and spatial covariance, timing relative to the
speech destroyed.

Rankings should be computed on one half of the test windows and evaluated on
the other (cross_fit_halves), so perturbation-derived rankings are not scored on
the very windows they were measured on.

All functions take eval-mode models; none changes model weights. Attention is
not assumed faithful: every ranking is compared with a random ranking.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr

from ..models.sgat import StimulusGAT
from ..models.sgat_frequency_encoder import DEFAULT_BANDS_HZ

Z_EPS = 1e-8  # same epsilon as the datasets' per-window z-scoring


# ============================================================================
#  Electrode-space scoring
# ============================================================================

class ElectrodeSpaceScorer(nn.Module):
    """Logits for raw (unaligned) EEG windows.

    x_raw (B, C, T) -> W x_raw -> per-window, per-channel z-score -> model.
    W : (C, C) alignment matrix of the subject the windows belong to, or None.
    """

    def __init__(self, model: StimulusGAT, W: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        self.register_buffer("W", None if W is None else torch.as_tensor(W, dtype=torch.float32))

    def prepare(self, x_raw: torch.Tensor) -> torch.Tensor:
        """(B, C, T) raw windows -> (B, C, T) model input."""
        x = x_raw if self.W is None else torch.einsum("ij,bjt->bit", self.W, x_raw)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, unbiased=False, keepdim=True)
        return (x - mu) / (sd + Z_EPS)

    def forward(self, x_raw: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        return self.model(self.prepare(x_raw), env)                     # (B, 2)


def margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Attended minus ignored logit, (B,). Positive = correct decision."""
    idx = torch.arange(len(y), device=logits.device)
    return logits[idx, y] - logits[idx, 1 - y]


@torch.no_grad()
def evaluate(scorer, x, env, y, batch: int = 256) -> tuple[float, float]:
    """(accuracy, mean margin) over all windows."""
    m = torch.cat([margin(scorer(x[i:i + batch], env[i:i + batch]), y[i:i + batch])
                   for i in range(0, len(x), batch)])
    return float((m > 0).float().mean()), float(m.mean())


def alignment_locality(W: np.ndarray, remove_common_mode: bool = True) -> float:
    """Median over channels of M_ii^2 / sum_j M_ij^2 (1 = no mixing), M = W P.

    remove_common_mode: P = I - 11^T / C. Average-referenced EEG has no
    common-mode component, and the alignment ridge inflates W along exactly that
    direction, so W alone overstates the mixing the data actually undergoes.
    """
    W = np.asarray(W, dtype=float)
    C = W.shape[0]
    M = W @ (np.eye(C) - np.ones((C, C)) / C) if remove_common_mode else W
    return float(np.median(np.diag(M) ** 2 / (M ** 2).sum(axis=1)))


def cross_fit_halves(trial_ids) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks splitting windows into two halves by whole trials (alternating
    in sorted trial order), so no trial contributes to both halves."""
    trial_ids = np.asarray(trial_ids)
    order = {t: i for i, t in enumerate(sorted(set(trial_ids.tolist())))}
    first = np.array([order[t] % 2 == 0 for t in trial_ids])
    return first, ~first


# ============================================================================
#  Channel (electrode) importance
# ============================================================================

def delete_electrodes(x: torch.Tensor, idx, mode: str = "zero", seed: int = 0) -> torch.Tensor:
    """Remove electrodes idx from raw windows x (B, C, T).

    mode "zero": set to zero. mode "permute": replace each window's electrode
    signal with the same electrode from another window (keeps its statistics,
    breaks its relation to this window's speech).
    """
    idx = list(idx)
    out = x.clone()
    if not idx:
        return out
    if mode == "zero":
        out[:, idx, :] = 0.0
    elif mode == "permute":
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(x), generator=g).to(x.device)
        if len(x) > 1:
            fixed = perm == torch.arange(len(x), device=x.device)
            perm[fixed] = (perm[fixed] + 1) % len(x)             # no window keeps its own signal
        out[:, idx, :] = x[perm][:, idx, :]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return out


@torch.no_grad()
def occlusion_scores(scorer, x, env, y, mode: str = "zero", batch: int = 256) -> np.ndarray:
    """(C,) mean margin drop when each electrode alone is removed."""
    _, base = evaluate(scorer, x, env, y, batch)
    return np.array([base - evaluate(scorer, delete_electrodes(x, [c], mode), env, y, batch)[1]
                     for c in range(x.shape[1])])


def phase_randomized(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Multivariate phase-randomised surrogate of x (B, C, T): the same random
    phases for every channel, so spectra and cross-channel covariance are kept."""
    g = torch.Generator().manual_seed(seed)
    X = torch.fft.rfft(x, dim=-1)                                     # (B, C, F)
    phase = torch.rand(x.shape[0], 1, X.shape[-1], generator=g) * 2 * torch.pi
    phase[..., 0] = 0.0                                                # keep the mean
    if x.shape[-1] % 2 == 0:
        phase[..., -1] = 0.0                                           # Nyquist bin stays real
    return torch.fft.irfft(X * torch.exp(1j * phase.to(x.device)), n=x.shape[-1], dim=-1)


def integrated_gradients(scorer, x, env, y, *, steps: int = 32, seed: int = 0,
                         batch: int = 32) -> torch.Tensor:
    """IG of the margin with a phase-randomised baseline. Returns (B, C, T)."""
    out = []
    for i in range(0, len(x), batch):
        xb, eb, yb = x[i:i + batch], env[i:i + batch], y[i:i + batch]
        base = phase_randomized(xb, seed + i)
        grads = torch.zeros_like(xb)
        for a in (torch.arange(steps, dtype=torch.float32) + 0.5) / steps:   # midpoint rule
            xa = (base + a * (xb - base)).detach().requires_grad_(True)
            with torch.enable_grad():
                m = margin(scorer(xa, eb), yb).sum()
                grads += torch.autograd.grad(m, xa)[0]
        out.append(((xb - base) * grads / steps).detach())
    return torch.cat(out)


def ig_channel_scores(scorer, x, env, y, **kw) -> np.ndarray:
    """(C,) mean over windows of the summed-over-time IG (signed: positive
    supports the correct decision)."""
    return integrated_gradients(scorer, x, env, y, **kw).sum(dim=-1).mean(dim=0).cpu().numpy()


@contextmanager
def _capture_attention(model: StimulusGAT):
    """Capture the attention tensor the GAT actually uses: (N*B, heads, C, C),
    candidate-major (index k*B + b)."""
    store = {}

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            store["attn"] = out[1]

    h = model.gat.register_forward_hook(hook)
    try:
        yield store
    finally:
        h.remove()


def _attended_attention(model, xp, env, y, with_grad: bool = False):
    """Attention of each window's attended candidate, (B, heads, C, C); and, if
    with_grad, d margin / d attention for the same entries."""
    with _capture_attention(model) as store, torch.set_grad_enabled(with_grad):
        logits, _ = model.score_candidates(xp, env, return_attention_weights=True)
        attn = store["attn"]                                           # (2B, heads, C, C)
        B = len(y)
        sel = y.to(attn.device) * B + torch.arange(B, device=attn.device)   # attended candidate rows
        if not with_grad:
            return attn[sel].detach(), None
        g = torch.autograd.grad(margin(logits, y).sum(), attn)[0]
        return attn[sel].detach(), g[sel].detach()


def attention_node_scores(scorer: ElectrodeSpaceScorer, x, env, y, batch: int = 128) -> np.ndarray:
    """(C,) attention received per node (column sum), attended candidate,
    mean over windows and heads. Node space (aligned channels)."""
    model = scorer.model
    acc = []
    for i in range(0, len(x), batch):
        xp = scorer.prepare(x[i:i + batch]).detach()
        attn, _ = _attended_attention(model, xp, env[i:i + batch], y[i:i + batch])
        acc.append(attn.mean(dim=1).sum(dim=1))                        # (b, C) received
    return torch.cat(acc).mean(dim=0).cpu().numpy()


@torch.no_grad()
def deletion_curve(scorer, x, env, y, ranking, ks, *, keep: bool = False, mode: str = "zero",
                   batch: int = 256) -> list[tuple[int, float, float]]:
    """[(k, accuracy, mean margin)] after removing the top-k electrodes of
    ranking (keep=False), or removing all but the top-k (keep=True)."""
    ranking = list(ranking)
    C = x.shape[1]
    out = []
    for k in ks:
        removed = ranking[:k] if not keep else ranking[k:]
        acc, mar = evaluate(scorer, delete_electrodes(x, removed[:C], mode), env, y, batch)
        out.append((int(k), acc, mar))
    return out


def random_rankings(n_items: int, n: int, seed: int = 0) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [rng.permutation(n_items).tolist() for _ in range(n)]


def curve_area(values) -> float:
    """Mean height of a curve sampled at its points (trapezoid, unit width)."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()) if len(v) < 2 else float((v.sum() - 0.5 * v[0] - 0.5 * v[-1]) / (len(v) - 1))


def ranking_from_scores(scores) -> list[int]:
    """Indices from most to least important (largest score first)."""
    return np.argsort(-np.asarray(scores), kind="stable").tolist()


# ============================================================================
#  Edge importance
# ============================================================================

def directed_edges(model: StimulusGAT) -> list[tuple[int, int]]:
    """Directed graph edges (i attends to j), self-loops excluded."""
    m = model.gat.adj_mask.clone().cpu()
    m.fill_diagonal_(False)
    return [tuple(e) for e in torch.nonzero(m).tolist()]


@contextmanager
def edges_removed(model: StimulusGAT, edges):
    """Temporarily drop directed edges from the attention mask (self-loops stay,
    so every node keeps at least one neighbour)."""
    original = model.gat.adj_mask.clone()
    try:
        for i, j in edges:
            if i != j:
                model.gat.adj_mask[i, j] = False
        yield model
    finally:
        model.gat.adj_mask.copy_(original)


def edge_scores(scorer: ElectrodeSpaceScorer, x, env, y, edges, *, method: str = "attention",
                batch: int = 64) -> np.ndarray:
    """(E,) edge importance for the attended candidate, mean over windows and heads.
    method "attention": attention weight; "attn_x_grad": attention x d margin / d attention."""
    model = scorer.model
    tot = None
    n = 0
    device = model.gat.adj_mask.device
    ii = torch.tensor([e[0] for e in edges], device=device)
    jj = torch.tensor([e[1] for e in edges], device=device)
    for k in range(0, len(x), batch):
        xp = scorer.prepare(x[k:k + batch]).detach()
        attn, grad = _attended_attention(model, xp, env[k:k + batch], y[k:k + batch],
                                         with_grad=(method == "attn_x_grad"))
        val = attn if method == "attention" else attn * grad            # (b, heads, C, C)
        s = val.mean(dim=1)[:, ii, jj].sum(dim=0)                        # (E,)
        tot = s if tot is None else tot + s
        n += len(xp)
    return (tot / n).cpu().numpy()


@torch.no_grad()
def edge_deletion_curve(scorer, x, env, y, edges, ranking, ks, batch: int = 256):
    """[(k, accuracy, mean margin)] after removing the top-k edges of ranking."""
    out = []
    for k in ks:
        with edges_removed(scorer.model, [edges[i] for i in list(ranking)[:k]]):
            acc, mar = evaluate(scorer, x, env, y, batch)
        out.append((int(k), acc, mar))
    return out


# ============================================================================
#  Frequency-band contributions
# ============================================================================

def band_feature_attribution(scorer: ElectrodeSpaceScorer, x, env, y, *, steps: int = 16,
                             batch: int = 64) -> np.ndarray:
    """(n_bands,) integrated gradients of the margin with respect to the frequency
    encoder's log band-power features, baseline = the batch-mean features,
    summed over electrodes and averaged over windows (signed)."""
    model = scorer.model
    enc = model.frequency_encoder
    n_bands = len(enc.bands_hz)
    C = enc.n_channels
    total = torch.zeros(n_bands)
    n = 0
    for k in range(0, len(x), batch):
        xp = scorer.prepare(x[k:k + batch]).detach()
        eb, yb = env[k:k + batch], y[k:k + batch]
        feat = enc._band_power(xp).detach()                            # (b, C, n_bands)
        base = feat.mean(dim=0, keepdim=True).expand_as(feat)
        grads = torch.zeros_like(feat)
        for a in (torch.arange(steps, dtype=torch.float32) + 0.5) / steps:
            fa = (base + a * (feat - base)).detach().requires_grad_(True)
            box = {"f": fa}

            def pre_hook(_mod, inp, box=box):
                b = inp[0].shape[0]
                f = box["f"]
                # the 2 candidates are batched candidate-major; band features repeat per candidate
                return (f.repeat(b // f.shape[0], 1, 1).reshape(b, C * n_bands, 1),)

            h = enc.proj.register_forward_pre_hook(pre_hook)
            try:
                with torch.enable_grad():
                    m = margin(model(xp, eb), yb).sum()
                    grads += torch.autograd.grad(m, fa)[0]
            finally:
                h.remove()
        attr = (feat - base) * grads / steps                            # (b, C, n_bands)
        total += attr.sum(dim=1).sum(dim=0).cpu()
        n += len(xp)
    return (total / n).numpy()


def band_stop(x: torch.Tensor, sfreq: float, band: tuple[float, float], order: int = 4) -> torch.Tensor:
    """Remove one band from raw windows (B, C, T) with a zero-phase band-stop filter."""
    lo, hi = band
    nyq = sfreq / 2.0
    sos = butter(order, [lo / nyq, min(hi, nyq - 0.5) / nyq], btype="bandstop", output="sos")
    y = sosfiltfilt(sos, x.detach().cpu().numpy(), axis=-1)
    return torch.as_tensor(np.ascontiguousarray(y), dtype=x.dtype, device=x.device)


def band_stop_drops(scorer, x, env, y, sfreq: float, bands=DEFAULT_BANDS_HZ, batch: int = 256):
    """[(band, accuracy drop, margin drop)] when each band is filtered out of all electrodes."""
    acc0, mar0 = evaluate(scorer, x, env, y, batch)
    out = []
    for band in bands:
        acc, mar = evaluate(scorer, band_stop(x, sfreq, band), env, y, batch)
        out.append((tuple(band), acc0 - acc, mar0 - mar))
    return out


# ============================================================================
#  Sanity check (cascading randomisation)
# ============================================================================

def cascade_sanity(model: StimulusGAT, attr_fn, seed: int = 0) -> dict[str, float]:
    """Adebayo et al. (2018) cascading randomisation, from the readout down.

    attr_fn(model) -> 1-D array of scores. After re-drawing every parameter of
    each top-level module in turn (cumulatively; Gaussian with the parameter's own
    scale, so modules without reset_parameters, such as the attention vectors, are
    randomised too), returns the Spearman correlation with the trained model's
    scores. A sound explanation decays towards 0 as the cascade proceeds.
    """
    g = torch.Generator().manual_seed(seed)
    m = copy.deepcopy(model).eval()
    original = np.asarray(attr_fn(m), dtype=float)
    out = {}
    for name, child in reversed(list(m.named_children())):
        with torch.no_grad():
            for p in child.parameters():
                scale = float(p.std()) if p.numel() > 1 and float(p.std()) > 0 else 1.0
                p.copy_((torch.randn(p.shape, generator=g) * scale).to(p.device, p.dtype))
        rho = spearmanr(original, np.asarray(attr_fn(m), dtype=float)).correlation
        out[name] = float(0.0 if np.isnan(rho) else rho)
    return out
