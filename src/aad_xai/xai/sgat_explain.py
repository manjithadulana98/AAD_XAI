"""StimulusGAT-specific interpretability orchestrator.

Mirrors trf_explain.py's role: a model-specific orchestrator that reuses the
generic xai/ tools (faithfulness.py, sanity_checks.py, composite_stability.py)
plus this file's own channel/frequency-band/attention-summary functions,
rather than reimplementing statistics or perturbation primitives.

Scope: operates on ONE trained StimulusGAT checkpoint + one batch of its own
(eeg, env, labels) windows. StimulusGAT trains one independent model per LOSO
fold (not one shared group-level model like VLAAI) -- cross-fold aggregation
of any of this module's outputs is a deliberate non-goal here, left for a
future extension once there's a reason to compare importance rankings across
independently-trained checkpoints.

Per CLAUDE.md: attention weights are exposed (extract_attention/
summarize_attention) but explicitly NOT assumed faithful -- faithfulness_
comparison() is the concrete test of that, comparing attention-derived vs.
occlusion-derived channel rankings' deletion/insertion curves.
"""
from __future__ import annotations

import numpy as np
import torch

from ..models.sgat import StimulusGAT
from ..models.sgat_decision import StimulusGATDecisionWrapper
from .composite_stability import bootstrap_ci, fdr_correction, safe_spearman, sign_flip_p_value
from .faithfulness import deletion_curve, insertion_curve
from .frequency_bands import BANDS, ablate_band
from .perturbations import permute_channel_group
from .sanity_checks import cascading_randomization


# ============================================================================
#  Shared helper
# ============================================================================

def _p_attended(wrapper: StimulusGATDecisionWrapper, eeg: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """P(attended) per sample: softmax(logits)[b, labels[b]].

    eeg : (B, C, T); labels : (B,) int. Returns (B,).
    """
    logits = wrapper(eeg)  # (B, 2)
    probs = torch.softmax(logits, dim=1)
    return probs.gather(1, labels.view(-1, 1).long()).squeeze(1)


def _to_json_safe(obj):
    """Recursively convert numpy/torch scalars, arrays, and tensors to plain
    Python types, so a result dict round-trips through json.dumps."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return _to_json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ============================================================================
#  Channel importance (occlusion + permutation)
# ============================================================================

def _channel_importance(
    wrapper: StimulusGATDecisionWrapper,
    eeg: torch.Tensor,
    env: torch.Tensor,
    labels: torch.Tensor,
    perturb_fn,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    fdr_alpha: float = 0.05,
) -> list[dict]:
    """Shared machinery for occlusion/permutation channel importance.

    perturb_fn(eeg_np: (B,C,T), ch: int) -> eeg_np_perturbed: (B,C,T)
    """
    wrapper.eval()
    wrapper.set_env(env)
    with torch.no_grad():
        p_orig = _p_attended(wrapper, eeg, labels)  # (B,)

    eeg_np = eeg.detach().cpu().numpy()
    n_channels = eeg_np.shape[1]

    per_channel: list[dict] = []
    p_values: list[float] = []
    for ch in range(n_channels):
        eeg_pert_np = perturb_fn(eeg_np, ch)
        eeg_pert = torch.as_tensor(eeg_pert_np, dtype=eeg.dtype, device=eeg.device)
        with torch.no_grad():
            p_pert = _p_attended(wrapper, eeg_pert, labels)
        dp = (p_orig - p_pert).detach().cpu().numpy()  # (B,) per-window delta-P for this channel

        mean_dp, ci_lo, ci_hi = bootstrap_ci(dp, n_boot=n_boot, seed=seed)
        p_value = sign_flip_p_value(dp, seed=seed)

        per_channel.append({
            "channel": ch, "mean_dp": mean_dp, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": p_value,
        })
        p_values.append(p_value)

    fdr_p, fdr_sig = fdr_correction(np.asarray(p_values), alpha=fdr_alpha)
    for row, fp, fs in zip(per_channel, fdr_p, fdr_sig):
        row["fdr_p"] = float(fp)
        row["fdr_sig"] = bool(fs)
    return per_channel


def channel_occlusion_importance(
    wrapper: StimulusGATDecisionWrapper, eeg: torch.Tensor, env: torch.Tensor, labels: torch.Tensor,
    *, n_boot: int = 1000, seed: int = 42, fdr_alpha: float = 0.05,
) -> list[dict]:
    """Per-channel zero-occlusion delta-P(attended).

    eeg : (B,C,T), env : (B,2,T), labels : (B,) int in {0,1}.

    Does NOT reuse perturbations.remove_channel_group -- its (B,T,C)-only
    ndim==3 branch would silently zero time indices, not channels, on
    StimulusGAT's (B,C,T) convention. Zeros channels inline instead.

    Returns list[dict], one per channel: channel, mean_dp, ci_lo, ci_hi,
    p_value (sign-flip), fdr_p, fdr_sig (BH-FDR across channels).
    """
    def _occlude(eeg_np, ch):
        x = eeg_np.copy()
        x[:, ch, :] = 0.0
        return x

    return _channel_importance(wrapper, eeg, env, labels, _occlude, n_boot=n_boot, seed=seed, fdr_alpha=fdr_alpha)


def channel_permutation_importance(
    wrapper: StimulusGATDecisionWrapper, eeg: torch.Tensor, env: torch.Tensor, labels: torch.Tensor,
    *, n_boot: int = 1000, seed: int = 42, fdr_alpha: float = 0.05,
) -> list[dict]:
    """Per-channel permutation-across-batch delta-P(attended), via
    perturbations.permute_channel_group(..., channel_axis=1). Same return
    contract as channel_occlusion_importance."""
    def _permute(eeg_np, ch):
        return permute_channel_group(eeg_np, [ch], channel_axis=1, seed=seed)

    return _channel_importance(wrapper, eeg, env, labels, _permute, n_boot=n_boot, seed=seed, fdr_alpha=fdr_alpha)


# ============================================================================
#  Edge / attention importance
# ============================================================================

def extract_attention(model: StimulusGAT, eeg: torch.Tensor, env: torch.Tensor) -> dict:
    """Raw attention weights straight from the model's own forward pass --
    no new extraction machinery.

    eeg : (B,C,T), env : (B,2,T).
    Returns {"attn_a": (B,heads,N,N), "attn_b": (B,heads,N,N)}.
    """
    model.eval()
    with torch.no_grad():
        _, attn = model(eeg, env, return_attention_weights=True)
    return attn


def summarize_attention(attn: torch.Tensor, adj_mask: torch.Tensor) -> dict:
    """Summarize one candidate's attention tensor into edge- and node-level
    importance.

    attn : (B, heads, N, N) -- attn[b,h,i,j] = how much query node i attends
        to neighbor node j.
    adj_mask : (N, N) bool -- model.gat.adj_mask. Masked (non-adjacent)
        entries carry exactly zero attention mass by construction (softmax
        restricted to neighbors); excluded here, not treated as "zero
        importance" evidence.

    Returns
    -------
    dict with:
      "edge_importance": (N,N) ndarray -- mean over batch+heads, exact zero
          outside adj_mask.
      "node_received": (N,) ndarray -- total incoming attention per node
          (column sum over query index i). NOTE: row-sum ("attention given")
          is NOT informative here -- softmax normalizes every row to 1 -- only
          column-sum varies meaningfully node to node.
      "_caveat": str -- attention-based evidence only, not perturbation-
          validated; see faithfulness_comparison().
    """
    mean_attn = attn.mean(dim=(0, 1))  # (N, N)
    mask = adj_mask.to(mean_attn.dtype)
    edge_importance = (mean_attn * mask).detach().cpu().numpy()
    node_received = edge_importance.sum(axis=0)  # (N,)
    return {
        "edge_importance": edge_importance,
        "node_received": node_received,
        "_caveat": (
            "Attention-based evidence only -- NOT validated by ablation/perturbation. "
            "Do not treat as a faithfulness claim on its own; see faithfulness_comparison()."
        ),
    }


# ============================================================================
#  Frequency-band contribution
# ============================================================================

def channel_band_contribution(
    wrapper: StimulusGATDecisionWrapper, eeg: torch.Tensor, env: torch.Tensor, labels: torch.Tensor,
    sfreq: float, channel_idx: list[int], *, n_boot: int = 1000, seed: int = 42, fdr_alpha: float = 0.05,
) -> dict:
    """Per-(channel, band) delta-P(attended) via subtractive band ablation.

    eeg : (B,C,T), env : (B,2,T), labels : (B,) int in {0,1}.
    Returns {channel: {band_name: {"mean_dp","ci_lo","ci_hi","p_value","fdr_p","fdr_sig"}}}
    -- FDR applied across all (channel,band) pairs jointly, not per channel alone.
    """
    wrapper.eval()
    wrapper.set_env(env)
    with torch.no_grad():
        p_orig = _p_attended(wrapper, eeg, labels)  # (B,)

    eeg_np = eeg.detach().cpu().numpy()

    pairs: list[tuple[int, str]] = []
    p_values: list[float] = []
    stats: dict[int, dict[str, dict]] = {ch: {} for ch in channel_idx}

    for ch in channel_idx:
        for band_name, band in BANDS.items():
            eeg_ablated_np = ablate_band(
                eeg_np, sfreq, band, channel_idx=[ch], time_axis=-1, channel_axis=1,
            )
            eeg_ablated = torch.as_tensor(eeg_ablated_np, dtype=eeg.dtype, device=eeg.device)
            with torch.no_grad():
                p_ablated = _p_attended(wrapper, eeg_ablated, labels)
            dp = (p_orig - p_ablated).detach().cpu().numpy()  # (B,)

            mean_dp, ci_lo, ci_hi = bootstrap_ci(dp, n_boot=n_boot, seed=seed)
            p_value = sign_flip_p_value(dp, seed=seed)

            stats[ch][band_name] = {"mean_dp": mean_dp, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": p_value}
            pairs.append((ch, band_name))
            p_values.append(p_value)

    fdr_p, fdr_sig = fdr_correction(np.asarray(p_values), alpha=fdr_alpha)
    for (ch, band_name), fp, fs in zip(pairs, fdr_p, fdr_sig):
        stats[ch][band_name]["fdr_p"] = float(fp)
        stats[ch][band_name]["fdr_sig"] = bool(fs)

    return stats


# ============================================================================
#  Faithfulness (deletion/insertion), comparing rankings
# ============================================================================

def faithfulness_from_channel_ranking(
    wrapper: StimulusGATDecisionWrapper, eeg: torch.Tensor, env: torch.Tensor, labels: torch.Tensor,
    channel_ranking: list[int], *, steps: int = 20,
) -> dict:
    """Deletion/insertion curves for one channel ranking.

    channel_ranking : list[int] -- channel indices, most-to-least important.
    Builds a per-sample (B,C,T) importance tensor with a channel-constant
    score (broadcast across each channel's full time slice), so
    deletion_curve/insertion_curve's element-wise top-|importance| masking
    deletes/reveals one whole channel at a time, in ranking order.

    wrapper.set_env(env) is called here -- caller does not need to call it first.
    Returns {"deletion": list[float], "insertion": list[float]}.
    """
    wrapper.eval()
    wrapper.set_env(env)

    n_channels = eeg.shape[1]
    scores = torch.zeros(n_channels)
    for rank, ch in enumerate(channel_ranking):
        scores[ch] = float(n_channels - rank)  # higher score = more important
    importance = scores.view(1, n_channels, 1).expand_as(eeg).clone()  # (B,C,T); .clone() for contiguity

    deletion = deletion_curve(wrapper, eeg, labels, importance, steps=steps)
    insertion = insertion_curve(wrapper, eeg, labels, importance, steps=steps)
    return {"deletion": deletion, "insertion": insertion}


def faithfulness_comparison(
    wrapper: StimulusGATDecisionWrapper, eeg: torch.Tensor, env: torch.Tensor, labels: torch.Tensor,
    attention_ranking: list[int], occlusion_ranking: list[int], *, steps: int = 20, seed: int = 42,
) -> dict:
    """Runs faithfulness_from_channel_ranking for the attention-derived
    ranking, the occlusion-derived ranking, AND a random-ranking control,
    side by side -- the concrete test of CLAUDE.md's "faithfulness will
    later be tested" instruction. If occlusion produces a steeper deletion
    curve than attention, that is the reportable finding, not a failure --
    attention is never assumed faithful going in.
    """
    n_channels = eeg.shape[1]
    rng = np.random.RandomState(seed)
    random_ranking = list(rng.permutation(n_channels))

    return {
        "attention": faithfulness_from_channel_ranking(wrapper, eeg, env, labels, attention_ranking, steps=steps),
        "occlusion": faithfulness_from_channel_ranking(wrapper, eeg, env, labels, occlusion_ranking, steps=steps),
        "random": faithfulness_from_channel_ranking(wrapper, eeg, env, labels, random_ranking, steps=steps),
    }


# ============================================================================
#  Sanity check (cascading randomization)
# ============================================================================

def sanity_check_cascading_randomization(
    model: StimulusGAT, eeg: torch.Tensor, env: torch.Tensor, *,
    importance_fn: str = "occlusion", labels: torch.Tensor | None = None,
) -> dict:
    """Adebayo et al. cascading-randomization sanity check, applied to
    StimulusGAT's top-level children (temporal_encoder, frequency_encoder,
    node_fusion, speech_encoder, gat, readout).

    importance_fn : "occlusion" (requires labels) or "attention" (uses
        candidate A's attention summary; model.gat.adj_mask is a fixed
        buffer, unaffected by parameter randomization, so it stays valid
        across every cascade depth).

    A faithful attribution should DIVERGE (rank-correlation -> decay toward
    0) from the original as more layers are randomized; if it doesn't decay,
    that attribution may be insensitive to the model's learned weights.

    Returns {"raw": {layer_name: (C,) ndarray, "__original__": ...},
             "rho_by_layer": {layer_name: {"rho": float, "p_value": float}}}.
    """
    if importance_fn not in {"occlusion", "attention"}:
        raise ValueError("importance_fn must be 'occlusion' or 'attention'")
    if importance_fn == "occlusion" and labels is None:
        raise ValueError("labels is required when importance_fn='occlusion'")

    def attr_fn(m: StimulusGAT, x: torch.Tensor) -> torch.Tensor:
        if importance_fn == "attention":
            _, attn = m(x, env, return_attention_weights=True)
            summary = summarize_attention(attn["attn_a"], m.gat.adj_mask)
            return torch.as_tensor(summary["node_received"])

        wrapper = StimulusGATDecisionWrapper(m)
        wrapper.set_env(env)
        p_orig = _p_attended(wrapper, x, labels)  # (B,)
        n_channels = x.shape[1]
        dp = torch.zeros(n_channels)
        x_np = x.detach().cpu().numpy()
        for ch in range(n_channels):
            x_occ_np = x_np.copy()
            x_occ_np[:, ch, :] = 0.0
            x_occ = torch.as_tensor(x_occ_np, dtype=x.dtype, device=x.device)
            p_occ = _p_attended(wrapper, x_occ, labels)
            dp[ch] = (p_orig - p_occ).mean()
        return dp

    results = cascading_randomization(model, attr_fn, eeg)

    original = results["__original__"]
    rho_by_layer = {}
    for name, arr in results.items():
        if name == "__original__":
            continue
        rho, p = safe_spearman(original, arr)
        rho_by_layer[name] = {"rho": rho, "p_value": p}

    return {"raw": results, "rho_by_layer": rho_by_layer}


# ============================================================================
#  Top-level orchestrator
# ============================================================================

def run_sgat_explain(
    model: StimulusGAT, eeg: torch.Tensor, env: torch.Tensor, labels: torch.Tensor, *,
    sfreq: float, n_boot: int = 1000, seed: int = 42, fdr_alpha: float = 0.05,
    top_k_freq_channels: int = 10, faithfulness_steps: int = 20,
) -> dict:
    """Given ONE trained StimulusGAT + one batch of its own (eeg, env, labels)
    windows: runs channel occlusion + permutation importance, attention
    extraction + summary, frequency-band contribution (on the top-K channels
    by |occlusion mean_dp|), a faithfulness comparison (occlusion vs.
    attention vs. random channel ranking), and the cascading-randomization
    sanity check (occlusion-based).

    eeg : (B,C,T), env : (B,2,T), labels : (B,) int in {0,1}.
    Returns one flat, JSON-serializable dict (plain floats/lists/dicts only).
    """
    model.eval()
    wrapper = StimulusGATDecisionWrapper(model)
    wrapper.set_env(env)

    occlusion = channel_occlusion_importance(wrapper, eeg, env, labels, n_boot=n_boot, seed=seed, fdr_alpha=fdr_alpha)
    permutation = channel_permutation_importance(wrapper, eeg, env, labels, n_boot=n_boot, seed=seed, fdr_alpha=fdr_alpha)

    attn = extract_attention(model, eeg, env)
    attn_summary_a = summarize_attention(attn["attn_a"], model.gat.adj_mask)
    attn_summary_b = summarize_attention(attn["attn_b"], model.gat.adj_mask)

    occlusion_sorted = sorted(occlusion, key=lambda r: abs(r["mean_dp"]), reverse=True)
    top_channels = [r["channel"] for r in occlusion_sorted[:top_k_freq_channels]]
    band_contribution = channel_band_contribution(
        wrapper, eeg, env, labels, sfreq, top_channels, n_boot=n_boot, seed=seed, fdr_alpha=fdr_alpha,
    )

    occlusion_ranking = [r["channel"] for r in occlusion_sorted]
    attention_ranking = list(np.argsort(-attn_summary_a["node_received"]))
    faithfulness = faithfulness_comparison(
        wrapper, eeg, env, labels, attention_ranking, occlusion_ranking, steps=faithfulness_steps, seed=seed,
    )

    sanity = sanity_check_cascading_randomization(model, eeg, env, importance_fn="occlusion", labels=labels)

    result = {
        "channel_occlusion_importance": occlusion,
        "channel_permutation_importance": permutation,
        "attention_summary_a": attn_summary_a,
        "attention_summary_b": attn_summary_b,
        "frequency_band_contribution": band_contribution,
        "faithfulness_comparison": faithfulness,
        "sanity_check": sanity,
    }
    return _to_json_safe(result)
