from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .sgat_temporal_encoder import PerChannelTemporalEncoder
from .sgat_frequency_encoder import PerChannelFrequencyEncoder
from .sgat_speech_encoder import SharedSpeechEncoder
from .sgat_film import TimeResolvedFiLM
from .sgat_gat_layer import CandidateConditionedGAT


class NodeFeatureFusion(nn.Module):
    """Combine per-electrode temporal + frequency features into one feature
    vector per electrode (architecture step 3).

    Per-electrode Linear (implemented as a grouped 1x1 Conv1d,
    ``groups=n_channels``) -- no cross-channel mixing at this stage; that is
    reserved for the graph-attention layer, which uses the physical electrode
    adjacency as its prior.

    Followed by a per-node LayerNorm (normalizing each electrode's own
    out_features vector independently, so it's unaffected by batch
    composition -- unlike BatchNorm, which this pipeline already has reason
    to be cautious about with heterogeneous real subjects in one batch).
    The raw linear projection alone has no normalization and no
    nonlinearity, so nothing bounds its output scale as ``proj``'s weights
    update during training; observed directly (Oct 2026 gradient/attention
    instrumentation diagnostic): this output's std grew unboundedly (0.84 ->
    20.5 over 20 epochs on real DTU data), which saturates the downstream
    GAT's attention softmax into a degenerate near-one-hot pattern
    (entropy collapsed to ~2% of its uniform-attention reference) with zero
    accuracy improvement -- this LayerNorm is the fix for that observed
    failure mode, not a speculative addition.
    """

    def __init__(self, n_channels: int, in_features: int, out_features: int):
        super().__init__()
        self.n_channels = n_channels
        self.out_features = out_features
        self.proj = nn.Conv1d(
            n_channels * in_features, n_channels * out_features,
            kernel_size=1, groups=n_channels,
        )
        self.norm = nn.LayerNorm(out_features)

    def forward(self, temporal: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        temporal : Tensor, shape (B, C, F_t)
        freq : Tensor, shape (B, C, F_f)

        Returns
        -------
        Tensor, shape (B, C, out_features)
        """
        B, C = temporal.shape[0], temporal.shape[1]
        h = torch.cat([temporal, freq], dim=-1)      # (B, C, F_t + F_f)
        h = h.reshape(B, -1, 1)                        # (B, C*(F_t+F_f), 1), electrode-major
        out = self.proj(h).squeeze(-1)                  # (B, C*out_features)
        out = out.view(B, self.n_channels, self.out_features)  # (B, C, out_features)
        return self.norm(out)  # per-node LayerNorm over the out_features axis


def _load_default_adjacency(montage_path: Path | None = None) -> torch.Tensor:
    """Build the default distance-based k-NN (k=6) electrode adjacency from
    ``stgcn/adjacency.py``, following the same ``sys.path`` import trick
    ``aadnet_external.py`` uses for its own vendored dependency -- ``stgcn/``
    has no ``__init__.py`` and is not part of the installed ``aad_xai``
    package.
    """
    repo_root = Path(__file__).resolve().parents[3]
    stgcn_dir = repo_root / "stgcn"
    if str(stgcn_dir) not in sys.path:
        sys.path.insert(0, str(stgcn_dir))
    import adjacency as stgcn_adjacency  # stgcn/adjacency.py

    path = montage_path or stgcn_adjacency.DEFAULT_MONTAGE_PATH
    montage = stgcn_adjacency.load_montage(path)
    A = stgcn_adjacency.build_adjacency_distance(montage, k=6)
    return torch.as_tensor(A, dtype=torch.float32)


class StimulusGAT(nn.Module):
    """Stimulus-conditioned graph-attention model for auditory attention
    decoding.

    Pipeline (see CLAUDE.md for the full 9-step spec)::

        EEG -> temporal encoder -> h (B, C, F_t, T) ---------------+
        Speech X -> shared speech encoder -> s_X (B, D_s, T) -> TimeResolvedFiLM(h, s_X)
                                                     -> time-pooled (B, C, F_t)
        EEG -> frequency encoder (B, C, F_f) ---------> NodeFeatureFusion -> node_X
        node_X -> CandidateConditionedGAT -> mean over nodes -> readout -> score_X
        logits = [score_A, score_B]

    Speech conditions the EEG features at every time step *before* they are
    averaged over time, so the node features (and hence the graph attention)
    can depend on EEG-speech temporal covariance. Conditioning after pooling
    cannot represent envelope tracking (see TimeResolvedFiLM).

    Both candidates go through one shared instance of every module, batched
    together along the batch axis, so BatchNorm statistics in the speech
    encoder are computed over both candidates at once (identically in
    training and evaluation).

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (electrodes / graph nodes).
    sfreq : float
        EEG sampling rate in Hz (needed by the frequency encoder).
    adjacency : Tensor | None
        Optional (n_channels, n_channels) electrode adjacency. If ``None``,
        the default distance-based k-NN (k=6) adjacency is built from
        ``config/dtu_channel_montage.csv`` via ``stgcn/adjacency.py``.
    speech_bands : int
        Envelope channels per candidate. 1 (default): one broadband envelope,
        ``env`` of shape (B, 2, T). K > 1: K sub-band envelopes per candidate,
        ``env`` of shape (B, 2, K, T), read jointly by the shared speech
        encoder's first convolution.
    film_pooling : {"cov", "corr"}
        How the FiLM layer pools the EEG-speech term over time (see
        TimeResolvedFiLM). "cov" (default) is the original model.
    """

    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        adjacency: torch.Tensor | None = None,
        temporal_features: int = 16,
        freq_features: int = 8,
        node_features: int = 24,
        speech_embed_dim: int = 16,
        gat_hidden: int = 32,
        gat_heads: int = 4,
        dropout: float = 0.3,
        speech_bands: int = 1,
        film_pooling: str = "cov",
    ):
        super().__init__()
        self.speech_bands = speech_bands
        if adjacency is None:
            adjacency = _load_default_adjacency()

        self.temporal_encoder = PerChannelTemporalEncoder(
            n_channels, out_features=temporal_features, dropout=dropout,
        )
        self.frequency_encoder = PerChannelFrequencyEncoder(
            n_channels, sfreq=sfreq, out_features=freq_features,
        )
        self.node_fusion = NodeFeatureFusion(
            n_channels, in_features=temporal_features + freq_features, out_features=node_features,
        )
        self.speech_encoder = SharedSpeechEncoder(
            embed_dim=speech_embed_dim, dropout=dropout, in_channels=speech_bands,
        )
        self.film = TimeResolvedFiLM(speech_dim=speech_embed_dim, n_features=temporal_features,
                                     pooling=film_pooling)
        self.gat = CandidateConditionedGAT(
            n_channels=n_channels, in_features=node_features,
            out_features=gat_hidden, adjacency=adjacency, n_heads=gat_heads, dropout=dropout,
        )
        self.readout_norm = nn.LayerNorm(gat_hidden)
        # No bias: it adds equally to both candidates' scores, so it cannot
        # affect predictions and gets exactly zero gradient under 2-class
        # softmax CE -- yet Adam amplifies rounding noise into steady drift.
        self.readout = nn.Linear(gat_hidden, 1, bias=False)

    def forward(
        self, eeg: torch.Tensor, env: torch.Tensor, return_attention_weights: bool = False,
    ):
        """
        Parameters
        ----------
        eeg : Tensor, shape (B, C, T)
        env : Tensor, shape (B, 2, T), or (B, 2, K, T) if speech_bands = K > 1
            -- env[:, 0] = candidate A, env[:, 1] = candidate B
        return_attention_weights : bool
            If True, also return a dict with each candidate's GAT attention
            weights, shape (B, heads, N, N), for interpretability.

        Returns
        -------
        logits : Tensor, shape (B, 2) -- [score_a, score_b]
        attn : dict[str, Tensor], only if return_attention_weights=True
        """
        if env.shape[1] != 2:
            raise ValueError(f"forward expects 2 candidates, got env shape {tuple(env.shape)}")
        out = self.score_candidates(eeg, env, return_attention_weights)
        if return_attention_weights:
            logits, attn = out
            return logits, {"attn_a": attn[0], "attn_b": attn[1]}
        return out

    def score_candidates(
        self, eeg: torch.Tensor, env: torch.Tensor, return_attention_weights: bool = False,
    ):
        """Score any number of candidate speech streams against one EEG window.

        Every candidate goes through the same shared modules, so with the
        model in eval mode a candidate's score does not depend on the other
        candidates. ``forward`` is this method with 2 candidates.

        Parameters
        ----------
        eeg : Tensor, shape (B, C, T)
        env : Tensor, shape (B, N, T), or (B, N, K, T) if speech_bands = K > 1

        Returns
        -------
        scores : Tensor, shape (B, N)
        attn : Tensor, shape (N, B, heads, C, C), only if return_attention_weights=True
        """
        B, _, T = eeg.shape
        N = env.shape[1]
        h = self.temporal_encoder(eeg)                 # (B, C, F_t, T)
        f_feat = self.frequency_encoder(eeg)            # (B, C, F_f)

        # All candidates in one batch: index k*B + b is candidate k of sample b.
        if self.speech_bands == 1 and env.dim() == 4 and env.shape[2] == 1:
            env = env.squeeze(2)                                               # (B, N, 1, T) -> (B, N, T)
        expected = (B, N, T) if self.speech_bands == 1 else (B, N, self.speech_bands, T)
        if tuple(env.shape) != expected:
            raise ValueError(f"env shape {tuple(env.shape)} != expected {expected}")
        speech = self.speech_encoder(env.transpose(0, 1).reshape(N * B, *env.shape[2:]))  # (NB, D_s, T)
        speech = speech.view(N, B, -1, T)                                      # (N, B, D_s, T)
        t_cond = torch.cat([self.film(h, speech[k]) for k in range(N)])        # (NB, C, F_t)

        node_feats = self.node_fusion(t_cond, f_feat.repeat(N, 1, 1))  # (NB, C, F_node)
        gat_out = self.gat(node_feats, return_attention_weights)
        g, attn = gat_out if return_attention_weights else (gat_out, None)  # (NB, C, F_out)

        score = self.readout(self.readout_norm(g.mean(dim=1))).squeeze(-1)  # (NB,)
        logits = score.view(N, B).t()                                        # (B, N)
        if return_attention_weights:
            return logits, attn.view(N, B, *attn.shape[1:])                  # (N, B, heads, C, C)
        return logits
