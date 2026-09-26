from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .sgat_temporal_encoder import PerChannelTemporalEncoder
from .sgat_frequency_encoder import PerChannelFrequencyEncoder
from .sgat_speech_encoder import SharedSpeechEncoder
from .sgat_gat_layer import CandidateConditionedGAT


class NodeFeatureFusion(nn.Module):
    """Combine per-electrode temporal + frequency features into one feature
    vector per electrode (architecture step 3).

    Per-electrode Linear (implemented as a grouped 1x1 Conv1d,
    ``groups=n_channels``) -- no cross-channel mixing at this stage; that is
    reserved for the graph-attention layer, which uses the physical electrode
    adjacency as its prior.
    """

    def __init__(self, n_channels: int, in_features: int, out_features: int):
        super().__init__()
        self.n_channels = n_channels
        self.out_features = out_features
        self.proj = nn.Conv1d(
            n_channels * in_features, n_channels * out_features,
            kernel_size=1, groups=n_channels,
        )

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
        return out.view(B, self.n_channels, self.out_features)  # (B, C, out_features)


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

        EEG -> temporal encoder  -+
            -> frequency encoder -+-> NodeFeatureFusion -> per-electrode node features
        Speech A -> shared speech encoder -> CandidateConditionedGAT(node_feats, speech_a) -> score_a
        Speech B -> shared speech encoder -> CandidateConditionedGAT(node_feats, speech_b) -> score_b
        logits = [score_a, score_b]

    The speech encoder and the GAT are each a *single* module instance,
    invoked once per candidate -- weight sharing comes from Python-level
    reuse of the same submodule, not from any special code.

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
    ):
        super().__init__()
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
        # Single shared instances -- invoked twice below, once per candidate.
        self.speech_encoder = SharedSpeechEncoder(embed_dim=speech_embed_dim, dropout=dropout)
        self.gat = CandidateConditionedGAT(
            n_channels=n_channels, in_features=node_features, speech_dim=speech_embed_dim,
            out_features=gat_hidden, adjacency=adjacency, n_heads=gat_heads, dropout=dropout,
        )
        self.readout = nn.Linear(gat_hidden, 1)

    def _score_candidate(
        self, node_feats: torch.Tensor, env_candidate: torch.Tensor, return_attention_weights: bool,
    ):
        speech_embed = self.speech_encoder(env_candidate)  # (B, D_s)
        gat_out = self.gat(node_feats, speech_embed, return_attention_weights)
        if return_attention_weights:
            g, attn = gat_out                                # (B, N, F_out), (B, heads, N, N)
        else:
            g, attn = gat_out, None
        score = self.readout(g.mean(dim=1)).squeeze(-1)       # (B,)
        return score, attn

    def forward(
        self, eeg: torch.Tensor, env: torch.Tensor, return_attention_weights: bool = False,
    ):
        """
        Parameters
        ----------
        eeg : Tensor, shape (B, C, T)
        env : Tensor, shape (B, 2, T) -- env[:, 0] = candidate A, env[:, 1] = candidate B
        return_attention_weights : bool
            If True, also return a dict with each candidate's GAT attention
            weights, shape (B, heads, N, N), for interpretability.

        Returns
        -------
        logits : Tensor, shape (B, 2) -- [score_a, score_b]
        attn : dict[str, Tensor], only if return_attention_weights=True
        """
        t_feat = self.temporal_encoder(eeg)          # (B, C, F_t)
        f_feat = self.frequency_encoder(eeg)          # (B, C, F_f)
        node_feats = self.node_fusion(t_feat, f_feat)  # (B, C, F_node) -- shared by both candidates

        score_a, attn_a = self._score_candidate(node_feats, env[:, 0, :], return_attention_weights)
        score_b, attn_b = self._score_candidate(node_feats, env[:, 1, :], return_attention_weights)

        logits = torch.stack([score_a, score_b], dim=1)  # (B, 2)
        if return_attention_weights:
            return logits, {"attn_a": attn_a, "attn_b": attn_b}
        return logits
