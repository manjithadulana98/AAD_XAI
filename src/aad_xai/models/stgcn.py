from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
#  Hand-written graph convolution (no torch_geometric dependency)
# --------------------------------------------------------------------------- #

def _default_adjacency(n_channels: int) -> torch.Tensor:
    """Return a fixed binary adjacency matrix for the standard 10-20 layout.

    For channel counts ≤64 we use an approximate *distance-based* adjacency
    (connect each node to its ~6 nearest neighbours on a 2-D grid).
    For simplicity this creates a ring + skip-1 topology which is a reasonable
    proxy when no montage information is available.

    TODO: Replace with a true geodesic distance matrix when montage info
    is available (e.g., from ``mne.channels.make_standard_montage``).
    """
    A = torch.zeros(n_channels, n_channels)
    for i in range(n_channels):
        for offset in [-2, -1, 1, 2]:
            j = (i + offset) % n_channels
            A[i, j] = 1.0
    # Self-loops
    A = A + torch.eye(n_channels)
    # Symmetric normalisation  D^{-1/2} A D^{-1/2}
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-8)))
    return D_inv_sqrt @ A @ D_inv_sqrt


class GraphConv(nn.Module):
    """Simple spectral-free graph convolution:  H' = σ(Â · H · W).

    Parameters
    ----------
    in_features : int
    out_features : int
    adj : Tensor
        Pre-normalised adjacency matrix of shape ``(N, N)`` where N is the
        number of graph nodes (EEG channels).
    """

    def __init__(self, in_features: int, out_features: int, adj: torch.Tensor):
        super().__init__()
        self.register_buffer("adj", adj)  # (N, N)
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, F_in)  →  out: (B, N, F_out)."""
        # Graph diffusion: aggregate neighbours
        h = torch.matmul(self.adj, x)      # (B, N, F_in)
        h = torch.matmul(h, self.weight)   # (B, N, F_out)
        return h + self.bias


# --------------------------------------------------------------------------- #
#  ST-GCN: Spatiotemporal Graph Convolutional Network for AAD
# --------------------------------------------------------------------------- #

class STGCN(nn.Module):
    """CNN + GCN hybrid for spatiotemporal EEG modelling.

    Pipeline::

        Temporal Conv1d → BN → ELU   (per-channel temporal features)
        Reshape to (B, N, F)
        GraphConv → ELU              (spatial message-passing)
        GraphConv → ELU
        Global pool → Linear head

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (graph nodes).
    n_classes : int
        Output classes.
    temp_filters : int
        Number of temporal Conv1d filters.
    gcn_hidden : int
        Hidden dim of graph conv layers.
    adj : Tensor | None
        Optional custom adjacency.  If ``None`` a default ring adjacency
        is used (replace with a real montage-based matrix for production).
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 2,
        temp_filters: int = 32,
        gcn_hidden: int = 32,
        adj: torch.Tensor | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if adj is None:
            adj = _default_adjacency(n_channels)

        # Temporal feature extraction (shared across channels)
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, temp_filters, kernel_size=9, padding=4),
            nn.BatchNorm1d(temp_filters),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),  # (B, temp_filters, 1)
        )

        # Map temporal features back to per-node representations
        # After pooling we have (B, temp_filters).  We reshape to
        # (B, n_channels, F_node) via a learned projection.
        self.node_proj = nn.Linear(temp_filters, n_channels * 8)
        self.n_channels = n_channels
        self.node_dim = 8

        # Graph convolution layers
        self.gc1 = GraphConv(self.node_dim, gcn_hidden, adj)
        self.gc2 = GraphConv(gcn_hidden, gcn_hidden, adj)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.head = nn.Linear(gcn_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → logits (B, n_classes)."""
        B = x.size(0)

        # Temporal encoding
        h = self.temporal(x).squeeze(-1)          # (B, temp_filters)
        h = self.node_proj(h)                     # (B, n_channels * node_dim)
        h = h.view(B, self.n_channels, self.node_dim)  # (B, N, F)

        # Graph convolutions
        h = F.elu(self.gc1(h))
        h = self.dropout(h)
        h = F.elu(self.gc2(h))                    # (B, N, gcn_hidden)

        # Global mean pool over nodes
        h = h.mean(dim=1)                         # (B, gcn_hidden)
        return self.head(h)


# Backward-compat alias
STGCNStub = STGCN

