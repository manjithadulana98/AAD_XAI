"""Phase 2 -- ST-GCN Stages (a)+(b): fixed-graph spectral conv, no temporal attention.

Reference: Wang, Cai & Li, "EEG-based Auditory Attention Detection with
Spatiotemporal Graph and Graph Convolutional Network," Interspeech 2023,
pp. 1144-1148, doi: 10.21437/Interspeech.2023-620.

Design note on the graph-convolution layer (read before changing it)
----------------------------------------------------------------------
The paper's Stage (b) is a literal spectral filter x' = U g_theta(Lambda) U^T x
with an independent trainable filter g_theta per kernel, g_theta in
R^(C x 64 x 64). Per the brief, this is implemented as a first-order/ChebNet-
style approximation (Kipf & Welling, arXiv:1609.02907) instead of a literal
eigendecomposition.

A naive single-term Kipf-Welling layer -- out = A_hat @ x @ W, with the fixed
propagation operator A_hat shared across kernels -- degenerates when the
input feature dimension is 1 (a scalar EEG amplitude per channel per
timestep, exactly our case): W becomes a (1 x C) row vector, so every
kernel's spatial pattern is just a scalar multiple of the SAME diffused
signal A_hat @ x. That collapses "C independent spectral filters" into one
filter with C different scales, which is not what the paper intends.

A two-term basis (identity + one-hop diffusion) was tried first and rejected:
with only 2 shared basis operators, any set of C kernel patterns is confined
to a 2-D subspace of R^64, so with C=5 kernels several pairs came out with
cosine similarity > 0.9 (checked empirically) -- better than exact rank-1,
but still far short of C independent filters.

Instead this uses a K-term Chebyshev-style polynomial basis of the diffusion
operator, {I, A_hat, A_hat^2, ..., A_hat^(K-1)}, precomputed once (K defaults
to n_kernels so the C kernels have enough basis directions to potentially
span an independent, non-degenerate pattern each), with an INDEPENDENT
learnable coefficient per (kernel, basis-term) pair:
    out[..., c] = bias[c] + sum_k theta[k, c] * (A_hat^k @ x)
Params stay tiny (K*C coefficients + C biases -- e.g. 5*5+5=30 for the
defaults) while each kernel can now mix the K basis operators differently
enough to produce a genuinely distinct spatial filter, not just a rescaled
copy of the others (verified empirically -- see stgcn/model.py's module
docstring test invocation in the Phase 2 report).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def normalized_diffusion_operator(adjacency: np.ndarray) -> torch.Tensor:
    """Symmetric-normalized one-hop diffusion operator D^-1/2 A D^-1/2.

    ``adjacency`` is the fixed, pre-determined Phase 1 adjacency (zero
    diagonal by construction -- self-loops are handled separately as their
    own basis term in ``GraphConvKW``, not folded in here).
    """
    A = torch.as_tensor(adjacency, dtype=torch.float32)
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg.clamp(min=1e-8)))
    return deg_inv_sqrt @ A @ deg_inv_sqrt


class GraphConvKW(nn.Module):
    """Fixed-graph, K-order Chebyshev-style spectral graph convolution.

    Input:  x  (B, 64, T)  -- one scalar EEG amplitude per channel per sample.
    Output: F  (B, C, 64, T)  -- C independent graph-filtered feature maps,
            matching the paper's F = [E^1_p, ..., E^C_p] in R^(C x 64 x T).
    """

    def __init__(self, adjacency: np.ndarray, n_kernels: int = 5, chebyshev_order: int | None = None):
        super().__init__()
        K = chebyshev_order if chebyshev_order is not None else n_kernels
        self.K = K
        self.n_kernels = n_kernels

        A_hat = normalized_diffusion_operator(adjacency)
        n = A_hat.shape[0]
        basis = [torch.eye(n)]
        for _ in range(1, K):
            basis.append(basis[-1] @ A_hat)
        self.register_buffer("basis", torch.stack(basis, dim=0))  # (K, 64, 64)

        self.theta = nn.Parameter(torch.randn(K, n_kernels) * 0.1)  # per (basis-term, kernel)
        self.bias = nn.Parameter(torch.zeros(n_kernels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, T) -> diffuse through each of the K basis operators once,
        # then mix into C kernels with independent coefficients per kernel.
        diffused = torch.einsum("kij,bjt->kbit", self.basis, x)          # (K, B, 64, T)
        out = torch.einsum("kc,kbit->bcit", self.theta, diffused)        # (B, C, 64, T)
        return out + self.bias.view(1, -1, 1, 1)


class STGCNGCNOnly(nn.Module):
    """Stages (a)+(b) plus a minimal classifier head (Stage (d)).

    No temporal attention (Stage (c)) -- that is Phase 3.

    Input:  x  (B, 64, T)  -- channel-major, matching AADNet's DTUDataset
            output directly (no transpose needed vs. the paper's (T, 64)
            notation -- this is the same data, just indexed the other way,
            and the graph-conv einsum above doesn't care which axis is last).
    Output: logits (B, 2)  -- attended/unattended class logits.
    """

    def __init__(self, adjacency: np.ndarray, n_kernels: int = 5, fc_hidden: int = 8,
                 n_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.graph_conv = GraphConvKW(adjacency, n_kernels=n_kernels)
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pool over time
        flat_dim = n_kernels * n_channels
        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        f = self.graph_conv(x)                       # (B, C, 64, T)
        C, N, T = f.shape[1], f.shape[2], f.shape[3]
        f = f.reshape(B, C * N, T)                    # (B, C*64, T)
        f = self.pool(f).squeeze(-1)                  # (B, C*64)
        h = self.act(self.bn1(self.fc1(f)))
        h = self.dropout(h)
        return self.fc2(h)                            # (B, 2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
