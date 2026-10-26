from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateConditionedGAT(nn.Module):
    """Multi-head graph-attention layer over candidate-conditioned node
    features, restricted to a fixed physical electrode adjacency.

    Standard additive-attention GAT (Velickovic et al., 2018). Attention
    logits for non-adjacent electrode pairs are set to ``-inf`` before the
    softmax, so attention mass is *provably* zero there -- the physical
    electrode-adjacency graph hard-constrains attention support, it does not
    merely bias it.

    Speech conditioning happens upstream (``TimeResolvedFiLM``, before
    time-pooling): the node features this layer receives already depend on
    the candidate, so the attention scores computed from them are
    candidate-conditioned. The same instance scores both candidates.

    Parameters
    ----------
    n_channels : int
        Number of graph nodes (EEG electrodes).
    in_features : int
        Input node feature dimension.
    out_features : int
        Output node feature dimension (must be divisible by ``n_heads``).
    adjacency : Tensor
        (n_channels, n_channels) electrode adjacency matrix. Non-zero entries
        mark physical neighbors; self-loops are added automatically.
    n_heads : int
    dropout : float
        Dropout applied to attention weights.
    """

    def __init__(
        self,
        n_channels: int,
        in_features: int,
        out_features: int,
        adjacency: torch.Tensor,
        n_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        if out_features % n_heads != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by n_heads ({n_heads})")
        if adjacency.shape != (n_channels, n_channels):
            raise ValueError(f"adjacency must be ({n_channels},{n_channels}), got {tuple(adjacency.shape)}")

        self.n_channels = n_channels
        self.n_heads = n_heads
        self.out_features = out_features
        self.head_dim = out_features // n_heads

        adj = (adjacency.float() + torch.eye(n_channels)) > 0  # add self-loops, binarize
        self.register_buffer("adj_mask", adj)  # (N, N) bool

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_feats: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        """
        Parameters
        ----------
        node_feats : Tensor, shape (B, N, F_in) -- candidate-conditioned
        return_attention_weights : bool
            If True, also return the (B, heads, N, N) attention tensor.

        Returns
        -------
        out : Tensor, shape (B, N, F_out)
        attn : Tensor, shape (B, heads, N, N), only if return_attention_weights=True
        """
        B, N, _ = node_feats.shape
        Wh = self.W(node_feats).view(B, N, self.n_heads, self.head_dim)   # (B, N, heads, d)

        # Additive attention, decomposed: e_ij = LeakyReLU(a_src . Wh_i + a_dst . Wh_j)
        src = torch.einsum("bnhd,hd->bnh", Wh, self.a_src)        # (B, N, heads)
        dst = torch.einsum("bnhd,hd->bnh", Wh, self.a_dst)        # (B, N, heads)
        e = src.unsqueeze(2) + dst.unsqueeze(1)                    # (B, N_i, N_j, heads)
        e = self.leaky_relu(e).permute(0, 3, 1, 2)                  # (B, heads, N_i, N_j)

        mask = self.adj_mask.unsqueeze(0).unsqueeze(0)               # (1, 1, N, N)
        e = e.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(e, dim=-1)                               # (B, heads, N_i, N_j)
        attn = self.dropout(attn)

        Wh_perm = Wh.permute(0, 2, 1, 3)                                # (B, heads, N, d)
        out = torch.matmul(attn, Wh_perm)                                # (B, heads, N, d)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.out_features)    # (B, N, F_out)
        out = F.elu(out)

        return (out, attn) if return_attention_weights else out
