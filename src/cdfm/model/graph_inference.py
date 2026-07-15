"""Causal structure inference: bilinear graph head with learned temperature and bias."""

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class GraphInference(nn.Module):
    """Causal structure inference from aggregated representations.

    Uses L2-normalized cause/effect vectors with cosine similarity,
    learned temperature and sparsity bias.

    Args:
        agg_dim: Input dimension from aggregation. Default 128.
        p_dim: Projection dimension for U, V matrices. Default 64.
        temp_init: Initial value for learned temperature. Default 2.0.
        bias_init: Initial value for learned sparsity bias. Default -3.0.
    """

    def __init__(
        self,
        agg_dim: int = 128,
        p_dim: int = 64,
        temp_init: float = 2.0,
        bias_init: float = -3.0,
    ):
        super().__init__()
        self.p_dim = p_dim
        self.linear_cause = nn.Linear(agg_dim, p_dim)
        self.linear_effect = nn.Linear(agg_dim, p_dim)
        self.log_temp = nn.Parameter(torch.tensor(temp_init).log())
        self.final_bias = nn.Parameter(torch.tensor(bias_init))

    def forward(self, E_global: Tensor) -> Tensor:
        """Infer causal adjacency matrix from aggregated representations.

        Args:
            E_global: (batch_size, d, 2, agg_dim)
                E_global[:, :, 0, :] = cause embeddings
                E_global[:, :, 1, :] = effect embeddings

        Returns:
            W: (batch_size, d, d) — edge probabilities, zero diagonal.
                W[b, i, j] = probability of edge i -> j.
        """
        # Extract cause and effect vectors
        U = self.linear_cause(E_global[:, :, 0, :])    # (B, d, p)
        V = self.linear_effect(E_global[:, :, 1, :])    # (B, d, p)

        # Raw dot product (unbounded, avoids cosine sim clustering at 0)
        # Scale by sqrt(p) for variance control
        W_raw = torch.bmm(U, V.transpose(1, 2)) / (self.p_dim ** 0.5)  # (B, d, d)

        # Apply learned temperature and bias
        temp = torch.exp(self.log_temp)
        bias = self.final_bias

        logits = temp * W_raw + bias

        # Zero diagonal (no self-loops) — set to very negative so sigmoid → 0
        D = logits.shape[-1]
        eye = torch.eye(D, device=logits.device, dtype=logits.dtype).unsqueeze(0)
        logits = logits * (1 - eye) + eye * (-1e9)

        return logits
