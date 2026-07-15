"""Column embedding with per-column SetTransformer."""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
from cdfm.model.encoders import SetTransformerEncoder

class ColEmbedding(nn.Module):
    """Column embedding: SetTransformer over samples + weight/bias affine transform.

    Pipeline:
      1. in_linear: 2-channel (value, mask) → embedding_dim
      2. SetTransformer: induced self-attention over N samples per column
      3. ln_w(out_w(…)): per-sample weights
      4. ln_b(out_b(…)): per-sample biases
      5. embeddings = value * weights + biases

    Processes all D variables in parallel via (B*D, N, E) reshaping.

    Args:
        embed_dim: Embedding dimension. Default 128.
        num_blocks: SetTransformer blocks. Default 3.
        nhead: Attention heads. Default 8.
        dim_feedforward: FFN hidden dim. Default 512.
        num_inds: Inducing points. Default 16.
        dropout: Dropout. Default 0.0.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_blocks: int = 3,
        nhead: int = 8,
        dim_feedforward: int = 512,
        num_inds: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.in_linear = nn.Linear(2, embed_dim)


        self.tf_col = SetTransformerEncoder(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
        )

        self.ln_w = nn.LayerNorm(embed_dim)
        self.ln_b = nn.LayerNorm(embed_dim)
        self.out_w = nn.Linear(embed_dim, embed_dim)
        self.out_b = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, X: Tensor, n_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Embed 2-channel values with per-column SetTransformer.

        Args:
            X: (B, N, D, 2) — padded observations with mask channel.
            n_mask: (B, N) — True at padded sample positions.

        Returns:
            (B, N, D, E)
        """
        B, N, D, _ = X.shape
        E = self.embed_dim

        features = X.permute(0, 2, 1, 3).reshape(B * D, N, 2)
        src = self.in_linear(features)

        if n_mask is not None:
            mask = n_mask.unsqueeze(1).expand(B, D, N).reshape(B * D, N)
        else:
            mask = None
        src = self.tf_col(src, key_padding_mask=mask)

        weights = self.ln_w(self.out_w(src))
        biases = self.ln_b(self.out_b(src))

        features_scalar = features[..., :1]
        embeddings = features_scalar * weights + biases

        return embeddings.reshape(B, D, N, E).permute(0, 2, 1, 3)
