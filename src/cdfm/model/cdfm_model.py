"""CDFM model architecture.

Alternating attention model for causal discovery from observational data.

Architecture:
  1. ColEmbedding:       SetTransformer over N → (B, N, D, E)
  2. Alternating Blocks: num_layers×2 MHA+FFN blocks, interleaved D↔N
  3. Imputation:         ResidualBlock → per-position quantile predictions
  4. Experts:            InducedSelfAttentionBlock (optional MoE)
  5. Compression:        InducedSelfAttentionBlock N→2 → (B, D, 2, E)
  6. Graph head:         Bilinear GraphInference → (B, D, D)

Input: 2-channel (value, missing_mask).
"""

import torch
from torch import nn, Tensor
from typing import Optional

from cdfm.model.embedding import ColEmbedding
from cdfm.model.layers import MultiheadAttentionBlock, InducedSelfAttentionBlock
from cdfm.model.graph_inference import GraphInference
from cdfm.model.imputation import ResidualBlock


class CDFM(nn.Module):
    """CDFM causal discovery model.

    Architecture:
      Embed:      ColEmbedding (SetTransformer blocks over N)
      Blocks:     num_layers×2 MHA+FFN blocks, alternating D↔N
      Impute:     ResidualBlock → per-position quantile predictions
      Experts:    InducedSelfAttentionBlock (optional MoE)
      Compress:   InducedSelfAttentionBlock N→2 → (B, D, 2, E)
      Graph:      Bilinear GraphInference → (B, D, D)

    Input: 2-channel (value, missing_mask).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        p_dim: int = 64,
        nhead: int = 8,
        dim_feedforward: int = 512,
        num_inds: int = 128,
        dropout: float = 0.0,
        temp_init: float = 2.0,
        bias_init: float = -3.0,
        num_quantiles: int = 21,
        impute_h_dim: int = 256,
        num_of_experts: int = 0,
        num_layers: int = 4,
        num_blocks_of_embedding: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_inds = num_inds
        self.num_quantiles = num_quantiles
        self.num_of_experts = num_of_experts
        self.num_layers = num_layers
        self.num_total_blocks = num_layers * 2

        self.embedding = ColEmbedding(
            embed_dim=embed_dim, num_blocks=num_blocks_of_embedding, nhead=nhead,
            dim_feedforward=dim_feedforward, num_inds=num_inds, dropout=dropout,
        )

        # Stage 2: Alternating single MHA+FFN blocks (GELU activation, pre-norm)
        # Even indices (0,2,4,...): attention over D (per-sample variable attention)
        # Odd indices  (1,3,5,...): attention over N (per-variable sample attention)
        self.blocks = nn.ModuleList([
            MultiheadAttentionBlock(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(self.num_total_blocks)
        ])

        # Imputation: ResidualBlock per-position (E → num_quantiles)
        self.impute_head = ResidualBlock(embed_dim, impute_h_dim, num_quantiles, dropout)

        # Stage 3: Optional experts (over N, before compression)
        if self.num_of_experts > 0:
            self.experts = InducedSelfAttentionBlock(
                d_model=embed_dim, nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_inds=self.num_of_experts * 2, dropout=dropout,
            )

        # Stage 4: Compression N -> 2 (cause/effect inducing points)
        self.compression = InducedSelfAttentionBlock(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=2, dropout=dropout,
        )

        # Stage 5: GraphInference (bilinear graph head)
        self.inference = GraphInference(
            agg_dim=embed_dim, p_dim=p_dim,
            temp_init=temp_init, bias_init=bias_init,
        )

    def forward(
        self,
        X: Tensor,
        n_vec: Optional[Tensor] = None,
        d_vec: Optional[Tensor] = None,
        row_attn_mask: Optional[Tensor] = None,
    ):
        """Forward pass: from observed data (with mask) to causal graph + imputations.

        Args:
            X: (batch_size, max_n, max_d, 2) — padded observation matrices
               with 2 channels: (value, missing_mask).
            n_vec: (batch_size,) — number of actual samples per batch element.
            d_vec: (batch_size,) — number of actual variables per batch element.
            row_attn_mask: (batch_size, max_d, max_d) — per-item causal attention
                           mask for D-axis blocks. True/negative = blocked.

        Returns:
            W: (batch_size, max_d, max_d) — edge logits, zero diagonal.
            missing_preds: (num_missing, num_quantiles) — quantile predictions
                           for missing positions. (0, num_quantiles) when no
                           values are missing (never None, for torch.compile).
        """
        B, N, D, _ = X.shape
        device = X.device
        E = self.embed_dim

        # ---- Build masks (reused throughout) ----
        n_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        n_mask = n_idx >= (n_vec.unsqueeze(1) if n_vec is not None
                           else torch.full((B, 1), N + 1, device=device))

        d_idx = torch.arange(D, device=device).unsqueeze(0).expand(B, D)
        d_mask = d_idx >= (d_vec.unsqueeze(1) if d_vec is not None
                           else torch.full((B, 1), D + 1, device=device))

        # ---- Expand per-item causal mask for D-axis blocks ----
        # row_attn_mask: (B, D, D) → (B*N, D, D)
        attn_mask_row = None
        if row_attn_mask is not None:
            attn_mask_row = row_attn_mask.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, D, D)

        # ---- Stage 1: ColEmbedding (SetTransformer blocks over N) ----
        z = self.embedding(X, n_mask=n_mask)  # (B, N, D, E)
        mask_d = d_mask.unsqueeze(1).expand(B, N, D).reshape(B * N, D)
        mask_n = n_mask.unsqueeze(1).expand(B, D, N).reshape(B * D, N)

        # ---- Stage 2: Alternating MHA+FFN blocks ----
        for i in range(self.num_total_blocks):
            block = self.blocks[i]
            if i % 2 == 0:
                z_flat = z.reshape(B * N, D, self.embed_dim)
                z_flat = block(z_flat, key_padding_mask=mask_d, attn_mask=attn_mask_row)
                z = z_flat.reshape(B, N, D, self.embed_dim)
            else:
                z = z.permute(0, 2, 1, 3)
                z_flat = z.reshape(B * D, N, self.embed_dim)
                z_flat = block(z_flat, key_padding_mask=mask_n)
                z = z_flat.reshape(B, D, N, self.embed_dim)
                z = z.permute(0, 2, 1, 3)

        # ---- Imputation: ResidualBlock only on missing positions ----
        missing_mask = X[:, :, :, 1].bool()  # (B, N, D)
        missing_emb = z[missing_mask]           # (num_missing, E), may be (0, E)
        missing_preds = self.impute_head(missing_emb)  # (num_missing, Q), may be (0, Q)

        # ---- Stage 3+4: Experts + Compression N -> 2 ----
        x = z.permute(0, 2, 1, 3).reshape(B * D, N, E)
        mask_n = n_mask.unsqueeze(1).expand(B, D, N).reshape(B * D, N)
        if self.num_of_experts > 0:
            x = self.experts(x, key_padding_mask=mask_n)  # (B*D, num_of_experts*2, E)
            E_global = self.compression(x)                 # (B*D, 2, E)
        else:
            E_global = self.compression(x, key_padding_mask=mask_n)  # (B*D, 2, E)
        E_global = E_global.reshape(B, D, 2, E)  # (B, D, 2, E)

        # ---- Stage 5: GraphInference ----
        W = self.inference(E_global)  # (B, D, D)

        return W, missing_preds
