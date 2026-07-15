"""Attention layers for CDFM."""

from __future__ import annotations
import math
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
) -> Tensor:
    """Multi-head attention forward .

    Always uses batch_first=True. Uses F.scaled_dot_product_attention for efficiency.

    Args:
        query: (..., tgt_len, embed_dim)
        key: (..., src_len, embed_dim)
        value: (..., src_len, embed_dim)
        num_heads: number of attention heads
        in_proj_weight: combined QKV projection weight
        in_proj_bias: combined QKV projection bias
        dropout_p: attention dropout probability
        out_proj_weight: output projection weight
        out_proj_bias: output projection bias
        training: whether in training mode
        key_padding_mask: (..., src_len), True = ignore
        attn_mask: attention mask

    Returns:
        (..., tgt_len, embed_dim)
    """
    tgt_len, bsz, embed_dim = query.shape[-2], query.shape[0], query.shape[-1]
    src_len = key.shape[-2]
    head_dim = embed_dim // num_heads

    # Project Q, K, V separately from their respective inputs
    # in_proj_weight has shape (3*embed_dim, embed_dim)
    w_q = in_proj_weight[:embed_dim, :]
    w_k = in_proj_weight[embed_dim:2*embed_dim, :]
    w_v = in_proj_weight[2*embed_dim:, :]
    if in_proj_bias is not None:
        b_q = in_proj_bias[:embed_dim]
        b_k = in_proj_bias[embed_dim:2*embed_dim]
        b_v = in_proj_bias[2*embed_dim:]
    else:
        b_q = b_k = b_v = None

    q = F.linear(query, w_q, b_q)
    k = F.linear(key, w_k, b_k)
    v = F.linear(value, w_v, b_v)

    # Reshape to multi-head: (B, num_heads, seq_len, head_dim)
    q = q.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, src_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, src_len, num_heads, head_dim).transpose(1, 2)

    # Handle key_padding_mask for SDPA: needs shape (B, src_len), True = ignore
    if key_padding_mask is not None:
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=q.dtype,
        )

    if attn_mask is not None:
        # If 3D mask (B, L, S): add head dimension for broadcasting over heads.
        # SDPA q has shape (B, num_heads, L, head_dim), so a (B, L, S) mask
        # needs to become (B, 1, L, S) to broadcast correctly.
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=q.dtype,
            check_other=False,
        )

    # Use PyTorch's efficient SDPA
    attn_output = scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    )

    # Reshape back: (B, tgt_len, embed_dim)
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)

    # Output projection
    return F.linear(attn_output, out_proj_weight, out_proj_bias)


class MultiheadAttention(nn.Module):
    """Multi-head attention .

    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.out_proj.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.out_proj.bias, -bound, bound)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute multi-head attention.

        Args:
            query: (..., tgt_len, embed_dim)
            key: (..., src_len, embed_dim)
            value: (..., src_len, embed_dim)
            key_padding_mask: (..., src_len), True = ignore
            attn_mask: attention mask tensor

        Returns:
            (..., tgt_len, embed_dim)
        """
        return multi_head_attention_forward(
            query, key, value,
            self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.dropout,
            self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )


class MultiheadAttentionBlock(nn.Module):
    """Attention block Pre-norm architecture.

    Args:
        d_model: model dimension
        nhead: number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: dropout probability
        activation: activation function name
        norm_first: if True, use pre-norm (default)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.attn = MultiheadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}
        act_fn = act_map.get(activation, nn.GELU)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = act_fn()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Use PyTorch default init (Kaiming uniform) for output projections."""
        # PyTorch Linear layers use Kaiming uniform by default, so we just reset
        # the layers to use their default init (overwrites the parent class zero init)
        nn.init.kaiming_uniform_(self.attn.out_proj.weight, a=math.sqrt(5))
        if self.attn.out_proj.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.attn.out_proj.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.attn.out_proj.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.linear2.weight, a=math.sqrt(5))
        if self.linear2.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear2.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.linear2.bias, -bound, bound)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Process input through attention + FFN.

        Args:
            q: (..., tgt_len, d_model)
            k: (..., src_len, d_model), defaults to q
            v: (..., src_len, d_model), defaults to q
            key_padding_mask: (..., src_len), True = ignore
            attn_mask: attention mask

        Returns:
            (..., tgt_len, d_model)
        """
        k = q if k is None else k
        v = q if v is None else v

        x = q
        if self.norm_first:
            attn_out = self._attn_block(self.norm1(q), self.norm1(k), self.norm1(v),
                                        key_padding_mask, attn_mask)
            x = x + attn_out
            x = x + self._ff_block(self.norm2(x))
        else:
            attn_out = self._attn_block(q, k, v, key_padding_mask, attn_mask)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self._ff_block(x))
        return x

    def _attn_block(self, q, k, v, key_padding_mask, attn_mask):
        return self.dropout1(self.attn(q, k, v, key_padding_mask, attn_mask))

    def _ff_block(self, x):
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))


class InducedSelfAttentionBlock(nn.Module):
    """Compression-mode induced self-attention: input -> inducing points only.

    
    only performs Stage 1: inducing points attend to the input set and produce a
    compressed representation. Output shape is (..., num_inds, d_model).

    Complexity: O(n * m) where m = num_inds << n.

    Args:
        d_model: model dimension
        nhead: number of attention heads
        dim_feedforward: FFN hidden dimension
        num_inds: number of inducing points (e.g., 2 for cause/effect)
        dropout: dropout probability
        activation: activation function name
        norm_first: if True, use pre-norm
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.num_inds = num_inds
        self.d_model = d_model

        # Inducing points attention to input (Stage 1 only)
        self.multihead_attn = MultiheadAttentionBlock(
            d_model, nhead, dim_feedforward, dropout, activation, norm_first
        )

        # Learnable inducing points
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Compress input sequence into inducing point outputs.

        Args:
            src: (..., seq_len, d_model)
            key_padding_mask: (..., seq_len), True = ignore padded positions

        Returns:
            (..., num_inds, d_model) -- compressed representation
        """
        *batch_shape, seq_len, d_model = src.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)

        # Inducing points (as query) attend to input (as key/value)
        # Output: (..., num_inds, d_model)
        hidden = self.multihead_attn(
            q=ind_vectors, k=src, v=src,
            key_padding_mask=key_padding_mask,
        )
        return hidden


class BidirectionalInducedAttentionBlock(nn.Module):
    """Full two-stage induced self-attention preserving sequence length.

    Implements bidirectional induced attention:
      1. Stage 1: Learnable inducing points attend to input (Q=inds, K,V=input)
      2. Stage 2: Input attends to inducing point outputs (Q=input, K,V=hidden)

    Output shape matches input shape: (..., seq_len, d_model).
    Complexity: O(N * num_inds) where num_inds << N.

    Args:
        d_model: model dimension
        nhead: number of attention heads
        dim_feedforward: FFN hidden dimension
        num_inds: number of inducing points (e.g., 16)
        dropout: dropout probability
        activation: activation function name
        norm_first: if True, use pre-norm architecture
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.num_inds = num_inds
        self.d_model = d_model

        # Stage 1: inducing points attend to input
        self.multihead_attn1 = MultiheadAttentionBlock(
            d_model, nhead, dim_feedforward, dropout, activation, norm_first
        )
        # Stage 2: input attends to inducing point outputs
        self.multihead_attn2 = MultiheadAttentionBlock(
            d_model, nhead, dim_feedforward, dropout, activation, norm_first
        )

        # Learnable inducing points
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Apply bidirectional induced self-attention.

        Args:
            src: (..., seq_len, d_model)
            key_padding_mask: (..., seq_len), True = ignore padded positions.
                Only applied in Stage 1 (inducing points should not attend to padding).

        Returns:
            (..., seq_len, d_model) -- same shape as input
        """
        *batch_shape, seq_len, d_model = src.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)

        # Stage 1: inducing points attend to input
        # key_padding_mask prevents attending to padded positions
        hidden = self.multihead_attn1(
            q=ind_vectors, k=src, v=src,
            key_padding_mask=key_padding_mask,
        )

        # Stage 2: input attends to inducing point outputs
        # No mask needed — inducing points are never padding
        out = self.multihead_attn2(
            q=src, k=hidden, v=hidden,
        )

        return out
