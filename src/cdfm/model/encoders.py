"""Transformer encoder stacks for causal discovery model.


"""

from __future__ import annotations
from typing import Optional

from torch import nn, Tensor

from .layers import MultiheadAttentionBlock, InducedSelfAttentionBlock, BidirectionalInducedAttentionBlock


class Encoder(nn.Module):
    """Stack of multihead attention blocks. No positional encoding.

    Args:
        num_blocks: number of transformer layers
        d_model: model dimension
        nhead: number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: dropout probability
        activation: activation function name
        norm_first: if True, use pre-norm architecture
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList([
            MultiheadAttentionBlock(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, activation=activation,
                norm_first=norm_first,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Process input through stacked transformer blocks.

        Args:
            src: (..., seq_len, d_model)
            key_padding_mask: (..., seq_len), True = ignore
            attn_mask: attention mask

        Returns:
            (..., seq_len, d_model)
        """
        out = src
        for block in self.blocks:
            out = block(q=out, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return out


class SetTransformer(nn.Module):
    """Stack of compression-mode induced self-attention blocks.

    
    this version compresses to num_inds outputs per block. Each block's
    inducing points attend to the previous block's outputs, refining
    the compressed representation.

    Args:
        num_blocks: number of induced attention blocks
        d_model: model dimension
        nhead: number of attention heads
        dim_feedforward: FFN hidden dimension
        num_inds: number of inducing points
        dropout: dropout probability
        activation: activation function name
        norm_first: if True, use pre-norm
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList([
            InducedSelfAttentionBlock(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_inds=num_inds, dropout=dropout,
                activation=activation, norm_first=norm_first,
            )
            for _ in range(num_blocks)
        ])
        self.num_inds = num_inds

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Compress set through stacked induced attention blocks.

        Args:
            src: (..., seq_len, d_model)
            key_padding_mask: (..., seq_len), True = ignore

        Returns:
            (..., num_inds, d_model)
        """
        out = src
        for block in self.blocks:
            out = block(out, key_padding_mask=key_padding_mask)
        return out


class SetTransformerEncoder(nn.Module):
    """Stack of bidirectional induced self-attention blocks preserving sequence length.

    Unlike the compression-mode SetTransformer (which reduces N to num_inds),
    this encoder preserves the full sequence length using two-stage induced attention
    (Stage 1: inducing → input, Stage 2: input → inducing → input).

    Used for per-variable processing over samples in alternating attention rounds.
    Complexity: O(N * num_inds) per block.

    Args:
        num_blocks: number of bidirectional induced attention blocks
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
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int = 16,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList([
            BidirectionalInducedAttentionBlock(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_inds=num_inds, dropout=dropout,
                activation=activation, norm_first=norm_first,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Process input through stacked bidirectional induced attention blocks.

        Args:
            src: (..., seq_len, d_model)
            key_padding_mask: (..., seq_len), True = ignore padded positions.
                Applied in Stage 1 of each block.

        Returns:
            (..., seq_len, d_model) -- same shape as input
        """
        out = src
        for block in self.blocks:
            out = block(out, key_padding_mask=key_padding_mask)
        return out
