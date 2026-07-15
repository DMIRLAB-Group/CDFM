"""Missing value imputation utilities.

Follows chronos2's ResidualBlock design for quantile prediction:
- ResidualBlock: Linear(in→h) → ReLU → Linear(h→out) → Dropout + skip connection
- Directly maps per-position embeddings to num_quantiles predictions
"""

import torch
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    """Residual MLP block matching chronos2's design.

    Two-layer feedforward with a linear residual shortcut.
    No layer norm (matching chronos2).

    Args:
        in_dim: Input dimension.
        h_dim: Hidden dimension (expansion).
        out_dim: Output dimension.
        dropout_p: Dropout probability (default 0.0).
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: hidden → act → output → dropout + residual.

        Args:
            x: (*, in_dim) — any leading dimensions preserved.

        Returns:
            (*, out_dim)
        """
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)
        return out + res
