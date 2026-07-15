"""Core inference functions for CDFM.

Provides single forward-pass inference and a full predict pipeline.
"""

import time
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from cdfm.preprocessing import create_input_tensor

logger = logging.getLogger("cdfm")


@dataclass
class CDFMResult:
    """Result of a CDFM causal discovery inference.

    Attributes:
        logits: (D, D) raw edge logits (zero diagonal).
        probabilities: (D, D) sigmoid(logits), in [0, 1].
        adjacency: (D, D) binary adjacency after thresholding.
        threshold: Probability threshold in [0, 1] used for binarization.
                   edge exists if sigmoid(logit) > threshold.
        runtime_sec: Wall-clock time for the forward pass.
    """
    logits: np.ndarray
    probabilities: np.ndarray
    adjacency: np.ndarray | None
    threshold: float | None
    runtime_sec: float

    def __repr__(self) -> str:
        D = self.logits.shape[0]
        n_edges = int(self.adjacency.sum()) if self.adjacency is not None else 0
        return (
            f"CDFMResult(n_vars={D}, edges={n_edges}, "
            f"threshold={self.threshold}, "
            f"runtime={self.runtime_sec:.2f}s)"
        )




def single_inference(
    model: nn.Module,
    X_t: torch.Tensor,
    n_vec: torch.Tensor,
    d_vec: torch.Tensor,
) -> np.ndarray:
    """Run a single forward pass through the model.

    Args:
        model: CDFM model in eval mode.
        X_t: (1, N, D, 2) input tensor (values + mask).
        n_vec: (1,) tensor with actual sample count.
        d_vec: (1,) tensor with actual variable count.

    Returns:
        (D, D) numpy array of edge logits, zero diagonal.

    Raises:
        RuntimeError: On CUDA OOM (after cleanup).
    """
    with torch.no_grad():
        try:
            W, _ = model(X_t, n_vec, d_vec)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error("CUDA OOM during inference. Try fewer variables or use CPU.")
            raise

    W_np = W[0].cpu().numpy().astype(np.float64)
    np.fill_diagonal(W_np, 0.0)
    return W_np


def single_inference_with_imputation(
    model: nn.Module,
    X_t: torch.Tensor,
    n_vec: torch.Tensor,
    d_vec: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run a single forward pass and return both logits and imputation predictions.

    Unlike ``single_inference`` which discards the model's imputation output,
    this function returns the full ``missing_preds`` tensor from the model's
    imputation head (ResidualBlock).

    Args:
        model: CDFM model in eval mode.
        X_t: (1, N, D, 2) input tensor (values + mask channels).
        n_vec: (1,) tensor with actual sample count.
        d_vec: (1,) tensor with actual variable count.

    Returns:
        (logits, missing_preds) tuple:
        - logits: (D, D) numpy array of edge logits, zero diagonal.
        - missing_preds: (M, Q) numpy array of quantile predictions, or None
          if there are no missing positions (mask channel is all zeros).

    Raises:
        RuntimeError: On CUDA OOM (after cleanup).
    """
    with torch.no_grad():
        try:
            W, missing_preds = model(X_t, n_vec, d_vec)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error("CUDA OOM during inference. Try fewer variables or use CPU.")
            raise

    W_np = W[0].cpu().numpy().astype(np.float64)
    np.fill_diagonal(W_np, 0.0)

    if missing_preds is not None and missing_preds.numel() > 0:
        mp_np = missing_preds.cpu().numpy().astype(np.float64)
    else:
        mp_np = None

    return W_np, mp_np


def predict(
    model: nn.Module,
    data: np.ndarray,
    device: torch.device = torch.device("cpu"),
    missing_mask: np.ndarray | None = None,
) -> CDFMResult:
    """Run inference on preprocessed data, return logits and probabilities.

    Expects already-preprocessed data (encoded, missing-filled, standardized).
    Does NOT binarize — the wrapper handles thresholding.

    Args:
        model: CDFM model in eval mode.
        data: (N, D) float32, standardized with 0 at missing positions.
        device: Torch device.
        missing_mask: (N, D) bool mask, True = missing.

    Returns:
        CDFMResult with logits and probabilities (adjacency unset, threshold=0).
    """
    N, D = data.shape
    if N < 2:
        raise ValueError(f"At least 2 samples required, got N={N}")
    if D < 2:
        raise ValueError(f"At least 2 variables required, got D={D}")

    data = data.astype(np.float32, copy=False)

    t0 = time.perf_counter()
    X_t, n_vec, d_vec = create_input_tensor(
        data, missing_mask=missing_mask, device=device
    )
    logits = single_inference(model, X_t, n_vec, d_vec)
    runtime = time.perf_counter() - t0

    logits_clipped = np.clip(logits, -50.0, 50.0)
    probs = 1.0 / (1.0 + np.exp(-logits_clipped))
    np.fill_diagonal(probs, 0.0)

    return CDFMResult(
        logits=logits,
        probabilities=probs,
        adjacency=None,
        threshold=None,
        runtime_sec=runtime,
    )
