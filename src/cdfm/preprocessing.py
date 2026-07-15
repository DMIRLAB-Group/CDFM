"""Data preprocessing utilities for CDFM inference.

Provides standardization, categorical encoding, missing value handling,
and input tensor assembly.
"""

import numpy as np
import torch


def encode_categorical(data: np.ndarray) -> tuple[np.ndarray, dict]:
    """Convert string/object columns to integer codes (0, 1, 2, …).

    Detects non-numeric columns and replaces them with ordinal integer codes.
    Numeric columns are left unchanged.  Returns the encoded float32 array
    and a mapping dict for inspection.

    Args:
        data: (N, D) numpy array, possibly with mixed dtypes.

    Returns:
        (encoded, mapping) — encoded is (N, D) float32; mapping is
        ``{col_idx: {category: code}}`` for non-numeric columns.
    """
    N, D = data.shape
    encoded = np.zeros((N, D), dtype=np.float32)
    mapping = {}

    for j in range(D):
        col = data[:, j]
        # Check if column is non-numeric (object, string, or mixed type)
        if col.dtype.kind in ('U', 'S', 'O'):
            unique = sorted(set(col))
            code_map = {v: i for i, v in enumerate(unique)}
            encoded[:, j] = np.array([code_map[v] for v in col], dtype=np.float32)
            mapping[j] = code_map
        else:
            encoded[:, j] = col.astype(np.float32)

    return encoded, mapping


def standardize(
    data: np.ndarray,
    missing_mask: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-variable z-score standardization.

      mean = X.mean(axis=0), std = X.std(axis=0)
      return (X - mean) / where(std == 0, 1.0, max(std, eps))

    If missing_mask is provided, mean/std are computed only on observed
    (non-missing) values.

    Args:
        data: (N, D) float array (NaN positions should be zero-filled
              or handled via missing_mask before calling).
        missing_mask: (N, D) bool, True = missing. If provided, mean/std are
                      computed ignoring these positions.
        eps: Minimum std to prevent division by zero. Default 1e-6.

    Returns:
        (data_std, mean, std) — all float32 arrays.
        data_std shape (N, D), mean/std shape (D,).
    """
    if missing_mask is not None and missing_mask.any():
        # Compute mean/std on observed values only
        observed = ~missing_mask
        mean = np.zeros(data.shape[1], dtype=np.float32)
        std = np.zeros(data.shape[1], dtype=np.float32)
        for j in range(data.shape[1]):
            col = data[observed[:, j], j]
            if len(col) > 0:
                mean[j] = col.mean()
                std[j] = col.std()
            # else: mean=0, std=0 → will be guarded below
    else:
        mean = data.mean(axis=0).astype(np.float32)
        std = data.std(axis=0).astype(np.float32)

    result = (data - mean) / np.where(std == 0.0, 1.0, np.maximum(std, eps))
    return result.astype(np.float32), mean, std


def auto_detect_missing(data: np.ndarray) -> np.ndarray:
    """Detect missing values in data.

    Checks for NaN and ±inf.

    Args:
        data: (N, D) numpy array.

    Returns:
        (N, D) bool mask, True where value is missing.
    """
    mask = np.isnan(data) | np.isinf(data)
    if mask.any():
        count = mask.sum()
        total = mask.size
        import logging
        logging.getLogger("cdfm").info(
            f"Auto-detected {count}/{total} ({100 * count / total:.1f}%) missing values"
        )
    return mask


def preprocess(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Full preprocessing pipeline: encode → detect missing → standardize.

    Handles categorical (string) columns, NaN/inf detection, and z-score
    standardization in one call.

    Args:
        data: (N, D) numpy array, possibly with string columns or NaN.

    Returns:
        (data_std, missing_mask, mean, std) — data_std is (N, D) float32
        with 0 at missing positions; missing_mask is (N, D) bool or None;
        mean/std are (D,) float32 for unstandardization.
    """
    data = np.asarray(data)
    N, D = data.shape

    # 1. Detect NaN/inf per column (robust to mixed types)
    mask = np.zeros((N, D), dtype=bool)
    for j in range(D):
        col = data[:, j]
        try:
            fcol = col.astype(np.float64)
            mask[:, j] = np.isnan(fcol) | np.isinf(fcol)
        except (ValueError, TypeError):
            pass

    # 2. Encode categorical columns (string → int)
    encoded, _ = encode_categorical(data)
    mask |= auto_detect_missing(encoded)
    if not mask.any():
        mask = None

    # 3. Zero-fill and standardize
    clean = encoded.copy()
    if mask is not None:
        clean[mask] = 0.0
    data_std, mean, std = standardize(clean, missing_mask=mask)
    if mask is not None:
        data_std[mask] = 0.0

    return data_std.astype(np.float32), mask, mean, std


def create_input_tensor(
    data_std: np.ndarray,
    missing_mask: np.ndarray | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Package standardized data into model input format.

    The model expects (batch_size, max_n, max_d, 2) with channels:
        [:, :, :, 0] = standardized values
        [:, :, :, 1] = missing mask (1.0 = missing, 0.0 = observed)

    Args:
        data_std: (N, D) float array of standardized values (no NaN).
        missing_mask: (N, D) bool array, True = missing. If None, all zeros.
        device: Target torch device.

    Returns:
        X_t:   (1, N, D, 2) float tensor on device.
        n_vec: (1,) int64 tensor = [N].
        d_vec: (1,) int64 tensor = [D].

    Raises:
        ValueError: If N=0 or D=0.
    """
    N, D = data_std.shape
    if N == 0 or D == 0:
        raise ValueError(f"Data must have N>0 and D>0, got (N={N}, D={D})")

    X = np.zeros((1, N, D, 2), dtype=np.float32)
    X[0, :, :, 0] = data_std
    if missing_mask is not None:
        X[0, :, :, 1] = missing_mask.astype(np.float32)

    X_t = torch.from_numpy(X).to(device)
    n_vec = torch.tensor([N], dtype=torch.int64, device=device)
    d_vec = torch.tensor([D], dtype=torch.int64, device=device)

    return X_t, n_vec, d_vec
