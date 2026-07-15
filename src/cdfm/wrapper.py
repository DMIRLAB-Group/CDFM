"""
Causal Discovery Foundation Model (CDFM) for zero-shot causal discovery.
"""

import logging
import copy
import time
import numpy as np
import torch
from torch import nn

from cdfm.checkpoint import load_checkpoint
from cdfm.inference import predict as _predict, CDFMResult
from cdfm.preprocessing import encode_categorical, auto_detect_missing

logger = logging.getLogger("cdfm")


class CDFM:
    """Causal Discovery Foundation Model (CDFM) for zero-shot causal discovery.

    Loads a pretrained checkpoint and provides a predict() interface.
    Takes observational data X (N, D) and predicts the causal graph
    G (D, D) as edge logits / probabilities / adjacency.

    Usage::
        # From HuggingFace Hub
        model = CDFM.from_pretrained("DMIRLAB/CDFM")
        # From local checkpoint
        model = CDFM.from_pretrained("./checkpoint/")

        result = model.predict(data)                     # (N, D) → CDFMResult
        print(result.adjacency)                          # (D, D) binary graph

        # Override threshold
        result = model.predict(data, threshold=0.5)

    Args:
        checkpoint: Path to a directory (config.json + model.safetensors)
                    or HF Hub repo ID.
        device: 'auto' (cuda if available), 'cuda:N', 'cpu', or torch.device.
        threshold: Probability threshold in [0, 1].  None = auto-calibrate
                   using the model's stored calibration coefficients.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str | torch.device = "auto",
        threshold: float | None = None,
    ):
        self._checkpoint_path = checkpoint
        self._device = self._resolve_device(device)
        self._threshold = threshold

        # Lazy-loaded
        self._model: nn.Module | None = None
        self._model_cfg: dict = {}
        self._calibration: dict | None = None  # {type, coefficients, feature_names}

    @property
    def info(self) -> dict:
        """Model metadata. Triggers lazy loading on first access."""
        self._ensure_loaded()
        info = {
            "architecture": type(self._model).__name__,
            "parameters": sum(p.numel() for p in self._model.parameters()),
            "device": str(self._device),
            "threshold": self._threshold,
            "config": self._model_cfg,
        }
        if self._calibration is not None:
            info["calibration"] = {
                "type": self._calibration["type"],
                "n_features": len(self._calibration["feature_names"]),
            }
        return info

    @property
    def calibration_type_(self) -> str | None:
        """Calibration method type (e.g. 'linear')."""
        self._ensure_loaded()
        if self._calibration is not None:
            return self._calibration["type"]
        return None

    @property
    def calibration_coefficients_(self) -> np.ndarray | None:
        """Calibration coefficients (intercept + feature weights)."""
        self._ensure_loaded()
        if self._calibration is not None:
            return self._calibration["coefficients"]
        return None

    @property
    def calibration_feature_names_(self) -> list[str] | None:
        """Calibration feature names."""
        self._ensure_loaded()
        if self._calibration is not None:
            return self._calibration["feature_names"]
        return None

    def predict(
        self,
        data: np.ndarray,
        threshold: float | None = None,
        standardize: bool = True,
        missing_mask: np.ndarray | None = None,
    ) -> CDFMResult:
        """Predict the causal graph from observational data.

        Automatically encodes categorical (string) columns, detects and
        handles missing values, and standardizes before inference.

        Args:
            data: (N, D) numpy array (numeric or mixed with string columns).
            threshold: Probability threshold in [0, 1].
                       None = auto-calibrate via :func:`suggest_threshold`.
            standardize: Z-score standardize columns before inference.
            missing_mask: (N, D) bool mask, True = missing (optional;
                          auto-detected from NaN/inf if not provided).

        Returns:
            CDFMResult with logits, probabilities, adjacency, and metadata.
        """
        self._ensure_loaded()

        thresh = threshold if threshold is not None else self._threshold
        t0 = time.perf_counter()

        # Preprocess if needed; skip if user provides pre-standardized data + mask
        if standardize:
            from cdfm.preprocessing import preprocess
            data_std, auto_mask, _, _ = preprocess(np.asarray(data))
            mask = missing_mask if missing_mask is not None else auto_mask
        else:
            data_std = data
            mask = missing_mask

        result = _predict(
            self._model, data_std, self._device,
            missing_mask=mask,
        )

        # Binarize into adjacency
        if thresh is None:
            from cdfm.utils import suggest_threshold
            thresh = suggest_threshold(
                result.logits,
                weights=self.calibration_coefficients_,
                features=self.calibration_feature_names_,
            )


        probs = 1.0 / (1.0 + np.exp(-np.clip(result.logits, -50.0, 50.0)))
        adj = (probs > thresh).astype(np.int8)
        np.fill_diagonal(adj, 0)
        result = CDFMResult(
            logits=result.logits,
            probabilities=probs,
            adjacency=adj,
            threshold=thresh,
            runtime_sec=result.runtime_sec,
        )

        result.runtime_sec = time.perf_counter() - t0

        return result

    def imputation(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """Impute missing values using the model's built-in imputation head.

        Single forward pass → median quantile prediction at each missing
        position.  Handles categorical encoding and NaN detection automatically.

        Args:
            data: (N, D) numpy array, possibly with string columns or NaN.

        Returns:
            (N, D) float64 array with imputed values at missing positions.
        """
        self._ensure_loaded()
        from cdfm.preprocessing import preprocess, create_input_tensor
        from cdfm.inference import single_inference_with_imputation

        data_std, mask, mean, std = preprocess(np.asarray(data))
        if mask is None:
            return data_std.astype(np.float64)

        X_t, n_vec, d_vec = create_input_tensor(data_std, missing_mask=mask, device=self._device)
        _, mp_np = single_inference_with_imputation(self._model, X_t, n_vec, d_vec)

        if mp_np is None or mp_np.shape[0] == 0:
            import logging
            logging.getLogger("cdfm").warning("No imputation predictions, falling back to column mean")
            result = data_std.copy()
            for j in range(data_std.shape[1]):
                col = data_std[~mask[:, j], j]
                result[mask[:, j], j] = col.mean() if len(col) > 0 else 0.0
            return (result * std + mean).astype(np.float64)

        _MEDIAN_QUANTILE_IDX = 10
        median_preds = mp_np[:, _MEDIAN_QUANTILE_IDX]
        result_std = data_std.copy()
        result_std.ravel()[mask.ravel()] = median_preds
        return (result_std * std + mean).astype(np.float64)

    # ── HuggingFace Hub API ──────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "DMIRLAB/CDFM",
        device: str | torch.device = "auto",
        threshold: float | None = None,
        **kwargs,
    ) -> "CDFM":
        """Load from a local directory or HuggingFace Hub.

        Args:
            pretrained_model_name_or_path: Local directory or HF Hub repo ID
                (default ``"DMIRLAB/CDFM"``).
            device: ``"auto"``, ``"cuda:N"``, ``"cpu"``, or ``torch.device``.
            threshold: Probability threshold (None = auto-calibrate).

        Returns:
            CDFM with loaded weights and calibration.

        Example:
            >>> model = CDFM.from_pretrained("DMIRLAB/CDFM")
            >>> model = CDFM.from_pretrained("./checkpoint/")
        """
        return cls(
            checkpoint=pretrained_model_name_or_path,
            device=device,
            threshold=threshold,
        )

    def _ensure_loaded(self):
        if self._model is None:
            if self._checkpoint_path is None:
                raise ValueError(
                    "No checkpoint specified. Use from_pretrained('DMIRLAB/CDFM') "
                    "or pass a local path."
                )
            logger.info(f"Loading CDFM from {self._checkpoint_path}")
            self._model, self._model_cfg, self._calibration = load_checkpoint(
                self._checkpoint_path, self._device
            )

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "not loaded"
        path = self._checkpoint_path or "(not loaded)"
        t = f"{self._threshold:.2f}" if self._threshold is not None else "auto"
        return (
            f"CDFM(device={self._device}, "
            f"threshold={t}, checkpoint={path}, status={loaded})"
        )
