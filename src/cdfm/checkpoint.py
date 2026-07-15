"""Checkpoint loading for the CDFM model.

Supports:
  - HF Hub: ``from_pretrained("DMIRLAB/CDFM")`` via huggingface_hub
  - Local directory: config.json + model.safetensors
"""

import os
import json
import logging
import numpy as np
import torch
from torch import nn

logger = logging.getLogger("cdfm")

def _load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def _load_weights_safetensors(safetensors_path: str, device: torch.device) -> dict:
    import safetensors.torch
    return safetensors.torch.load_file(safetensors_path, device=str(device))


def load_checkpoint(
    checkpoint_path: str | None = None,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict, dict | None]:
    """Load CDFM model from checkpoint.

    Supports:
      - HF Hub ID: ``"username/my-cdfm-model"`` → downloads via huggingface_hub
      - HF-style directory: contains config.json + model.safetensors
    Args:
        checkpoint_path: Path to a directory (config.json + model.safetensors)
                         or HF Hub repo ID (e.g. ``"DMIRLAB/CDFM"``).
        device: Target torch device.

    Returns:
        (model, model_cfg, calibration) tuple:
        - model: CDFM model in eval mode on device.
        - model_cfg: dict with architecture hyperparameters.
        - calibration: dict with ``weights`` (np.ndarray), ``features`` (list[str]),
          or None if not in config.
    """
    from cdfm.model.cdfm_model import CDFM as CDFMModel

    if isinstance(device, str):
        device = torch.device(device)

    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required (local dir or HF Hub repo id)")

    # ── HF Hub ID ──
    if (
        isinstance(checkpoint_path, str)
        and not os.path.exists(checkpoint_path)
        and "/" in checkpoint_path
        and not checkpoint_path.startswith((".", "/", "\\"))
    ):
        try:
            from huggingface_hub import snapshot_download
            logger.info(f"Downloading CDFM from HuggingFace Hub: {checkpoint_path}")
            checkpoint_path = snapshot_download(
                repo_id=checkpoint_path,
                allow_patterns=["config.json", "model.safetensors", "*.md"],
            )
            logger.info(f"  → cached at {checkpoint_path}")
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to load from HF Hub. "
                "Install with: pip install huggingface_hub"
            )

    checkpoint_path = os.path.abspath(checkpoint_path)

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = os.path.join(checkpoint_path, "config.json")
    weights_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Expected config.json + model.safetensors in {checkpoint_path}. "
            f"Contents: {os.listdir(checkpoint_path)[:20]}"
        )

    logger.info(f"Loading CDFM from {checkpoint_path}...")
    cfg = _load_config(config_path)
    state_dict = _load_weights_safetensors(weights_path, device)

    # ── Build model ──
    model = CDFMModel(
        embed_dim=cfg.get("embed_dim", 128),
        p_dim=cfg.get("p_dim", 64),
        nhead=cfg.get("nhead", 8),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        num_inds=cfg.get("num_inds", 128),
        dropout=cfg.get("dropout", 0.0),
        num_of_experts=cfg.get("num_of_experts", 0),
        num_layers=cfg.get("num_layers", 4),
        num_blocks_of_embedding=cfg.get("num_blocks_of_embedding", 3),
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # Load calibration config
    calibration = None
    cal_cfg = cfg.get("calibration_config")
    if cal_cfg:
        calibration = {
            "type": cal_cfg["type"],
            "coefficients": np.array(cal_cfg["coefficients"], dtype=np.float64),
            "feature_names": cal_cfg["feature_names"],
        }
        logger.info(f"Loaded calibration: {cal_cfg['type']}, {len(cal_cfg['feature_names'])} features")

    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded CDFM: {n_params:,} params")

    return model, cfg, calibration
