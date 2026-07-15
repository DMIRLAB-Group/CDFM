"""cdfm: Causal Discovery Foundation Model

Zero-shot causal discovery from observational data.

Quick start:
    from cdfm import CDFM

    model = CDFM.from_pretrained("DMIRLAB/CDFM")
    result = model.predict(data)                # data shape (N, D)
    print(result.adjacency)                     # (D, D) binary causal graph
"""

import os as _os
import sys as _sys

# Ensure package is importable when used without pip install
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)

from cdfm.wrapper import CDFM
from cdfm.inference import CDFMResult, single_inference, predict
from cdfm.checkpoint import load_checkpoint
from cdfm.preprocessing import standardize, create_input_tensor, auto_detect_missing, encode_categorical, preprocess
from cdfm.utils import (
    edge_logits_to_adjacency,
    adjacency_to_edge_list,
    summarize_graph,
    suggest_threshold,
    suggest_threshold_by_density,
    remove_cycles,
    structural_hamming_distance,
    edge_f1,
    evaluate_graph,
    edge_auroc,
)

__version__ = "0.3.0"
__all__ = [
    "CDFM",
    "CDFMResult",
    "single_inference",
    "predict",
    "load_checkpoint",
    "standardize",
    "create_input_tensor",
    "auto_detect_missing",
    "encode_categorical",
    "preprocess",
    "edge_logits_to_adjacency",
    "adjacency_to_edge_list",
    "summarize_graph",
    "suggest_threshold",
    "suggest_threshold_by_density",
    "remove_cycles",
    "structural_hamming_distance",
    "edge_f1",
    "evaluate_graph",
    "edge_auroc",
]
