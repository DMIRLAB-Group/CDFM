"""Graph utility functions for CDFM."""

import numpy as np
from sklearn.metrics import roc_auc_score

def suggest_threshold(
    logits: np.ndarray,
    D: int | None = None,
    weights: np.ndarray | None = None,
    features: list[str] | None = None,
) -> float:
    """Graph-adaptive logistic threshold trained on 75,000 synthetic datasets.

        T = σ(w₀ + Σ wᵢ₊₁ · featureᵢ)

    where weights[0] is the intercept and weights[1:] match features.

    Args:
        logits: (D, D) raw edge logits or probabilities, zero diagonal.
        D: Number of variables.  If None, inferred from ``logits.shape[0]``.
        weights: Length 11 array (intercept + 10 coefs).
        features: Feature name list (length 10).

    Returns:
        Probability threshold in [0, 1].
    """
    if weights is None or features is None:
        raise ValueError(
            "suggest_threshold requires weights and features. "
            "Use model.predict(threshold=None) for auto-thresholding."
        )

    if D is None:
        D = logits.shape[0]

    mask = ~np.eye(D, dtype=bool)
    probs = logits[mask]
    if probs.min() < 0.0 or probs.max() > 1.0:
        probs = 1.0 / (1.0 + np.exp(-np.clip(probs, -50.0, 50.0)))

    ps = np.sort(probs)[::-1]
    n = len(ps)

    fv_map = {
        'mean': float(np.mean(probs)),
        'std': float(np.std(probs)),
        'p25': float(np.percentile(probs, 25)),
        'p50': float(np.percentile(probs, 50)),
        'p75': float(np.percentile(probs, 75)),
        'p90': float(np.percentile(probs, 90)),
        'logD': np.log(D),
    }

    gaps = ps[:-1] - ps[1:]
    fv_map['median_gap'] = float(np.median(gaps)) if len(gaps) > 0 else 0.0
    fv_map['max_gap'] = float(np.max(gaps)) if len(gaps) > 0 else 0.0

    denom = max(ps[0] - ps[-1], 1e-10)
    y_curve = (ps - ps[-1]) / denom
    x_axis = np.arange(n) / max(n - 1, 1)
    diff = y_curve - x_axis
    min_idx = max(1, min(int(0.5 * D), n - 1))
    knee_idx = min_idx + np.argmax(diff[min_idx:])
    fv_map['knee_T'] = float((ps[knee_idx] + ps[knee_idx + 1]) / 2) if knee_idx < n - 1 else float(ps[knee_idx])

    z = weights[0] + sum(float(weights[i + 1]) * fv_map[features[i]] for i in range(len(features)))
    return float(1.0 / (1.0 + np.exp(-z)))



def suggest_threshold_by_density(
    logits: np.ndarray,
    target_edges: int | None = None,
    density: float | None = None,
) -> float:
    """Suggest a probability threshold based on desired graph sparsity.

    Purely heuristic — no ground truth required. Sorts all off-diagonal
    edge probabilities and picks the threshold that yields approximately
    the target number of edges.

    Args:
        logits: (D, D) raw edge logits, zero diagonal.
        target_edges: Desired number of edges. If None, uses density.
        density: Desired edge density ∈ [0, 1], defined as edges / (D*(D-1)).
                 Default: 2/D (roughly 2 edges per variable, a common
                 heuristic for sparse DAGs).

    Returns:
        Probability threshold in [0, 1].

    Example:
        >>> result = model.predict(data)
        >>> t = suggest_threshold_by_density(result.logits, target_edges=20)
        >>> adj = edge_logits_to_adjacency(result.logits, threshold=t)
    """
    D = logits.shape[0]
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

    # Collect off-diagonal probabilities, sorted descending
    off_diag = []
    for i in range(D):
        for j in range(D):
            if i != j:
                off_diag.append(float(probs[i, j]))
    off_diag.sort(reverse=True)

    if target_edges is None:
        if density is None:
            density = 2.0 / D  # ~2 edges per variable
        max_edges = D * (D - 1)
        target_edges = max(1, int(density * max_edges))

    target_edges = min(target_edges, len(off_diag))

    if target_edges >= len(off_diag):
        return 0.0
    if target_edges <= 0:
        return 1.0

    # Threshold = probability of the target_edges-th highest edge
    t = off_diag[target_edges - 1]
    # Small margin to ensure we get exactly target_edges
    if target_edges < len(off_diag):
        t = (t + off_diag[target_edges]) / 2.0

    return float(max(0.0, min(1.0, t)))



# ═══════════════════════════════════════════════════════════════════════════
# Cycle removal
# ═══════════════════════════════════════════════════════════════════════════

def _find_cycle(adj: np.ndarray) -> list:
    """Find one cycle in a directed graph. Returns list of (i,j) edges."""
    import networkx as nx
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    try:
        cycle = nx.find_cycle(G, orientation="original")
        return [(u, v) for u, v, _ in cycle]
    except nx.NetworkXNoCycle:
        return []


def remove_cycles(
    probabilities: np.ndarray,
    adjacency: np.ndarray | None = None,
    threshold: float = 0.5,
    max_iter: int = 10000,
) -> np.ndarray:
    """Remove cycles by deleting the lowest-probability edge in each cycle.

    This is a post-processing step that ensures the output is a valid DAG
    (directed acyclic graph). Starting from a adjacency matrix,
    it repeatedly finds cycles and removes the edge with the lowest model
    confidence, until no cycles remain.

    Args:
        probabilities: (D, D) edge probabilities (sigmoid of logits),
            zero diagonal.
        adjacency: (D, D) initial binary adjacency. If None, uses
            ``probabilities > threshold``.
        threshold: Probability threshold for binarizing ``probabilities``
            when ``adjacency`` is not provided. Default 0.5.
        max_iter: Maximum number of cycle-breaking iterations (safety limit).

    Returns:
        (D, D) acyclic binary adjacency matrix (int8), zero diagonal.

    Example:
        >>> result = model.predict(data)
        >>> dag_adj = remove_cycles(result.probabilities)
        >>> # Or combine with kneedle threshold:
        >>> from CDFM import suggest_threshold
        >>> t = suggest_threshold(result.logits)
        >>> dag_adj = remove_cycles(result.probabilities, threshold=t)
    """
    if adjacency is None:
        D = probabilities.shape[0]
        adjacency = (probabilities > threshold).astype(np.int8)
        np.fill_diagonal(adjacency, 0)

    adj = adjacency.copy()
    n_removed = 0

    for _ in range(max_iter):
        cycle = _find_cycle(adj)
        if not cycle:
            break
        # Remove edge with lowest probability in this cycle
        min_edge = min(cycle, key=lambda e: probabilities[e[0], e[1]])
        adj[min_edge[0], min_edge[1]] = 0
        n_removed += 1

    return adj


def edge_logits_to_adjacency(
    logits: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert (D, D) edge logits to binary adjacency matrix.

    Args:
        logits: (D, D) edge logits/scores.
        threshold: Probability threshold in [0, 1].
                   sigmoid(logit) > threshold → edge exists.

    Returns:
        (D, D) int8 binary adjacency, zero diagonal.
    """
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    adj = (probs > threshold).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj


def adjacency_to_edge_list(
    adj: np.ndarray,
    var_names: list[str] | None = None,
) -> list[tuple[int, int, str, str]]:
    """Convert (D, D) binary adjacency to edge list.

    Args:
        adj: (D, D) binary adjacency.
        var_names: Optional list of variable names for labeling.

    Returns:
        List of (src_idx, tgt_idx, src_name, tgt_name) tuples.
    """
    D = adj.shape[0]
    edges = []
    for i in range(D):
        for j in range(D):
            if adj[i, j] and i != j:
                src_name = var_names[i] if var_names else str(i)
                tgt_name = var_names[j] if var_names else str(j)
                edges.append((i, j, src_name, tgt_name))
    return edges


def summarize_graph(
    adj: np.ndarray,
    var_names: list[str] | None = None,
    logits: np.ndarray | None = None,
    max_edges: int = 30,
) -> str:
    """Pretty-print a causal graph summary.

    Args:
        adj: (D, D) binary adjacency.
        var_names: Optional variable name list.
        logits: Optional (D, D) edge scores for ranking.
        max_edges: Max edges to list.

    Returns:
        Multi-line summary string.
    """
    D = adj.shape[0]
    n_edges = int(adj.sum())

    lines = [
        f"Causal Graph: {n_edges} edges, {D} variables",
        "-" * 50,
    ]

    edges = adjacency_to_edge_list(adj, var_names)
    if logits is not None:
        edges.sort(key=lambda e: logits[e[0], e[1]], reverse=True)

    shown = edges[:max_edges]
    for src, tgt, sname, tname in shown:
        score_str = ""
        if logits is not None:
            score_str = f" ({logits[src, tgt]:+.2f})"
        lines.append(f"  {sname} → {tname}{score_str}")

    if len(edges) > max_edges:
        lines.append(f"  ... and {len(edges) - max_edges} more edges")

    if not edges:
        lines.append("  (no edges)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Graph evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════


def structural_hamming_distance(
    learned: np.ndarray,
    true: np.ndarray,
) -> int:
    """Structural Hamming Distance (SHD) between two graph adjacency matrices.

    Counts the number of edge insertions, deletions, or reversals needed to
    transform one graph into the other.  Each *unordered pair* of nodes
    (i, j), j > i, contributes at most 1 to the distance, regardless of
    whether the discrepancy is a missing edge, extra edge, or direction reversal.

    Args:
        learned: (D, D) binary adjacency matrix (predicted).
        true:    (D, D) binary adjacency matrix (ground truth).

    Returns:
        SHD value (0 = identical, higher = more different).
    """
    D = learned.shape[0]
    shd = 0
    for i in range(D):
        for j in range(i + 1, D):
            if (learned[i, j], learned[j, i]) != (true[i, j], true[j, i]):
                shd += 1
    return shd


def edge_f1(
    learned: np.ndarray,
    true: np.ndarray,
) -> dict:
    """Compute F1 score, precision, and recall for directed edge predictions.

    Compares all off-diagonal positions.  A predicted edge at (i, j) matches
    the ground truth only if ``true[i, j] == 1``.

    Args:
        learned: (D, D) binary adjacency matrix (predicted), zero diagonal.
        true:    (D, D) binary adjacency matrix (ground truth), zero diagonal.

    Returns:
        dict with keys: ``f1``, ``precision``, ``recall``, ``tp``, ``fp``, ``fn``.
    """
    D = learned.shape[0]
    off_mask = ~np.eye(D, dtype=bool)
    pred_flat = learned[off_mask]
    true_flat = true[off_mask]

    tp = int((pred_flat & true_flat).sum())
    fp = int((pred_flat & ~true_flat).sum())
    fn = int((~pred_flat & true_flat).sum())

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-10)

    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_graph(
    learned: np.ndarray,
    true: np.ndarray,
) -> dict:
    """Evaluate a predicted graph against ground truth.

    Computes F1, precision, recall, and SHD in one call.

    Args:
        learned: (D, D) binary adjacency (predicted), zero diagonal.
        true:    (D, D) binary adjacency (ground truth), zero diagonal.

    Returns:
        dict with ``f1``, ``precision``, ``recall``, ``tp``, ``fp``, 
        ``fn``, ``shd``.
    """
    metrics = edge_f1(learned, true)
    metrics["shd"] = structural_hamming_distance(learned, true)
    return metrics


def edge_auroc(
    logits: np.ndarray,
    true: np.ndarray,
) -> float:
    """Area under the ROC curve for directed edge prediction.

    Args:
        logits: (D, D) edge logits, zero diagonal.
        true:  (D, D) ground truth adjacency, zero diagonal.

    Returns:
        AUROC in [0, 1].
    """
    D = true.shape[0]
    off_mask = ~np.eye(D, dtype=bool)
    scores = logits[off_mask].ravel()
    labels = true[off_mask].ravel()

    return float(roc_auc_score(labels, scores))


