"""
Statistical significance tests for model comparison.

Provides paired permutation test for ablation studies
and bootstrap CI utility.
"""

import numpy as np
from typing import Dict, List, Callable


def permutation_test(
    scores_a: List[float],
    scores_b: List[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Per-conversation paired permutation test.

    H0: Model A and Model B have the same performance.
    H1: They differ (two-sided).

    Args:
        scores_a: Per-conversation metric values for Model A.
        scores_b: Per-conversation metric values for Model B.
        n_permutations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        Dict with observed_diff, p_value, and significance flags.
    """
    assert len(scores_a) == len(scores_b), (
        f"Score arrays must have same length: {len(scores_a)} vs {len(scores_b)}"
    )

    rng = np.random.RandomState(seed)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    observed_diff = float(np.mean(scores_a) - np.mean(scores_b))
    diffs = scores_a - scores_b

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = np.mean(diffs * signs)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations

    return {
        "observed_diff": round(observed_diff, 4),
        "p_value": round(p_value, 4),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
        "n_conversations": len(scores_a),
        "n_permutations": n_permutations,
    }
