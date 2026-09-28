"""Probabilistic pairwise-disagreement bridging (Blair et al., "The Structure of
Bridging", Sec. 5.3).

With fully observed approvals, the pairwise-disagreement score of comment x
sums, over every ordered pair of voters who both approve x, the fraction of
comments they disagree on, normalised by n^2 (Theorem 6: this equals B_PD).

Here approvals are probabilities: an observed vote is exactly 0 or 1, and an
unobserved one is a model's predicted approval. Treating each cell as an
independent Bernoulli draw, the score below is the exact expectation of that
sum:

    B(x) = 1/(n^2 m) * sum_{i != j} sum_{y != x}
               P[i,x] P[j,x] (P[i,y] + P[j,y] - 2 P[i,y] P[j,y])

P[i,x] P[j,x] is the chance both approve x, and the bracket the chance they
disagree on y. Two corrections to the naive "chance both approve times
expected Hamming distance": pairs i = j are excluded (a voter never disagrees
with themself, but the naive term would be positive for fractional cells),
and y = x is excluded (two approvers of x agree on x by definition).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pairwise_disagreement_scores(
    probabilities: np.ndarray, candidate_columns: Sequence[int]
) -> np.ndarray:
    """Expected pairwise-disagreement score of each candidate column.

    probabilities: voters x comments, each cell in [0, 1]. The voters are the
    whole population the score is measured over; every column is a comment
    disagreement is measured on.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    n, m = p.shape
    if n < 2:
        raise ValueError("pairwise disagreement needs at least two voters")

    # disagreement[i, j] = sum over all y of P(i and j differ on y).
    row_totals = p.sum(axis=1)
    disagreement = row_totals[:, None] + row_totals[None, :] - 2.0 * (p @ p.T)
    off_diagonal = ~np.eye(n, dtype=bool)

    scores = np.empty(len(candidate_columns))
    for k, x in enumerate(candidate_columns):
        approve = p[:, x]
        differ_on_x = approve[:, None] + approve[None, :] - 2.0 * np.outer(approve, approve)
        both_approve = np.outer(approve, approve)
        scores[k] = float(
            (both_approve * (disagreement - differ_on_x))[off_diagonal].sum()
        )
    return scores / (n * n * m)
