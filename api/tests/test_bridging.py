import itertools
import unittest

import numpy as np

from bridging import pairwise_disagreement_scores


def paper_pairwise_disagreement(approvals: np.ndarray, x: int) -> float:
    """Theorem 6 of the paper on a fully observed 0/1 matrix: over ordered
    pairs of approvers of x, the mean Hamming disagreement, normalised by n^2."""
    n, m = approvals.shape
    approvers = np.flatnonzero(approvals[:, x] == 1)
    total = sum(
        np.mean(approvals[i] != approvals[j]) for i in approvers for j in approvers
    )
    return total / (n * n)


class PairwiseDisagreementTests(unittest.TestCase):
    def test_equals_the_expected_paper_score_by_exhaustive_enumeration(self):
        """Every 0/1 outcome of a 3x4 matrix, weighted by its probability. The
        closed form must equal the expectation of the paper's exact score."""
        rng = np.random.default_rng(0)
        p = rng.uniform(size=(3, 4))
        p[0, 1] = 1.0  # observed votes mixed in
        p[2, 3] = 0.0
        cells = p.size
        expected = np.zeros(4)
        for bits in itertools.product((0, 1), repeat=cells):
            outcome = np.asarray(bits).reshape(p.shape)
            weight = np.prod(np.where(outcome == 1, p, 1.0 - p))
            if weight == 0.0:
                continue
            expected += weight * np.asarray(
                [paper_pairwise_disagreement(outcome, x) for x in range(4)]
            )
        np.testing.assert_allclose(
            pairwise_disagreement_scores(p, range(4)), expected, atol=1e-12
        )

    def test_reduces_to_the_paper_score_on_observed_votes(self):
        approvals = np.asarray([
            [1, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 1],
        ], dtype=float)
        for x in range(4):
            self.assertAlmostEqual(
                pairwise_disagreement_scores(approvals, [x])[0],
                paper_pairwise_disagreement(approvals, x),
            )

    def test_hand_worked_two_voter_example(self):
        # Both approve comment 0 and disagree on both other comments:
        # 2 ordered pairs x (2 of 3 comments differ) / n^2 = 2 * (2/3) / 4.
        approvals = np.asarray([[1, 1, 0], [1, 0, 1]], dtype=float)
        self.assertAlmostEqual(pairwise_disagreement_scores(approvals, [0])[0], 1 / 3)

    def test_a_comment_approved_by_agreeing_voters_scores_zero(self):
        approvals = np.asarray([[1, 1, 0], [1, 1, 0]], dtype=float)
        self.assertEqual(pairwise_disagreement_scores(approvals, [0])[0], 0.0)

    def test_prefers_the_comment_that_bridges(self):
        """Voters 0-1 and 2-3 are opposed camps. Comment 0 is approved by one
        camp only; comment 1 by one voter from each camp."""
        approvals = np.asarray([
            [1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
        ], dtype=float)
        scores = pairwise_disagreement_scores(approvals, [0, 1])
        self.assertGreater(scores[1], scores[0])

    def test_needs_two_voters(self):
        with self.assertRaises(ValueError):
            pairwise_disagreement_scores(np.ones((1, 3)), [0])


if __name__ == "__main__":
    unittest.main()
