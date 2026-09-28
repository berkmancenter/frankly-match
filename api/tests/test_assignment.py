import itertools
import unittest

import numpy as np

from assignment import min_cost_assignment


def brute_force_minimum(cost: np.ndarray) -> float:
    n, m = cost.shape
    return min(
        sum(cost[row, column] for row, column in enumerate(columns))
        for columns in itertools.permutations(range(m), n)
    )


class MinCostAssignmentTests(unittest.TestCase):
    def assert_optimal(self, cost):
        assignment = min_cost_assignment(cost)
        self.assertEqual(len(set(assignment.tolist())), cost.shape[0])
        total = cost[np.arange(cost.shape[0]), assignment].sum()
        self.assertAlmostEqual(total, brute_force_minimum(cost))

    def test_matches_brute_force_on_random_square_and_rectangular_costs(self):
        rng = np.random.default_rng(0)
        for n, m in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (2, 5), (3, 6), (4, 7), (5, 8)]:
            for trial in range(40):
                with self.subTest(n=n, m=m, trial=trial):
                    self.assert_optimal(rng.uniform(size=(n, m)))

    def test_matches_brute_force_with_ties_and_integer_ranks(self):
        rng = np.random.default_rng(1)
        for trial in range(100):
            with self.subTest(trial=trial):
                self.assert_optimal(rng.integers(0, 3, size=(4, 6)).astype(float))

    def test_forbidden_pairings_are_avoided_when_possible(self):
        big = 10_000.0
        cost = np.asarray([[0.0, big, 5.0], [0.0, 1.0, big]])
        # Both rows prefer column 0; row 1's fallback is cheaper.
        np.testing.assert_array_equal(min_cost_assignment(cost), [0, 1])

    def test_rejects_more_rows_than_columns_and_non_finite_costs(self):
        with self.assertRaises(ValueError):
            min_cost_assignment(np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            min_cost_assignment(np.asarray([[np.inf, 0.0]]))


if __name__ == "__main__":
    unittest.main()
