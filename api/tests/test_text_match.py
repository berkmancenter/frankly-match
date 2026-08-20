import unittest
from itertools import combinations

import numpy as np

from embedding_client import EmbeddingServiceError
from text_match import (
    FALLBACK_STATEMENT,
    TARGET_PULL_IN,
    TextMatchingService,
    cosine_distance_matrix,
    estimate_diversity_bounds,
    high_arm_target,
    optimize_diversity_groups,
    plan_arm_counts,
    plan_diversity_targets,
    plan_group_sizes,
    pool_mean_distance,
    select_diffusion_statement,
)


class QueueEmbeddingClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def embed(self, sentences):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class GroupSizePlanningTests(unittest.TestCase):
    def test_prioritizes_exact_sizes_without_singletons(self):
        self.assertEqual(plan_group_sizes(11, 5), [5, 6])
        self.assertEqual(plan_group_sizes(13, 5), [5, 5, 3])
        self.assertEqual(plan_group_sizes(14, 5), [5, 5, 4])

    def test_uses_three_person_groups_to_reduce_tied_deviation(self):
        self.assertEqual(plan_group_sizes(10, 4), [4, 3, 3])

    def test_rejects_groups_smaller_than_three(self):
        with self.assertRaisesRegex(ValueError, "at least 3 participants"):
            plan_group_sizes(2, 5)
        with self.assertRaisesRegex(ValueError, "targetGroupSize"):
            plan_group_sizes(5, 2)


def _distances(count: int, dimensions: int = 8, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cosine_distance_matrix(rng.normal(size=(count, dimensions)))


class DiversityTargetTests(unittest.TestCase):
    def test_always_plans_three_levels(self):
        """No silent switch to five levels at any event size."""
        distances = _distances(60)
        for group_count in (4, 12, 40):
            sizes = [5] * group_count
            _, levels = plan_diversity_targets(sizes, distances, seed=8)
            self.assertEqual(set(levels), {"low", "medium", "high"})

    def test_arms_are_balanced_thirds(self):
        """Balanced allocation: D-optimal for a quadratic dose-response, minimax
        under unknown per-arm variance, and no arm is one fallback from empty."""
        distances = _distances(60)
        sizes = [5] * 20
        _, levels = plan_diversity_targets(sizes, distances, seed=8)

        self.assertEqual(levels.count("low"), 7)
        self.assertEqual(levels.count("medium"), 6)
        self.assertEqual(levels.count("high"), 7)

    def test_arm_counts_never_differ_by_more_than_one(self):
        for group_count in range(3, 101):
            low, middle, high = plan_arm_counts(group_count)
            with self.subTest(groups=group_count):
                self.assertEqual(low + middle + high, group_count)
                self.assertLessEqual(max(low, middle, high) - min(low, middle, high), 1)
                self.assertGreaterEqual(min(low, middle, high), 1)
                # extras go to the extreme arms, never the middle
                self.assertGreaterEqual(low, middle)
                self.assertGreaterEqual(high, middle)

    def test_arm_counts_reject_events_too_small_for_three_levels(self):
        with self.assertRaisesRegex(ValueError, "at least 3 groups"):
            plan_arm_counts(2)

    def test_high_target_takes_exactly_what_the_low_arm_frees(self):
        # Two low groups each 0.4 below the mean free 0.8 for one high group.
        self.assertAlmostEqual(
            high_arm_target(1.0, [0.6, 0.6], high_count=1), 1.8
        )
        # Same slack shared between two high groups reaches half as far.
        self.assertAlmostEqual(
            high_arm_target(1.0, [0.6, 0.6], high_count=2), 1.4
        )

    def test_high_target_reduces_to_the_familiar_form_at_a_zero_floor(self):
        """H = mu * (1 + n_low / n_high) when the low arm sits at zero."""
        for low_count, high_count in ((1, 1), (3, 1), (5, 2)):
            with self.subTest(low=low_count, high=high_count):
                self.assertAlmostEqual(
                    high_arm_target(2.0, [0.0] * low_count, high_count),
                    2.0 * (1 + low_count / high_count),
                )

    def test_targets_respect_the_budget_constraint(self):
        """Mean of the assigned targets must not exceed the pool mean."""
        distances = _distances(60)
        sizes = [5] * 12
        targets, _ = plan_diversity_targets(sizes, distances, seed=8)

        self.assertLessEqual(
            float(np.mean(targets)),
            pool_mean_distance(distances) + 1e-9,
        )

    def test_high_target_is_budget_capped_by_the_pulled_in_ceiling(self):
        distances = _distances(60)
        sizes = [5] * 20
        targets, levels = plan_diversity_targets(sizes, distances, seed=8)
        bounds = estimate_diversity_bounds(distances, sizes, seed=8)
        mean_diversity = pool_mean_distance(distances)
        floor, ceiling = bounds[5]
        margin = TARGET_PULL_IN * (ceiling - floor)

        low = [t for t, level in zip(targets, levels) if level == "low"]
        high = {t for t, level in zip(targets, levels) if level == "high"}
        expected = (
            len(sizes) * mean_diversity
            - sum(low)
            - levels.count("medium") * mean_diversity
        ) / levels.count("high")

        self.assertEqual(len(high), 1)
        self.assertAlmostEqual(
            high.pop(), min(expected, ceiling - margin)
        )

    def test_extreme_targets_leave_symmetric_slack(self):
        """Targeting the exact floor or ceiling makes assignment error
        one-sided; the pull-in keeps both extremes strictly inside bounds."""
        distances = _distances(60)
        sizes = [5] * 20
        targets, levels = plan_diversity_targets(sizes, distances, seed=8)
        bounds = estimate_diversity_bounds(distances, sizes, seed=8)
        floor, ceiling = bounds[5]

        lows = [t for t, level in zip(targets, levels) if level == "low"]
        highs = [t for t, level in zip(targets, levels) if level == "high"]
        self.assertTrue(all(t > floor for t in lows))
        self.assertTrue(all(t < ceiling for t in highs))
        margin = TARGET_PULL_IN * (ceiling - floor)
        for t in lows:
            self.assertAlmostEqual(t, floor + margin)

    def test_targets_are_raw_distances_not_normalized(self):
        distances = _distances(60)
        sizes = [5] * 12
        targets, _ = plan_diversity_targets(sizes, distances, seed=8)

        self.assertNotIn(0.0, targets)
        self.assertNotIn(1.0, targets)
        self.assertIn(pool_mean_distance(distances), targets)

    def test_falls_back_to_uniform_targets_below_three_groups(self):
        distances = _distances(20)
        for group_count in (1, 2):
            sizes = [5] * group_count
            targets, levels = plan_diversity_targets(sizes, distances, seed=8)
            self.assertEqual(levels, ["medium"] * group_count)
            self.assertEqual(
                targets, [pool_mean_distance(distances)] * group_count
            )


class DiversityOptimizationTests(unittest.TestCase):
    def test_estimates_exact_bounds_for_small_instances(self):
        embeddings = np.random.default_rng(4).normal(size=(8, 5))
        distances = cosine_distance_matrix(embeddings)
        brute_force_scores = [
            np.mean(
                [
                    distances[first, second]
                    for first, second in combinations(group, 2)
                ]
            )
            for group in combinations(range(8), 3)
        ]

        bounds = estimate_diversity_bounds(
            distances,
            [3],
            seed=12,
        )

        np.testing.assert_allclose(
            bounds[3],
            (min(brute_force_scores), max(brute_force_scores)),
        )

    def test_preserves_sizes_and_targets_numeric_diversity(self):
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(18, 8))
        participant_ids = [f"p{index}" for index in range(18)]
        group_sizes = [5, 5, 5, 3]

        initial = optimize_diversity_groups(
            participant_ids,
            embeddings,
            group_sizes,
            seed=12,
            time_limit_seconds=0,
            max_restarts=1,
        )
        result = optimize_diversity_groups(
            participant_ids,
            embeddings,
            group_sizes,
            seed=12,
            time_limit_seconds=1.0,
            max_restarts=1,
            iterations_per_restart=500,
        )

        self.assertEqual([len(group) for group in result.groups], group_sizes)
        self.assertCountEqual(
            [
                participant_id
                for group in result.groups
                for participant_id in group
            ],
            participant_ids,
        )
        self.assertEqual(result.diversity_levels.count("low"), 2)
        self.assertEqual(result.diversity_levels.count("medium"), 1)
        self.assertEqual(result.diversity_levels.count("high"), 1)

        by_target = sorted(
            zip(result.assigned_targets, result.achieved_diversities)
        )
        self.assertLessEqual(by_target[0][1], by_target[-1][1])

    def test_cosine_distance_normalizes_input(self):
        distances = cosine_distance_matrix(
            np.asarray([[2.0, 0.0], [0.0, 3.0], [-4.0, 0.0]])
        )

        np.testing.assert_allclose(
            distances,
            np.asarray(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 0.0, 1.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
        )

    def test_selects_statement_with_best_minimum_distance(self):
        group_embeddings = np.asarray([[1.0, 0.0], [0.8, 0.2]])
        statement_embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        )

        selected = select_diffusion_statement(
            group_embeddings,
            statement_embeddings,
            ["near", "different", "opposite"],
        )

        self.assertEqual(selected, "opposite")


class TextMatchingServiceTests(unittest.TestCase):
    def test_returns_diffusion_statement_for_each_group(self):
        participant_embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.7, 0.7],
                [0.0, 1.0],
                [-0.7, 0.7],
                [-1.0, 0.0],
                [-0.7, -0.7],
            ]
        )
        diffusion_embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        )
        service = TextMatchingService(
            embedding_client=QueueEmbeddingClient(
                [participant_embeddings, diffusion_embeddings]
            ),
            diffusion_statements=("first", "second", "third"),
            optimization_seconds=0,
        )

        groups = service.match(
            {f"p{index}": f"response {index}" for index in range(6)},
            3,
        )

        self.assertEqual([len(group.participant_ids) for group in groups], [3, 3])
        self.assertTrue(all(not group.fallback_used for group in groups))
        # Two groups cannot identify three levels, so both are targeted at the
        # pool mean rather than silently running a two-level design.
        self.assertTrue(all(group.diversity_level == "medium" for group in groups))
        self.assertTrue(
            all(group.achieved_diversity is not None for group in groups)
        )
        self.assertTrue(
            all(
                group.diffusion_statement in {"first", "second", "third"}
                for group in groups
            )
        )

    def test_participant_embedding_failure_uses_random_fallback(self):
        service = TextMatchingService(
            embedding_client=QueueEmbeddingClient(
                [EmbeddingServiceError("offline")]
            ),
            optimization_seconds=0,
        )

        groups = service.match(
            {f"p{index}": f"response {index}" for index in range(11)},
            5,
        )

        self.assertEqual([len(group.participant_ids) for group in groups], [5, 6])
        self.assertTrue(all(group.fallback_used for group in groups))
        self.assertTrue(
            all(group.diversity_level == "unknown" for group in groups)
        )
        # No embeddings means no distance matrix, so nothing numeric is reported.
        self.assertTrue(
            all(group.achieved_diversity is None for group in groups)
        )
        self.assertTrue(all(group.assigned_target is None for group in groups))
        self.assertTrue(
            all(
                group.diffusion_statement == FALLBACK_STATEMENT
                for group in groups
            )
        )

    def test_statement_embedding_failure_preserves_optimized_groups(self):
        participant_embeddings = np.eye(6)
        service = TextMatchingService(
            embedding_client=QueueEmbeddingClient(
                [
                    participant_embeddings,
                    EmbeddingServiceError("offline"),
                ]
            ),
            optimization_seconds=0,
        )

        groups = service.match(
            {f"p{index}": f"response {index}" for index in range(6)},
            3,
        )

        self.assertTrue(all(group.fallback_used for group in groups))
        self.assertTrue(
            all(group.diversity_level != "unknown" for group in groups)
        )
        # Only statement selection failed, so the optimized groups and all their
        # diversity measurements survive.
        self.assertTrue(
            all(group.achieved_diversity is not None for group in groups)
        )
        self.assertTrue(
            all(
                group.diffusion_statement == FALLBACK_STATEMENT
                for group in groups
            )
        )


if __name__ == "__main__":
    unittest.main()
