import unittest
from itertools import combinations

import numpy as np

from embedding_client import EmbeddingServiceError
from text_match import (
    FALLBACK_STATEMENT,
    assign_sizes_to_arms,
    design_event,
    MEDIUM_ARM_WEIGHT,
    TARGET_PULL_IN,
    TextMatchingService,
    cosine_distance_matrix,
    estimate_diversity_bounds,
    plan_arm_counts,
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
    def test_reproduces_the_published_allocation_at_25_groups(self):
        self.assertEqual(plan_arm_counts(25), (7, 11, 7))

    def test_extreme_arms_stay_equal_so_contrasts_are_orthogonal(self):
        """Linear (-1, 0, 1) and quadratic (-1, 2, -1) contrasts are orthogonal
        iff sum(c1_i * c2_i / n_i) == 0, which holds exactly when the extreme
        arms are equal. The two registered hypotheses must stay independent."""
        for group_count in range(3, 101):
            low, middle, high = plan_arm_counts(group_count)
            with self.subTest(groups=group_count):
                self.assertEqual(low, high)
                self.assertEqual(low + middle + high, group_count)
                self.assertGreaterEqual(middle, 1)
                self.assertGreaterEqual(low, 1)
                orthogonality = 1.0 / low - 1.0 / high
                self.assertAlmostEqual(orthogonality, 0.0)

    def test_medium_arm_tracks_the_sqrt_two_weight_at_scale(self):
        for group_count in (30, 60, 100):
            low, middle, high = plan_arm_counts(group_count)
            with self.subTest(groups=group_count):
                self.assertAlmostEqual(middle / low, MEDIUM_ARM_WEIGHT, delta=0.12)

    def test_arm_counts_reject_events_too_small_for_three_levels(self):
        with self.assertRaisesRegex(ValueError, "at least 3 groups"):
            plan_arm_counts(2)

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



class EventDesignTests(unittest.TestCase):
    """The five-stage design: plan, randomise into pools, optimise endpoints
    minimax, then place the medium arm at the achieved midpoint."""

    def _pool(self, count=100, clusters=4, dimensions=48, seed=11):
        rng = np.random.default_rng(seed)
        centers = rng.normal(size=(clusters, dimensions))
        embeddings = np.vstack(
            [centers[i % clusters] + 0.8 * rng.normal(size=dimensions)
             for i in range(count)]
        )
        ids = [f"p{index}" for index in range(count)]
        return ids, cosine_distance_matrix(embeddings)

    def test_reproduces_the_specified_arm_and_pool_sizes(self):
        for participants, group_size, groups, people in (
            (100, 5, (6, 8, 6), (30, 40, 30)),
            (100, 4, (7, 11, 7), (28, 44, 28)),
            (98, 5, (6, 8, 6), (30, 38, 30)),
            (97, 5, (6, 7, 6), (30, 37, 30)),
        ):
            with self.subTest(participants=participants, group_size=group_size):
                sizes = plan_group_sizes(participants, group_size)
                by_arm = assign_sizes_to_arms(
                    sizes, group_size, plan_arm_counts(len(sizes))
                )
                order = ("low", "medium", "high")
                self.assertEqual(
                    tuple(len(by_arm[level]) for level in order), groups
                )
                self.assertEqual(
                    tuple(sum(by_arm[level]) for level in order), people
                )

    def test_ragged_group_sizes_go_to_the_medium_arm(self):
        """Achievable bounds depend on group size, so the extreme arms -- which
        carry the contrast -- are kept uniform."""
        sizes = plan_group_sizes(98, 5)
        self.assertIn(3, sizes)  # 98 = 19x5 + 3

        by_arm = assign_sizes_to_arms(sizes, 5, plan_arm_counts(len(sizes)))
        self.assertIn(3, by_arm["medium"])
        self.assertEqual(set(by_arm["low"]), {5})
        self.assertEqual(set(by_arm["high"]), {5})

    def test_every_event_gets_all_three_levels(self):
        """No silent switch to a different number of conditions at any size."""
        for group_count in (4, 12, 25, 40, 100):
            low, middle, high = plan_arm_counts(group_count)
            with self.subTest(groups=group_count):
                self.assertGreaterEqual(min(low, middle, high), 1)
                self.assertEqual(low + middle + high, group_count)

    def test_allocates_sqrt_two_to_one_to_one(self):
        """Medium arm is sqrt(2)x an extreme arm: minimises the larger marginal
        variance of the medium-versus-extreme contrasts, which the confirmatory
        curvature test depends on."""
        self.assertEqual(plan_arm_counts(20), (6, 8, 6))
        self.assertEqual(plan_arm_counts(25), (7, 11, 7))

    def test_pull_in_margin_is_symmetric_on_both_bounds(self):
        """TARGET_PULL_IN no longer sets targets -- minimax has none -- but it
        still defines the logged margin, and it must apply to both ends."""
        ids, distances = self._pool()
        design = design_event(ids, distances, 4, seed=7, time_limit_seconds=1)
        for arm in design.arms:
            with self.subTest(level=arm.level):
                self.assertAlmostEqual(
                    arm.margin, TARGET_PULL_IN * (arm.ceiling - arm.floor)
                )

    def test_pools_are_disjoint_and_cover_everyone(self):
        ids, distances = self._pool()
        design = design_event(ids, distances, 4, seed=7, time_limit_seconds=2)

        pooled = [pid for pool in design.pools.values() for pid in pool]
        self.assertEqual(sorted(pooled), sorted(ids))
        self.assertEqual(len(pooled), len(set(pooled)))

    def test_no_participant_crosses_pools_during_optimisation(self):
        """The optimiser may only rearrange within a pool. If it could move
        people between pools, assignment to condition would stop being random."""
        ids, distances = self._pool()
        design = design_event(ids, distances, 4, seed=7, time_limit_seconds=2)

        for arm in design.arms:
            assigned = {pid for group in arm.groups for pid in group}
            self.assertEqual(assigned, set(design.pools[arm.level]))

    def test_randomisation_is_reproducible_from_the_seed(self):
        ids, distances = self._pool()
        first = design_event(ids, distances, 4, seed=7, time_limit_seconds=1)
        second = design_event(ids, distances, 4, seed=7, time_limit_seconds=1)
        other = design_event(ids, distances, 4, seed=8, time_limit_seconds=1)

        self.assertEqual(first.pools, second.pools)
        self.assertNotEqual(first.pools, other.pools)

    def test_medium_target_is_the_midpoint_of_achieved_endpoints(self):
        """Equal spacing is what preserves beta_2 = -C/2 for the confirmatory
        contrast, so the midpoint must come from what the endpoints ACHIEVED."""
        ids, distances = self._pool()
        design = design_event(ids, distances, 4, seed=7, time_limit_seconds=2)

        low = next(a for a in design.arms if a.level == "low")
        high = next(a for a in design.arms if a.level == "high")
        self.assertAlmostEqual(design.endpoint_low, float(np.mean(low.achieved)))
        self.assertAlmostEqual(design.endpoint_high, float(np.mean(high.achieved)))
        self.assertAlmostEqual(
            design.medium_target,
            (design.endpoint_low + design.endpoint_high) / 2,
        )

    def test_endpoints_are_ordered_and_the_medium_arm_sits_between(self):
        ids, distances = self._pool()
        design = design_event(ids, distances, 4, seed=7, time_limit_seconds=3)
        means = {a.level: float(np.mean(a.achieved)) for a in design.arms}

        self.assertLess(means["low"], means["medium"])
        self.assertLess(means["medium"], means["high"])

    def test_bounds_are_computed_within_each_pool(self):
        """Not over the whole event: an arm's feasible range is set by the
        people randomised into it, not by the roster it was drawn from."""
        ids, distances = self._pool()
        design = design_event(ids, distances, 4, seed=7, time_limit_seconds=2)

        index_of = {pid: i for i, pid in enumerate(ids)}
        for arm in design.arms:
            picks = np.array([index_of[p] for p in design.pools[arm.level]])
            sub = distances[np.ix_(picks, picks)]
            expected = estimate_diversity_bounds(
                sub, arm.sizes, seed=7 ^ {"low": 0xA11, "high": 0xB22}.get(arm.level, 0xC33)
            )[max(set(arm.sizes), key=arm.sizes.count)]
            with self.subTest(level=arm.level):
                self.assertAlmostEqual(arm.floor, expected[0])
                self.assertAlmostEqual(arm.ceiling, expected[1])

    def test_minimax_holds_for_every_group_not_just_on_average(self):
        """An arm label is only meaningful if it holds group by group. MSE would
        let one group sit far off while others compensate."""
        ids, distances = self._pool()
        design = design_event(ids, distances, 4, seed=7, time_limit_seconds=3)
        low = next(a for a in design.arms if a.level == "low")
        high = next(a for a in design.arms if a.level == "high")
        medium = next(a for a in design.arms if a.level == "medium")

        self.assertLess(max(low.achieved), min(medium.achieved))
        self.assertGreater(min(high.achieved), max(medium.achieved))

    def test_convergence_is_reported_per_arm(self):
        ids, distances = self._pool()
        design = design_event(ids, distances, 4, seed=7, time_limit_seconds=3)

        for arm in design.arms:
            with self.subTest(level=arm.level):
                self.assertEqual(len(arm.restart_statistics), arm.restarts_used)
                self.assertGreaterEqual(arm.restart_spread, 0.0)
                self.assertIsInstance(arm.converged, bool)

    def test_below_three_groups_falls_back_to_a_single_level(self):
        ids, distances = self._pool(count=8)
        design = design_event(ids, distances, 4, seed=7, time_limit_seconds=1)

        self.assertEqual([a.level for a in design.arms], ["medium"])
        self.assertIsNone(design.arms_separated)
        self.assertAlmostEqual(design.medium_target, design.pool_mean)


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
