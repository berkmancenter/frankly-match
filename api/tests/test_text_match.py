import unittest
from itertools import combinations

import numpy as np

from embedding_client import EmbeddingServiceError
from text_match import (
    FALLBACK_STATEMENT,
    TextMatchingService,
    cosine_distance_matrix,
    estimate_diversity_bounds,
    optimize_diversity_groups,
    plan_diversity_targets,
    plan_group_sizes,
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


class DiversityOptimizationTests(unittest.TestCase):
    def test_plans_three_or_five_adaptive_target_levels(self):
        small_event_targets = plan_diversity_targets(39, seed=8)
        large_event_targets = plan_diversity_targets(40, seed=8)

        self.assertEqual(
            {target: small_event_targets.count(target) for target in set(small_event_targets)},
            {0.0: 13, 0.5: 13, 1.0: 13},
        )
        self.assertEqual(
            {target: large_event_targets.count(target) for target in set(large_event_targets)},
            {0.0: 8, 0.25: 8, 0.5: 8, 0.75: 8, 1.0: 8},
        )

    def test_assigns_extra_target_slots_from_the_center_out(self):
        targets = plan_diversity_targets(4, seed=3)

        self.assertEqual(targets.count(0.0), 1)
        self.assertEqual(targets.count(0.5), 2)
        self.assertEqual(targets.count(1.0), 1)

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
        self.assertCountEqual(result.assigned_targets, [0.0, 0.5, 0.5, 1.0])
        self.assertEqual(result.diversity_levels.count("low"), 1)
        self.assertEqual(result.diversity_levels.count("medium"), 2)
        self.assertEqual(result.diversity_levels.count("high"), 1)
        self.assertTrue(
            all(
                0.0 <= score <= 1.0
                for score in result.normalized_achieved_diversities
            )
        )

        by_target = sorted(
            zip(
                result.assigned_targets,
                result.normalized_achieved_diversities,
            )
        )
        self.assertLessEqual(by_target[0][1], by_target[-1][1])

        initial_error = np.mean(
            np.square(
                np.asarray(initial.normalized_achieved_diversities)
                - np.asarray(initial.assigned_targets)
            )
        )
        optimized_error = np.mean(
            np.square(
                np.asarray(result.normalized_achieved_diversities)
                - np.asarray(result.assigned_targets)
            )
        )
        self.assertLessEqual(optimized_error, initial_error)

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
        self.assertCountEqual(
            [group.assigned_target for group in groups],
            [0.0, 1.0],
        )
        self.assertTrue(
            all(group.achieved_diversity is not None for group in groups)
        )
        self.assertTrue(
            all(
                group.normalized_achieved_diversity is not None
                for group in groups
            )
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
        self.assertTrue(
            all(group.achieved_diversity is None for group in groups)
        )
        self.assertTrue(
            all(
                group.normalized_achieved_diversity is None
                for group in groups
            )
        )
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
