import itertools
import unittest
import hashlib
import os
from unittest.mock import patch
from itertools import combinations

import json

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
    PreSurveyContext,
    fill_missing_text_distances,
    maximin_scores,
    assign_statements,
    format_statements,
    table_preferences,
)
from presurvey import ApprovalMatrix, Link, parse_comment_catalog


def _catalog(entries):
    """entries: (comment_id, topic_id, author_pid, text, [x, y])."""
    return parse_comment_catalog(json.dumps({
        "schema_version": 1,
        "topics": {"stocking_growing": {}, "food_access": {}},
        "embedding_metadata": {"dimensions": 2},
        "comments": {
            cid: {"comment_id": cid, "topic_id": topic, "author_pid": author,
                  "text": text, "embedding": vector}
            for cid, topic, author, text, vector in entries
        },
    }).encode())


STUB_CATALOG = _catalog([
    ("near", "stocking_growing", "pid_near", "near", [1.0, 0.0]),
    ("side", "stocking_growing", "pid_side", "side  \n\n comment", [0.0, 1.0]),
    ("far", "stocking_growing", "pid_far", "far", [-1.0, 0.0]),
    ("screened_out", "stocking_growing", "pid_x", "screened out", [-1.0, 0.0]),
    ("other_topic", "food_access", "pid_y", "other topic", [-1.0, 0.0]),
])
STUB_ELIGIBLE = frozenset({"near", "side", "far", "other_topic"})
STUB_RANKING = ("side", "near", "far")
# Columns in catalog order: near, side, far, screened_out, other_topic.
# Voters a and b disagree on everything except that both approve "side".
STUB_MATRIX = ApprovalMatrix(
    pids=("v_a", "v_b", "pid_far"),
    emails=("a", "b", "c"),
    names=("A", "B", "C"),
    probabilities=np.asarray([
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ]),
    source="stub",
    sha256="stub",
)
LINKED_AB = {"a": Link("a", "email", 0, "v_a"), "b": Link("b", "email", 1, "v_b")}


def _context(links=None, matrix=STUB_MATRIX, catalog=STUB_CATALOG):
    return PreSurveyContext(
        catalog=catalog, matrix=matrix, links=links if links is not None else LINKED_AB,
        candidate_rows=(0, 1, 2), fallback_rows=(1, 0, 2),
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

    def test_maximin_scores_each_candidate_by_its_nearest_member(self):
        members = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        candidates = np.asarray([[1.0, 0.0], [-1.0, 0.0], [-1.0, -1.0]])
        scores = maximin_scores(members, candidates)
        # [-1, -1] is 135 degrees from both members; [-1, 0] is only 90
        # degrees from the second.
        np.testing.assert_allclose(scores, [0.0, 1.0, 1.0 + 2 ** -0.5])

    def test_maximin_refuses_mismatched_dimensions(self):
        with self.assertRaises(ValueError):
            maximin_scores(np.ones((2, 3)), np.ones((2, 2)))



def _assign(context, tables):
    """tables: list of (member_ids, member_embeddings or None)."""
    preferences = [
        slot for t, (members, embeddings) in enumerate(tables)
        for slot in table_preferences(context, t, members, embeddings)
    ]
    return assign_statements(context, preferences)


class MissingTextTests(unittest.TestCase):
    def test_no_text_participants_sit_at_the_mean_real_distance(self):
        real = cosine_distance_matrix(np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
        distances, fill = fill_missing_text_distances(["a", "x", "b", "c", "y"], ["a", "b", "c"], real)
        known = [real[0, 1], real[0, 2], real[1, 2]]
        self.assertAlmostEqual(fill, np.mean(known))
        # x and y are at the fill distance from everyone, each other included.
        for other in range(5):
            if other not in (1,):
                self.assertAlmostEqual(distances[1, other], fill)
        self.assertAlmostEqual(distances[1, 4], fill)
        self.assertEqual(distances[1, 1], 0.0)
        np.testing.assert_allclose(distances[np.ix_([0, 2, 3], [0, 2, 3])], real)
        # The pool mean, which every target is built on, is unchanged.
        self.assertAlmostEqual(pool_mean_distance(distances), pool_mean_distance(real))

    def _service(self, embeddings):
        return TextMatchingService(
            embedding_client=QueueEmbeddingClient([embeddings]), optimization_seconds=0,
        )

    def test_only_real_text_is_embedded_and_no_text_is_logged(self):
        class Recording(QueueEmbeddingClient):
            def embed(self, sentences):
                self.sentences = list(sentences)
                return super().embed(sentences)

        client = Recording([np.random.default_rng(2).normal(size=(5, 2))])
        service = TextMatchingService(embedding_client=client, optimization_seconds=0)
        responses = {f"p{i}": f"text {i}" for i in range(5)}
        responses["q"] = None
        with patch("text_match.load_comment_catalog", return_value=STUB_CATALOG), \
                patch("text_match.load_eligible_comment_ids", return_value=STUB_ELIGIBLE), \
                patch("text_match.load_bridging_ranking", return_value=STUB_RANKING), \
                patch("text_match.log.log_event") as logged:
            groups = service.match(responses, 3)

        self.assertEqual(client.sentences, [f"text {i}" for i in range(5)])
        self.assertEqual(sorted(p for g in groups for p in g.participant_ids), sorted(responses))
        complete = next(
            call.kwargs["extra_data"] for call in logged.call_args_list
            if call.args[1] == "Participant distances complete"
        )
        self.assertEqual(complete["missing_text_participant_ids"], ["q"])
        self.assertIsNotNone(complete["missing_text_fill_distance"])

    def test_a_table_where_nobody_wrote_text_falls_back_for_maximin_only(self):
        context = _context()
        maximin, bridging = table_preferences(
            context, 0, ["a", "b"], None, missing_embeddings_reason="no_member_text",
        )
        self.assertEqual(maximin.fallback_reason, "no_member_text")
        self.assertIsNone(bridging.fallback_reason)

    def test_fewer_than_two_texts_means_random_groups_with_real_statements(self):
        service = self._service(np.eye(2))
        responses = {f"p{i}": None for i in range(6)}
        responses["p0"] = "only one"
        with patch("text_match.log.log_event") as logged:
            groups = service.match(responses, 3)
        self.assertTrue(all(g.diversity_level == "unknown" for g in groups))
        self.assertTrue(all(FALLBACK_STATEMENT not in g.diffusion_statement for g in groups))
        self.assertTrue(any("usable text" in call.args[1] for call in logged.call_args_list))


class TablePreferenceTests(unittest.TestCase):
    MEMBERS = ["a", "b"]
    NEAR_1_0 = np.asarray([[1.0, 0.0], [1.0, 0.0]])

    def test_both_methods_score_every_candidate(self):
        maximin, bridging = table_preferences(_context(), 0, self.MEMBERS, self.NEAR_1_0)
        # Members sit at [1, 0]: near, side, far are 0, 1 and 2 away.
        np.testing.assert_allclose(maximin.scores, [0.0, 1.0, 2.0])
        # a and b disagree on 3 of the 4 other comments and both approve only
        # "side": 2 ordered pairs * 3 / (n^2 = 4 * m = 5).
        np.testing.assert_allclose(bridging.scores, [0.0, 0.3, 0.0])
        self.assertEqual((bridging.population_size, bridging.observed_share), (2, 1.0))
        self.assertIsNone(bridging.fallback_reason)

    def test_own_comments_are_not_allowed(self):
        links = {**LINKED_AB, "c": Link("c", "email", 2, "pid_far")}
        maximin, _ = table_preferences(_context(links), 0, ["a", "b", "c"], np.asarray([[1.0, 0.0]] * 3))
        np.testing.assert_array_equal(maximin.allowed, [True, True, False])

    def test_methods_that_cannot_run_rank_by_the_global_order(self):
        for context, members, embeddings, method, reason in (
            (_context(matrix=None), self.MEMBERS, self.NEAR_1_0, 1, "matrix_unavailable"),
            (_context(links={"a": LINKED_AB["a"]}), self.MEMBERS, self.NEAR_1_0, 1, "fewer_than_two_linked"),
            # Two registrants on one pre-survey row are one voter.
            (_context(links={"a": LINKED_AB["a"], "b": Link("b", "name", 0, "v_a")}),
             self.MEMBERS, self.NEAR_1_0, 1, "fewer_than_two_linked"),
            (_context(), self.MEMBERS, None, 0, "participant_embedding_failed"),
            (_context(), self.MEMBERS, np.ones((2, 3)), 0, "pick_failed"),
        ):
            with self.subTest(reason=reason):
                slot = table_preferences(context, 0, members, embeddings)[method]
                self.assertTrue(slot.fallback_reason.startswith(reason))
                # fallback_rows is (side, near, far).
                np.testing.assert_array_equal(slot.ranks(), [1, 0, 2])


SIX_CANDIDATES = PreSurveyContext(
    catalog=_catalog([
        (f"c{i}", "stocking_growing", f"author{i}", f"text {i}", [np.cos(i), np.sin(i)])
        for i in range(6)
    ]),
    matrix=None, links={}, candidate_rows=tuple(range(6)), fallback_rows=(3, 1, 4, 0, 5, 2),
)


class AssignmentTests(unittest.TestCase):
    def test_a_lone_table_gets_each_methods_top_choice(self):
        result = _assign(_context(), [(["a", "b"], np.asarray([[1.0, 0.0]] * 2))])
        maximin, bridging = result.choices
        self.assertEqual((maximin.comment_id, maximin.rank), ("far", 0))
        self.assertEqual((bridging.comment_id, bridging.rank), ("side", 0))
        self.assertAlmostEqual(bridging.score, 0.3)
        self.assertEqual(result.total_rank, 0)

    def test_tables_never_share_a_comment_while_candidates_last(self):
        # Two identical tables want the same comments; with 6 candidates for
        # 4 slots, each comment is used once and the second table gives way.
        context = SIX_CANDIDATES
        same = np.asarray([[1.0, 0.2]])
        result = _assign(context, [(["x"], same), (["y"], same)])
        self.assertEqual(result.max_uses_per_comment, 1)
        self.assertEqual(len({choice.comment_id for choice in result.choices}), 4)

    def test_comments_are_reused_only_once_candidates_run_out(self):
        # 4 slots over 3 candidates: each comment may appear twice.
        near = np.asarray([[1.0, 0.0]] * 2)
        result = _assign(_context(matrix=None), [(["a", "b"], near), (["a", "b"], near)])
        self.assertEqual(result.max_uses_per_comment, 2)
        ids = [choice.comment_id for choice in result.choices]
        # Every candidate is used before any is used twice: one reuse only.
        self.assertEqual(sorted(ids.count(c) for c in set(ids)), [1, 1, 2])

    def test_a_table_never_shows_the_same_comment_twice(self):
        near = np.asarray([[1.0, 0.0]] * 2)
        result = _assign(_context(matrix=None), [(["a", "b"], near)] * 3)
        for t in range(3):
            self.assertNotEqual(result.choices[2 * t].comment_id, result.choices[2 * t + 1].comment_id)

    def test_a_duplicate_is_worse_than_showing_a_table_its_own_comment(self):
        """The table's members wrote "near" and "side", leaving only "far".
        Rather than show "far" twice, one slot gets a member's own comment."""
        links = {"a": Link("a", "email", 0, "pid_near"), "b": Link("b", "email", 1, "pid_side")}
        result = _assign(_context(links=links, matrix=None), [(["a", "b"], np.asarray([[1.0, 0.0]] * 2))])
        ids = [c.comment_id for c in result.choices]
        self.assertEqual(ids[0], "far")
        self.assertNotEqual(ids[0], ids[1])
        self.assertEqual(result.unresolved_collisions, 0)

    def test_an_unresolvable_collision_ends_instead_of_looping(self):
        """Regression: with a single candidate both slots must share it. This
        used to retry forever."""
        context = PreSurveyContext(
            catalog=STUB_CATALOG, matrix=None, links={}, candidate_rows=(2,), fallback_rows=(2,),
        )
        result = _assign(context, [(["a"], np.asarray([[1.0, 0.0]]))])
        self.assertEqual([c.comment_id for c in result.choices], ["far", "far"])
        self.assertEqual(result.unresolved_collisions, 1)

    def test_the_assignment_minimises_total_rank(self):
        """Brute force over every way to give 2 tables' 4 slots distinct
        comments from 6 candidates."""
        context = SIX_CANDIDATES
        tables = [(["x"], np.asarray([[1.0, 0.2]])), (["y"], np.asarray([[-0.3, 1.0]]))]
        preferences = [s for t, (m, e) in enumerate(tables) for s in table_preferences(context, t, m, e)]
        ranks = np.vstack([p.ranks() for p in preferences])
        best = min(sum(ranks[k, c] for k, c in enumerate(cols))
                   for cols in itertools.permutations(range(6), 4))
        self.assertEqual(assign_statements(context, preferences).total_rank, best)

    def test_format(self):
        self.assertEqual(format_statements("x", "y"), "Statement A: x\n\nStatement B: y")


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
    def test_logged_distances_reconstruct_group_diversity(self):
        embeddings = np.random.default_rng(7).normal(size=(100, 5))
        service = TextMatchingService(
            embedding_client=QueueEmbeddingClient([embeddings, np.ones((1, 5))]), optimization_seconds=0,
        )
        with patch("text_match.log.log_event") as logged:
            groups = service.match({f"p{i}": f"text {i}" for i in range(100)}, 4)
        records = [call.kwargs["extra_data"] for call in logged.call_args_list if "extra_data" in call.kwargs]
        config = next(r for r in records if "participant_order" in r)
        matrix = np.full((100, 100), np.nan)
        row_entries = [r for r in records if "row_start" in r]
        for record in row_entries:
            self.assertEqual(record["participant_ids"], config["participant_order"][record["row_start"]:record["row_start"] + record["row_count"]])
            matrix[record["row_start"]:record["row_start"] + record["row_count"]] = record["distances"]
        np.testing.assert_array_equal(matrix, cosine_distance_matrix(embeddings))
        complete = next(r for r in records if "distance_matrix_sha256" in r)
        self.assertEqual(complete["distance_matrix_sha256"], hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest())
        self.assertTrue(complete["rows_logged"])
        # Rows are packed into entries, so an analyst can check completeness
        # against entry_count instead of counting rows.
        self.assertEqual(complete["entry_count"], len(row_entries))
        self.assertLess(len(row_entries), 100)
        self.assertEqual([sum(g.diversity_level == level for g in groups) for level in ("low", "medium", "high")], [7, 11, 7])
        for group in groups:
            self.assertEqual(len(group.participant_ids), 4)
            indices = [config["participant_order"].index(pid) for pid in group.participant_ids]
            self.assertAlmostEqual(group.achieved_diversity, np.mean([matrix[a, b] for a, b in combinations(indices, 2)]))

    def test_distance_export_is_skipped_above_the_participant_cap(self):
        """Every row entry is a synchronous write on the request path. Above the
        cap the export is dropped loudly: a WARNING, a completion record that
        says so, and the checksum still present."""
        embeddings = np.random.default_rng(3).normal(size=(12, 5))
        service = TextMatchingService(
            embedding_client=QueueEmbeddingClient([embeddings, np.ones((1, 5))]), optimization_seconds=0,
        )
        with patch("text_match.DISTANCE_LOG_MAX_PARTICIPANTS", 10), patch("text_match.log.log_event") as logged:
            service.match({f"p{i}": f"text {i}" for i in range(12)}, 4)
        calls = logged.call_args_list
        records = [call.kwargs["extra_data"] for call in calls if "extra_data" in call.kwargs]
        self.assertFalse(any("row_start" in r for r in records))
        self.assertTrue(any(call.args[0] == "WARNING" and "export cap" in call.args[1] for call in calls))
        complete = next(r for r in records if "distance_matrix_sha256" in r)
        self.assertFalse(complete["rows_logged"])
        self.assertEqual(complete["entry_count"], 0)
        self.assertEqual(complete["max_participants"], 10)

    def test_blank_provenance_is_logged_as_null(self):
        """A copied .env.example leaves the provenance variables set but empty.
        Those must read as missing in the study export, not as recorded."""
        embeddings = np.random.default_rng(5).normal(size=(12, 5))
        service = TextMatchingService(
            embedding_client=QueueEmbeddingClient([embeddings, np.ones((1, 5))]), optimization_seconds=0,
        )
        env = {"MATCH_CODE_REVISION": "   ", "HF_MODEL_REVISION": " abc123 "}
        with patch.dict(os.environ, env), patch("text_match.log.log_event") as logged:
            os.environ.pop("HF_MODEL_ID", None)
            service.match({f"p{i}": f"text {i}" for i in range(12)}, 4)
        config = next(
            call.kwargs["extra_data"] for call in logged.call_args_list
            if "participant_order" in call.kwargs.get("extra_data", {})
        )
        self.assertIsNone(config["code_revision"])
        self.assertIsNone(config["embedding_model"])
        self.assertEqual(config["embedding_model_revision"], "abc123")

    def _stub_service(self, participant_embeddings):
        return TextMatchingService(
            embedding_client=QueueEmbeddingClient([participant_embeddings]),
            optimization_seconds=0,
        )

    def _stubbed(self):
        return (
            patch("text_match.load_comment_catalog", return_value=STUB_CATALOG),
            patch("text_match.load_eligible_comment_ids", return_value=STUB_ELIGIBLE),
            patch("text_match.load_bridging_ranking", return_value=STUB_RANKING),
        )

    def test_each_group_gets_two_labelled_statements_in_a_logged_order(self):
        embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]] * 4)
        service = self._stub_service(embeddings)
        catalog, eligible, ranking = self._stubbed()
        with catalog, eligible, ranking, patch("text_match.log.log_event") as logged:
            groups = service.match({f"p{i}": f"r{i}" for i in range(12)}, 3)

        record = next(
            call.kwargs["extra_data"] for call in logged.call_args_list
            if call.args[1] == "Diffusion picks"
        )
        self.assertEqual(record["candidate_count"], 3)
        # 4 tables x 2 slots over 3 candidates: each comment is used at most
        # 3 times, and "far" (everyone's maximin favourite) goes to one table.
        self.assertEqual(record["max_uses_per_comment"], 3)
        maximin_ids = [g["maximin"]["comment_id"] for g in record["groups"]]
        self.assertLessEqual(max(maximin_ids.count(c) for c in set(maximin_ids)), 3)
        for group, logged_group in zip(groups, record["groups"]):
            # No matrix is configured: maximin runs, bridging falls back.
            self.assertIsNone(logged_group["maximin"]["fallback_reason"])
            self.assertEqual(logged_group["bridging"]["fallback_reason"], "matrix_unavailable")
            self.assertNotEqual(
                logged_group["maximin"]["comment_id"], logged_group["bridging"]["comment_id"]
            )
            first = logged_group[logged_group["slot_a_method"]]["text"]
            second_method = "bridging" if logged_group["slot_a_method"] == "maximin" else "maximin"
            self.assertEqual(
                group.diffusion_statement,
                format_statements(first, logged_group[second_method]["text"]),
            )
        for group in groups:
            self.assertIsNone(group.maximin_fallback_reason)
            self.assertEqual(group.bridging_fallback_reason, "matrix_unavailable")
        # The order is randomised per group, and reproducible from the seed.
        self.assertEqual(len({g["slot_a_method"] for g in record["groups"]}), 2)

    def test_a_catalog_failure_preserves_optimized_groups(self):
        service = self._stub_service(np.eye(6))
        with patch("text_match.load_comment_catalog", side_effect=OSError("missing")), \
                patch("text_match.log.log_event"):
            groups = service.match({f"p{i}": f"r{i}" for i in range(6)}, 3)

        self.assertTrue(all(group.fallback_used for group in groups))
        # Only the statements failed, so the optimized groups and all their
        # diversity measurements survive.
        self.assertTrue(all(group.diversity_level != "unknown" for group in groups))
        self.assertTrue(all(group.achieved_diversity is not None for group in groups))
        self.assertTrue(all(FALLBACK_STATEMENT in group.diffusion_statement for group in groups))

    def test_a_broken_eligibility_file_is_named_and_linking_still_runs(self):
        """Without the screen nothing is eligible, so tables get the fallback
        text rather than unscreened comments; the error names the file, and
        the pre-survey linking is still logged."""
        service = self._stub_service(np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]] * 2))
        catalog, _, ranking = self._stubbed()
        with catalog, ranking, \
                patch("text_match.load_eligible_comment_ids", side_effect=ValueError("truncated")), \
                patch("text_match.load_approval_matrix", return_value=STUB_MATRIX), \
                patch("text_match.log.log_event") as logged:
            service.approval_matrix_uri = "stub"
            groups = service.match({f"p{i}": f"r{i}" for i in range(6)}, 3)

        messages = [(call.args[0], call.args[1]) for call in logged.call_args_list]
        self.assertTrue(any(
            level == "ERROR" and "eligibility screen failed to load" in message
            for level, message in messages
        ))
        self.assertFalse(any("catalog failed to load" in message for _, message in messages))
        self.assertIn(("INFO", "Pre-survey linking"), messages)
        self.assertTrue(all(group.achieved_diversity is not None for group in groups))
        self.assertTrue(all(FALLBACK_STATEMENT in group.diffusion_statement for group in groups))

    def test_a_broken_ranking_file_only_affects_fallback_order(self):
        """The ranking is a fallback aid: losing it must not blank the
        statements tables can compute for themselves."""
        service = self._stub_service(np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]] * 2))
        catalog, eligible, _ = self._stubbed()
        with catalog, eligible, \
                patch("text_match.load_bridging_ranking", side_effect=ValueError("truncated")), \
                patch("text_match.log.log_event") as logged:
            groups = service.match({f"p{i}": f"r{i}" for i in range(6)}, 3)

        for group in groups:
            self.assertNotIn(FALLBACK_STATEMENT, group.diffusion_statement)
        self.assertTrue(any(
            call.args[0] == "ERROR" and "ranking failed to load" in call.args[1]
            for call in logged.call_args_list
        ))

    def test_a_dimension_mismatch_falls_back_without_touching_groups(self):
        """Registration embeddings from a different model than the catalog."""
        service = self._stub_service(np.eye(6))
        catalog, eligible, ranking = self._stubbed()
        with catalog, eligible, ranking, patch("text_match.log.log_event") as logged:
            groups = service.match({f"p{i}": f"r{i}" for i in range(6)}, 3)

        self.assertTrue(all(group.fallback_used for group in groups))
        self.assertTrue(all(group.achieved_diversity is not None for group in groups))
        self.assertTrue(any(
            call.args[0] == "WARNING" and "global bridging fallback" in call.args[1]
            for call in logged.call_args_list
        ))

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
        # The committed ranking still supplies real comments for both slots.
        for group in groups:
            self.assertEqual(group.maximin_fallback_reason, "participant_embedding_failed")
            self.assertTrue(group.diffusion_statement.startswith("Statement A: "))
            self.assertNotIn(FALLBACK_STATEMENT, group.diffusion_statement)


if __name__ == "__main__":
    unittest.main()
