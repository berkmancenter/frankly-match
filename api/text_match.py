from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Protocol

import numpy as np

from embedding_client import EmbeddingServiceError, HuggingFaceEmbeddingClient


logger = logging.getLogger(__name__)

DiversityLevel = Literal["high", "medium", "low", "unknown"]
FALLBACK_STATEMENT = "FALLBACK STATEMENT"
MIN_TEXT_GROUP_SIZE = 3
FIVE_LEVEL_GROUP_THRESHOLD = 40
EXACT_BOUND_COMBINATION_LIMIT = 20_000
THREE_LEVEL_TARGETS = (0.0, 0.5, 1.0)
FIVE_LEVEL_TARGETS = (0.0, 0.25, 0.5, 0.75, 1.0)

DUMMY_PARTICIPANT_STATEMENTS = (
    "Public transit should be free in major cities.",
    "Public transit fares are necessary to maintain reliable service.",
    "Governments should introduce a universal basic income.",
    "Employment programs are preferable to a universal basic income.",
    "Social media platforms should face stricter content regulation.",
    "People should have broad freedom to speak on social media platforms.",
    "Cities should build substantially more dense housing.",
    "Neighborhoods should retain local control over new housing development.",
    "Carbon taxes are the best way to reduce emissions.",
    "Climate policy should prioritize direct investment over carbon taxes.",
    "University tuition should be publicly funded.",
    "Students should contribute toward the cost of their university education.",
    "Voting should be mandatory in national elections.",
    "Voting should remain a voluntary civic choice.",
    "Governments should place tighter restrictions on artificial intelligence.",
    "Artificial intelligence should develop with minimal government restriction.",
    "Healthcare should primarily be delivered through a public system.",
    "Private healthcare options improve access and innovation.",
    "Police budgets should shift toward community-based services.",
    "Police departments need additional resources to improve public safety.",
    "Immigration policy should make permanent residency easier to obtain.",
    "Immigration levels should be reduced until infrastructure catches up.",
    "Workers should have stronger legal protections for collective bargaining.",
    "Labor policy should give employers greater flexibility in hiring.",
)

DIFFUSION_STATEMENTS = (
    "Political compromise is usually more valuable than ideological consistency.",
    "Local communities should have more authority than national governments.",
    "Economic inequality is a greater threat than slow economic growth.",
    "Public institutions should favor experimentation even when it creates risk.",
    "Individual freedom should take priority over collective security.",
    "Experts should have more influence over policy than public opinion.",
    "Long-term environmental goals justify meaningful short-term costs.",
    "Essential services should not be operated for profit.",
    "Social stability sometimes requires limiting rapid political change.",
    "Technology companies should be responsible for the social effects of their products.",
    "Equal outcomes matter more than equal opportunities.",
    "People have stronger obligations to their local community than to strangers.",
    "Governments should act on uncertain risks before conclusive evidence is available.",
    "Democratic decisions are legitimate even when they produce inefficient outcomes.",
    "A healthy society should tolerate views that most people find offensive.",
    "Economic policy should prioritize resilience over maximum efficiency.",
    "Public policy should reward personal responsibility more than compensate for disadvantage.",
    "Future generations should have formal representation in present-day decisions.",
)


class EmbeddingClient(Protocol):
    def embed(self, sentences: Sequence[str]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class TextMatchGroup:
    participant_ids: list[str]
    diversity_level: DiversityLevel
    diffusion_statement: str
    fallback_used: bool
    assigned_target: float
    achieved_diversity: float | None
    normalized_achieved_diversity: float | None


@dataclass(frozen=True)
class DiversityOptimizationResult:
    groups: list[list[str]]
    achieved_diversities: list[float]
    normalized_achieved_diversities: list[float]
    assigned_targets: list[float]
    diversity_levels: list[DiversityLevel]


def placeholder_responses(participant_ids: Sequence[str]) -> dict[str, str]:
    ids = list(participant_ids)
    pool = [
        DUMMY_PARTICIPANT_STATEMENTS[index % len(DUMMY_PARTICIPANT_STATEMENTS)]
        for index in range(len(ids))
    ]
    random.Random(_stable_seed(ids)).shuffle(pool)
    return dict(zip(ids, pool))


def plan_group_sizes(
    participant_count: int,
    target_group_size: int,
    minimum_group_size: int = MIN_TEXT_GROUP_SIZE,
) -> list[int]:
    if participant_count < minimum_group_size:
        raise ValueError(
            f"textGroupMatch requires at least {minimum_group_size} participants"
        )
    if target_group_size < minimum_group_size:
        raise ValueError(
            f"textGroupMatch targetGroupSize must be at least {minimum_group_size}"
        )

    exact_groups, remainder = divmod(participant_count, target_group_size)
    if remainder == 0:
        return [target_group_size] * exact_groups
    if remainder >= minimum_group_size:
        return [target_group_size] * exact_groups + [remainder]

    exact_groups -= 1
    residual_count = target_group_size + remainder
    candidates: list[list[int]] = []
    for group_count in range(1, residual_count // minimum_group_size + 1):
        base_size, extras = divmod(residual_count, group_count)
        sizes = [base_size + 1] * extras + [base_size] * (group_count - extras)
        candidates.append(sizes)

    residual_sizes = min(
        candidates,
        key=lambda sizes: (
            -sum(size == target_group_size for size in sizes),
            sum(abs(size - target_group_size) for size in sizes),
            max(abs(size - target_group_size) for size in sizes),
            len(sizes),
        ),
    )
    return [target_group_size] * exact_groups + residual_sizes


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite values")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embeddings must not contain zero-length vectors")
    normalized = matrix / norms
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distances = 1.0 - similarities
    np.fill_diagonal(distances, 0.0)
    return distances


def plan_diversity_targets(group_count: int, *, seed: int) -> list[float]:
    if group_count < 1:
        raise ValueError("at least one group is required")

    if group_count == 1:
        return [0.5]
    if group_count == 2:
        target_levels = (0.0, 1.0)
    elif group_count < FIVE_LEVEL_GROUP_THRESHOLD:
        target_levels = THREE_LEVEL_TARGETS
    else:
        target_levels = FIVE_LEVEL_TARGETS

    base_count, remainder = divmod(group_count, len(target_levels))
    counts = [base_count] * len(target_levels)
    extra_order = sorted(
        range(len(target_levels)),
        key=lambda index: (
            abs(target_levels[index] - 0.5),
            -target_levels[index],
        ),
    )
    for index in extra_order[:remainder]:
        counts[index] += 1

    targets = [
        target
        for target, count in zip(target_levels, counts)
        for _ in range(count)
    ]
    random.Random(seed ^ 0xD1A3E571).shuffle(targets)
    return targets


def estimate_diversity_bounds(
    distances: np.ndarray,
    group_sizes: Sequence[int],
    *,
    seed: int,
) -> dict[int, tuple[float, float]]:
    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distances must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("distances must contain only finite values")

    participant_count = matrix.shape[0]
    sizes = sorted(set(group_sizes))
    if any(size < MIN_TEXT_GROUP_SIZE or size > participant_count for size in sizes):
        raise ValueError("group sizes must be valid for the distance matrix")

    return {
        size: _estimate_bounds_for_size(
            matrix,
            size,
            seed=seed ^ (size * 0x9E3779B1),
        )
        for size in sizes
    }


def optimize_diversity_groups(
    participant_ids: Sequence[str],
    embeddings: np.ndarray,
    group_sizes: Sequence[int],
    *,
    seed: int,
    time_limit_seconds: float,
    max_restarts: int = 8,
    iterations_per_restart: int | None = None,
) -> DiversityOptimizationResult:
    ids = list(participant_ids)
    sizes = list(group_sizes)
    if sum(sizes) != len(ids):
        raise ValueError("group sizes must account for every participant")
    if any(size < MIN_TEXT_GROUP_SIZE for size in sizes):
        raise ValueError("all text match groups must contain at least 3 participants")
    if max_restarts < 1:
        raise ValueError("max_restarts must be at least 1")
    if iterations_per_restart is not None and iterations_per_restart < 0:
        raise ValueError("iterations_per_restart must not be negative")

    distances = cosine_distance_matrix(embeddings)
    if distances.shape[0] != len(ids):
        raise ValueError("embedding count must match participant count")

    assigned_targets = plan_diversity_targets(len(sizes), seed=seed)
    bounds_by_size = estimate_diversity_bounds(
        distances,
        sizes,
        seed=seed,
    )
    group_bounds = [bounds_by_size[size] for size in sizes]
    rng = random.Random(seed)
    deadline = time.monotonic() + max(0.0, time_limit_seconds)
    iteration_limit = (
        iterations_per_restart
        if iterations_per_restart is not None
        else max(2_000, len(ids) * 40)
    )

    best_groups: list[list[int]] | None = None
    best_scores: list[float] | None = None
    best_objective = -math.inf

    for restart in range(max_restarts):
        shuffled = list(range(len(ids)))
        rng.shuffle(shuffled)
        groups = _allocate_indices(shuffled, sizes)
        scores = [_group_diversity(group, distances) for group in groups]
        objective = _target_objective(
            scores,
            assigned_targets,
            group_bounds,
        )

        if objective > best_objective:
            best_objective = objective
            best_groups = [group[:] for group in groups]
            best_scores = scores[:]

        if len(groups) == 1 or time.monotonic() >= deadline:
            break

        temperature = 0.05
        for iteration in range(iteration_limit):
            if time.monotonic() >= deadline:
                break

            changed_groups: tuple[int, ...]
            previous_members: tuple[int, ...]

            if len(groups) >= 3 and iteration % 7 == 0:
                first, second, third = rng.sample(range(len(groups)), 3)
                first_position = rng.randrange(len(groups[first]))
                second_position = rng.randrange(len(groups[second]))
                third_position = rng.randrange(len(groups[third]))
                previous_members = (
                    groups[first][first_position],
                    groups[second][second_position],
                    groups[third][third_position],
                )
                groups[first][first_position] = previous_members[2]
                groups[second][second_position] = previous_members[0]
                groups[third][third_position] = previous_members[1]
                changed_groups = (first, second, third)
                positions = (first_position, second_position, third_position)
            else:
                first, second = rng.sample(range(len(groups)), 2)
                first_position = rng.randrange(len(groups[first]))
                second_position = rng.randrange(len(groups[second]))
                previous_members = (
                    groups[first][first_position],
                    groups[second][second_position],
                )
                groups[first][first_position], groups[second][second_position] = (
                    previous_members[1],
                    previous_members[0],
                )
                changed_groups = (first, second)
                positions = (first_position, second_position)

            previous_scores = tuple(scores[index] for index in changed_groups)
            for index in changed_groups:
                scores[index] = _group_diversity(groups[index], distances)
            candidate_objective = _target_objective(
                scores,
                assigned_targets,
                group_bounds,
            )
            delta = candidate_objective - objective
            cooling = max(0.002, temperature * (1.0 - iteration / iteration_limit))
            accepted = delta >= 0 or rng.random() < math.exp(delta / cooling)

            if accepted:
                objective = candidate_objective
                if objective > best_objective:
                    best_objective = objective
                    best_groups = [group[:] for group in groups]
                    best_scores = scores[:]
            else:
                for index, score in zip(changed_groups, previous_scores):
                    scores[index] = score
                if len(changed_groups) == 3:
                    first, second, third = changed_groups
                    first_position, second_position, third_position = positions
                    groups[first][first_position] = previous_members[0]
                    groups[second][second_position] = previous_members[1]
                    groups[third][third_position] = previous_members[2]
                else:
                    first, second = changed_groups
                    first_position, second_position = positions
                    groups[first][first_position] = previous_members[0]
                    groups[second][second_position] = previous_members[1]

        if restart + 1 >= max_restarts or time.monotonic() >= deadline:
            break

    if best_groups is None or best_scores is None:
        raise RuntimeError("text group optimization did not produce an assignment")

    normalized_scores = [
        _normalize_diversity(score, bounds)
        for score, bounds in zip(best_scores, group_bounds)
    ]
    return DiversityOptimizationResult(
        groups=[[ids[index] for index in group] for group in best_groups],
        achieved_diversities=best_scores,
        normalized_achieved_diversities=normalized_scores,
        assigned_targets=assigned_targets,
        diversity_levels=[
            _target_level(target)
            for target in assigned_targets
        ],
    )


def select_diffusion_statement(
    group_embeddings: np.ndarray,
    statement_embeddings: np.ndarray,
    statements: Sequence[str],
) -> str:
    group_matrix = _normalize_rows(group_embeddings)
    statement_matrix = _normalize_rows(statement_embeddings)
    if statement_matrix.shape[0] != len(statements):
        raise ValueError("statement embedding count must match statement count")

    distances = 1.0 - np.clip(group_matrix @ statement_matrix.T, -1.0, 1.0)
    minimum_distances = distances.min(axis=0)
    return list(statements)[int(np.argmax(minimum_distances))]


class TextMatchingService:
    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        diffusion_statements: Sequence[str] = DIFFUSION_STATEMENTS,
        optimization_seconds: float = 30.0,
    ):
        self.embedding_client = embedding_client or HuggingFaceEmbeddingClient()
        self.diffusion_statements = tuple(diffusion_statements)
        self.optimization_seconds = optimization_seconds
        self._diffusion_embeddings: np.ndarray | None = None
        self._diffusion_lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> "TextMatchingService":
        raw_seconds = os.getenv("TEXT_MATCH_OPTIMIZATION_SECONDS", "30")
        try:
            optimization_seconds = min(240.0, max(0.0, float(raw_seconds)))
        except ValueError:
            optimization_seconds = 30.0
        return cls(optimization_seconds=optimization_seconds)

    def match(
        self,
        participant_responses: dict[str, str],
        target_group_size: int,
    ) -> list[TextMatchGroup]:
        participant_ids = list(participant_responses)
        group_sizes = plan_group_sizes(len(participant_ids), target_group_size)
        seed = _stable_seed(participant_ids)

        try:
            participant_embeddings = self.embedding_client.embed(
                list(participant_responses.values())
            )
        except EmbeddingServiceError as exc:
            logger.warning("Participant embedding failed; using random fallback: %s", exc)
            return self._fallback_groups(participant_ids, group_sizes, seed)

        optimization = optimize_diversity_groups(
            participant_ids,
            participant_embeddings,
            group_sizes,
            seed=seed,
            time_limit_seconds=self.optimization_seconds,
        )
        embedding_by_id = {
            participant_id: participant_embeddings[index]
            for index, participant_id in enumerate(participant_ids)
        }

        try:
            diffusion_embeddings = self._get_diffusion_embeddings()
        except EmbeddingServiceError as exc:
            logger.warning(
                "Diffusion statement embedding failed; using fallback statement: %s",
                exc,
            )
            return [
                TextMatchGroup(
                    participant_ids=group,
                    diversity_level=level,
                    diffusion_statement=FALLBACK_STATEMENT,
                    fallback_used=True,
                    assigned_target=assigned_target,
                    achieved_diversity=achieved_diversity,
                    normalized_achieved_diversity=normalized_achieved_diversity,
                )
                for (
                    group,
                    level,
                    assigned_target,
                    achieved_diversity,
                    normalized_achieved_diversity,
                ) in zip(
                    optimization.groups,
                    optimization.diversity_levels,
                    optimization.assigned_targets,
                    optimization.achieved_diversities,
                    optimization.normalized_achieved_diversities,
                )
            ]

        results = []
        for (
            group,
            level,
            assigned_target,
            achieved_diversity,
            normalized_achieved_diversity,
        ) in zip(
            optimization.groups,
            optimization.diversity_levels,
            optimization.assigned_targets,
            optimization.achieved_diversities,
            optimization.normalized_achieved_diversities,
        ):
            group_embeddings = np.vstack(
                [embedding_by_id[participant_id] for participant_id in group]
            )
            results.append(
                TextMatchGroup(
                    participant_ids=group,
                    diversity_level=level,
                    diffusion_statement=select_diffusion_statement(
                        group_embeddings,
                        diffusion_embeddings,
                        self.diffusion_statements,
                    ),
                    fallback_used=False,
                    assigned_target=assigned_target,
                    achieved_diversity=achieved_diversity,
                    normalized_achieved_diversity=normalized_achieved_diversity,
                )
            )
        return results

    def _get_diffusion_embeddings(self) -> np.ndarray:
        with self._diffusion_lock:
            if self._diffusion_embeddings is None:
                self._diffusion_embeddings = self.embedding_client.embed(
                    self.diffusion_statements
                )
            return self._diffusion_embeddings

    def _fallback_groups(
        self,
        participant_ids: list[str],
        group_sizes: list[int],
        seed: int,
    ) -> list[TextMatchGroup]:
        # TODO: Design a more elegant fallback strategy than random assignment.
        shuffled = participant_ids[:]
        random.Random(seed).shuffle(shuffled)
        groups = _allocate_indices(shuffled, group_sizes)
        assigned_targets = plan_diversity_targets(len(groups), seed=seed)
        return [
            TextMatchGroup(
                participant_ids=group,
                diversity_level="unknown",
                diffusion_statement=FALLBACK_STATEMENT,
                fallback_used=True,
                assigned_target=assigned_target,
                achieved_diversity=None,
                normalized_achieved_diversity=None,
            )
            for group, assigned_target in zip(groups, assigned_targets)
        ]


def _stable_seed(values: Sequence[str]) -> int:
    digest = hashlib.sha256("\0".join(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _allocate_indices(items: Sequence, sizes: Sequence[int]) -> list[list]:
    groups = []
    offset = 0
    for size in sizes:
        groups.append(list(items[offset : offset + size]))
        offset += size
    return groups


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embedding matrix must be non-empty and two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("embedding matrix must contain only finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding matrix must not contain zero-length vectors")
    return values / norms


def _group_diversity(group: Sequence[int], distances: np.ndarray) -> float:
    return _group_pair_total(group, distances) / math.comb(len(group), 2)


def _group_pair_total(group: Sequence[int], distances: np.ndarray) -> float:
    total = 0.0
    for first_position, first in enumerate(group):
        for second in group[first_position + 1 :]:
            total += float(distances[first, second])
    return total


def _estimate_bounds_for_size(
    distances: np.ndarray,
    group_size: int,
    *,
    seed: int,
) -> tuple[float, float]:
    participant_count = distances.shape[0]
    combination_count = math.comb(participant_count, group_size)

    if combination_count <= EXACT_BOUND_COMBINATION_LIMIT:
        minimum = math.inf
        maximum = -math.inf
        for group in combinations(range(participant_count), group_size):
            score = _group_diversity(group, distances)
            minimum = min(minimum, score)
            maximum = max(maximum, score)
        return float(minimum), float(maximum)

    minimum = _search_extreme_group(
        distances,
        group_size,
        maximize=False,
        rng=random.Random(seed ^ 0x10A),
    )
    maximum = _search_extreme_group(
        distances,
        group_size,
        maximize=True,
        rng=random.Random(seed ^ 0xBEEF),
    )
    return min(minimum, maximum), max(minimum, maximum)


def _search_extreme_group(
    distances: np.ndarray,
    group_size: int,
    *,
    maximize: bool,
    rng: random.Random,
    restart_count: int = 12,
) -> float:
    participant_count = distances.shape[0]
    upper_rows, upper_columns = np.triu_indices(participant_count, k=1)
    pair_values = distances[upper_rows, upper_columns]
    extreme_position = (
        int(np.argmax(pair_values))
        if maximize
        else int(np.argmin(pair_values))
    )
    starting_pairs = [
        (
            int(upper_rows[extreme_position]),
            int(upper_columns[extreme_position]),
        )
    ]
    seen_pairs = {tuple(sorted(starting_pairs[0]))}

    while len(starting_pairs) < restart_count:
        pair = tuple(sorted(rng.sample(range(participant_count), 2)))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            starting_pairs.append(pair)

    best_score = -math.inf if maximize else math.inf
    for first, second in starting_pairs:
        selected = [first, second]
        selected_mask = np.zeros(participant_count, dtype=bool)
        selected_mask[selected] = True

        while len(selected) < group_size:
            candidates = np.flatnonzero(~selected_mask)
            candidate_totals = distances[np.ix_(candidates, selected)].sum(axis=1)
            chosen_position = (
                int(np.argmax(candidate_totals))
                if maximize
                else int(np.argmin(candidate_totals))
            )
            chosen = int(candidates[chosen_position])
            selected.append(chosen)
            selected_mask[chosen] = True

        selected = _improve_extreme_group(
            selected,
            distances,
            maximize=maximize,
        )
        score = _group_diversity(selected, distances)
        if (maximize and score > best_score) or (
            not maximize and score < best_score
        ):
            best_score = score

    return float(best_score)


def _improve_extreme_group(
    group: list[int],
    distances: np.ndarray,
    *,
    maximize: bool,
    maximum_passes: int = 20,
) -> list[int]:
    selected = group[:]
    participant_count = distances.shape[0]
    pair_total = _group_pair_total(selected, distances)

    for _ in range(maximum_passes):
        selected_mask = np.zeros(participant_count, dtype=bool)
        selected_mask[selected] = True
        outside = np.flatnonzero(~selected_mask)
        best_position: int | None = None
        best_candidate: int | None = None
        best_total = pair_total

        for position, current_member in enumerate(selected):
            retained = selected[:position] + selected[position + 1 :]
            removed_total = float(distances[current_member, retained].sum())
            candidate_totals = (
                pair_total
                - removed_total
                + distances[np.ix_(outside, retained)].sum(axis=1)
            )
            candidate_position = (
                int(np.argmax(candidate_totals))
                if maximize
                else int(np.argmin(candidate_totals))
            )
            candidate_total = float(candidate_totals[candidate_position])
            improved = (
                candidate_total > best_total + 1e-12
                if maximize
                else candidate_total < best_total - 1e-12
            )
            if improved:
                best_position = position
                best_candidate = int(outside[candidate_position])
                best_total = candidate_total

        if best_position is None or best_candidate is None:
            break
        selected[best_position] = best_candidate
        pair_total = best_total

    return selected


def _normalize_diversity(
    score: float,
    bounds: tuple[float, float],
) -> float:
    minimum, maximum = bounds
    spread = maximum - minimum
    if spread <= 1e-12:
        return 0.5
    return float(np.clip((score - minimum) / spread, 0.0, 1.0))


def _target_level(target: float) -> DiversityLevel:
    if target < 1 / 3:
        return "low"
    if target > 2 / 3:
        return "high"
    return "medium"


def _target_objective(
    scores: Sequence[float],
    assigned_targets: Sequence[float],
    group_bounds: Sequence[tuple[float, float]],
) -> float:
    normalized_scores = np.asarray(
        [
            _normalize_diversity(score, bounds)
            for score, bounds in zip(scores, group_bounds)
        ],
        dtype=np.float64,
    )
    targets = np.asarray(assigned_targets, dtype=np.float64)
    return -float(np.mean(np.square(normalized_scores - targets)))
