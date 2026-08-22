from __future__ import annotations

import hashlib
import math
import os
import random
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal, Protocol

from logger import log
import numpy as np

from embedding_client import EmbeddingServiceError, HuggingFaceEmbeddingClient

DiversityLevel = Literal["high", "medium", "low", "unknown"]
FALLBACK_STATEMENT = "FALLBACK STATEMENT"
MIN_TEXT_GROUP_SIZE = 3
EXACT_BOUND_COMBINATION_LIMIT = 20_000

# Group formation redistributes disagreement between groups, so a low arm
# funds a further high arm (see high_arm_target). An earlier revision leaned on
# that hard: a 2.5:1 low-to-high ratio pushing a thin high arm as far as the
# budget allowed. Two problems, both measured on synthetic pools: the
# achievable ceiling binds at a low-to-high ratio of roughly 0.3-0.5, long
# before a 2.5 budget does, so most of the extra low groups bought no
# additional contrast; and a 2-group arm has no usable within-arm variance and
# is one embedding fallback away from vanishing.
#
# The allocation is a deliberate compromise across the three quantities the
# analysis plan commits to, which have three different optimal allocations:
#
#   curvature contrast 2M - L - H (primary)   -> 2 : 1 : 1   (5/10/5 at G=20)
#   pairwise M - H (needed for the strong
#     "medium is best" claim, and for M - L)  -> sqrt(2) : 1 : 1  (6/8/6)
#   linear contrast H - L (registered hedge)  -> fatter extremes still
#
# sqrt(2) : 1 : 1 sits between them. Measured cost against allocating purely
# for the primary (5/10/5), at a true medium advantage of 1.0 group-outcome SD:
# curvature power 67.5% vs 69.2% (-1.7 points), M - H discrimination 41.4% vs
# 40.6%, linear contrast 37.3% vs 32.0% (+5.3 points). The 1.7-point loss on
# the primary buys more than three times that back on the other two. Five
# groups per extreme arm is also one dissolved group away from four, and the
# extreme arms carry the dose contrast.
#
# At 20 groups this is 6 / 8 / 6; at 25 groups, 7 / 11 / 7.
#
# Keeping n_low == n_high is not cosmetic: it makes the linear and quadratic
# contrasts exactly orthogonal (sum of c1_i * c2_i / n_i == 0), so the two
# registered hypotheses are statistically independent and neither borrows
# evidence from the other.
MEDIUM_ARM_WEIGHT = 2 ** 0.5
# Below this many groups a three-level design is not identifiable. Every group
# is targeted at the pool mean instead, which still improves on random
# assignment by removing scatter rather than by shifting the mean.
MIN_GROUPS_FOR_LEVELS = 3

# Targets are pulled in from the achievable extremes by TARGET_PULL_IN of the
# range. Targeting the exact floor or ceiling means the optimizer can only miss
# inward, which makes assignment error one-sided; a symmetric margin keeps the
# error roughly centered and the extreme arms reliably reachable.
TARGET_PULL_IN = 0.05

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
    assigned_target: float | None
    achieved_diversity: float | None


@dataclass(frozen=True)
class DiversityOptimizationResult:
    groups: list[list[str]]
    achieved_diversities: list[float]
    assigned_targets: list[float]
    diversity_levels: list[DiversityLevel]
    # Pool geometry the targets were derived from. Carried out so it can be
    # logged: r* = (ceiling - mean) / (mean - floor) locates where the
    # achievable ceiling starts binding, and it can only be measured against
    # real pools.
    pool_mean_diversity: float = 0.0
    bounds_by_size: dict[int, tuple[float, float]] = field(default_factory=dict)


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


def pool_mean_distance(distances: np.ndarray) -> float:
    """Expected diversity of a uniformly random group -- exactly, for any size.

    Every pair is equally likely to be co-assigned, so the expected mean pairwise
    distance of a random group equals the mean pairwise distance over the whole
    pool. This is an identity, not an approximation, and it does not depend on
    the group size.
    """
    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distances must be a square matrix")
    if matrix.shape[0] < 2:
        raise ValueError("at least two participants are required")
    return float(matrix[np.triu_indices(matrix.shape[0], k=1)].mean())


def plan_arm_counts(group_count: int) -> tuple[int, int, int]:
    """Allocate groups to arms as sqrt(2) : 1 : 1 (medium : low : high).

    Pure arithmetic on the group count -- no distances involved. The extreme
    arms are kept equal so the linear and quadratic contrasts stay orthogonal;
    the medium arm takes the remainder. Returns (low, middle, high).
    """
    if group_count < MIN_GROUPS_FOR_LEVELS:
        raise ValueError(
            f"three arms require at least {MIN_GROUPS_FOR_LEVELS} groups"
        )
    extreme = max(1, round(group_count / (2.0 + MEDIUM_ARM_WEIGHT)))
    while group_count - 2 * extreme < 1:
        extreme -= 1
    return extreme, group_count - 2 * extreme, extreme


def high_arm_target(
    mean_diversity: float,
    low_targets: Sequence[float],
    high_count: int,
) -> float:
    """The highest target the budget allows, before the ceiling is applied.

    Achieved diversities average to roughly the pool mean, so every unit of
    diversity a low group gives up is a unit the high arm can take:

        H = mu + sum(mu - low_g) / n_high

    which reduces to the familiar mu * (1 + n_low / n_high) when the low arm sits
    at zero. With equal-sized arms this caps at 2 * mu -- reaching further takes a
    larger low arm, not a bolder request.
    """
    if high_count < 1:
        raise ValueError("high_count must be at least 1")
    freed = sum(mean_diversity - target for target in low_targets)
    return mean_diversity + freed / high_count


def plan_diversity_targets(
    group_sizes: Sequence[int],
    distances: np.ndarray,
    *,
    seed: int,
    bounds_by_size: dict[int, tuple[float, float]] | None = None,
) -> tuple[list[float], list[DiversityLevel]]:
    """Assign a raw diversity target to each group.

    Targets are raw distances on the embedding's own scale, not values
    normalized against pool-specific bounds, so that a level means the same
    thing in every session and results can be pooled. Low groups target the
    achievable floor, middle groups the pool mean, and high groups whatever the
    budget then allows, capped at the achievable ceiling.
    """
    sizes = list(group_sizes)
    group_count = len(sizes)
    if group_count < 1:
        raise ValueError("at least one group is required")

    mean_diversity = pool_mean_distance(distances)
    if bounds_by_size is None:
        bounds_by_size = estimate_diversity_bounds(distances, sizes, seed=seed)

    if group_count < MIN_GROUPS_FOR_LEVELS:
        # Not enough groups to identify three levels. Target every group at the
        # pool mean; that still beats random assignment by removing scatter.
        return [mean_diversity] * group_count, ["medium"] * group_count

    low_count, middle_count, high_count = plan_arm_counts(group_count)
    arms: list[DiversityLevel] = (
        ["low"] * low_count + ["medium"] * middle_count + ["high"] * high_count
    )
    random.Random(seed ^ 0xD1A3E571).shuffle(arms)

    def pulled_bounds(size: int) -> tuple[float, float]:
        floor, ceiling = bounds_by_size[size]
        margin = TARGET_PULL_IN * (ceiling - floor)
        return floor + margin, ceiling - margin

    low_targets = {
        index: pulled_bounds(sizes[index])[0]
        for index, arm in enumerate(arms)
        if arm == "low"
    }
    high_target = high_arm_target(
        mean_diversity, list(low_targets.values()), high_count
    )

    targets: list[float] = []
    for index, arm in enumerate(arms):
        if arm == "low":
            targets.append(low_targets[index])
        elif arm == "medium":
            targets.append(mean_diversity)
        else:
            ceiling = pulled_bounds(sizes[index])[1]
            targets.append(max(mean_diversity, min(high_target, ceiling)))
    return targets, arms


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

    bounds_by_size = estimate_diversity_bounds(
        distances,
        sizes,
        seed=seed,
    )
    assigned_targets, diversity_levels = plan_diversity_targets(
        sizes,
        distances,
        seed=seed,
        bounds_by_size=bounds_by_size,
    )
    # Raw targets live on the embedding's own scale, so the annealing
    # temperature has to scale with it rather than assuming a [0, 1] objective.
    upper_triangle = distances[np.triu_indices(len(ids), k=1)]
    temperature_scale = max(float(upper_triangle.var()), 1e-12)

    rng = random.Random(seed)
    deadline = time.monotonic() + max(0.0, time_limit_seconds)
    # Sized so the wall-clock deadline is the binding constraint, not the
    # iteration count. The deadline is checked inside the loop, so a generous
    # limit costs nothing and lets an offline run spend the budget it was given.
    iteration_limit = (
        iterations_per_restart
        if iterations_per_restart is not None
        else max(20_000, len(ids) * 200)
    )

    best_groups: list[list[int]] | None = None
    best_scores: list[float] | None = None
    best_objective = -math.inf

    for restart in range(max_restarts):
        shuffled = list(range(len(ids)))
        rng.shuffle(shuffled)
        groups = _allocate_indices(shuffled, sizes)
        scores = [_group_diversity(group, distances) for group in groups]
        objective = _target_objective(scores, assigned_targets)

        if objective > best_objective:
            best_objective = objective
            best_groups = [group[:] for group in groups]
            best_scores = scores[:]

        if len(groups) == 1 or time.monotonic() >= deadline:
            break

        temperature = 0.05 * temperature_scale
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
            candidate_objective = _target_objective(scores, assigned_targets)
            delta = candidate_objective - objective
            cooling = max(
                0.002 * temperature_scale,
                temperature * (1.0 - iteration / iteration_limit),
            )
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

    return DiversityOptimizationResult(
        groups=[[ids[index] for index in group] for group in best_groups],
        achieved_diversities=best_scores,
        assigned_targets=assigned_targets,
        diversity_levels=diversity_levels,
        pool_mean_diversity=pool_mean_distance(distances),
        bounds_by_size=bounds_by_size,
    )


def _log_pool_geometry(
    optimization: DiversityOptimizationResult,
    request=None,
) -> None:
    """Log the distance geometry each event's targets were derived from.

    r_star is the low-to-high ratio at which the achievable ceiling starts
    binding, so low groups beyond it stop buying contrast. The balanced
    allocation sits near 1; logging r_star per event verifies against real
    pools that the ceiling, not the budget, is the binding constraint there
    too, and by how much.
    """
    mean = optimization.pool_mean_diversity
    bounds = {
        str(size): {"floor": low, "ceiling": high}
        for size, (low, high) in optimization.bounds_by_size.items()
    }
    r_star_by_size = {}
    for size, (low, high) in optimization.bounds_by_size.items():
        margin = TARGET_PULL_IN * (high - low)
        headroom = mean - (low + margin)
        if headroom > 1e-9:
            r_star_by_size[str(size)] = (high - margin - mean) / headroom

    level_counts: dict[str, int] = {}
    for level in optimization.diversity_levels:
        level_counts[level] = level_counts.get(level, 0) + 1

    log.log_event(
        "INFO",
        f"Diversity targets planned from pool mean {mean:.4f}",
        request=request,
        extra_data={
            "pool_mean_diversity": mean,
            "bounds_by_size": bounds,
            "r_star_by_size": r_star_by_size,
            "configured_target_pull_in": TARGET_PULL_IN,
            "arm_counts": level_counts,
            "assigned_targets": optimization.assigned_targets,
            "achieved_diversities": optimization.achieved_diversities,
            "target_error": [
                achieved - target
                for achieved, target in zip(
                    optimization.achieved_diversities,
                    optimization.assigned_targets,
                )
            ],
        },
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
        request=None,
    ) -> list[TextMatchGroup]:
        participant_ids = list(participant_responses)
        group_sizes = plan_group_sizes(len(participant_ids), target_group_size)
        seed = _stable_seed(participant_ids)

        try:
            participant_embeddings = self.embedding_client.embed(
                list(participant_responses.values())
            )
        except EmbeddingServiceError as exc:
            log.log_event(
                "WARNING",
                f"Participant embedding failed; using random fallback: {exc}",
                request=request,
                extra_data={"participant_count": len(participant_ids)},
            )
            return self._fallback_groups(participant_ids, group_sizes, seed)

        optimization = optimize_diversity_groups(
            participant_ids,
            participant_embeddings,
            group_sizes,
            seed=seed,
            time_limit_seconds=self.optimization_seconds,
        )
        _log_pool_geometry(optimization, request)

        embedding_by_id = {
            participant_id: participant_embeddings[index]
            for index, participant_id in enumerate(participant_ids)
        }

        try:
            diffusion_embeddings = self._get_diffusion_embeddings()
        except EmbeddingServiceError as exc:
            log.log_event(
                "WARNING",
                f"Diffusion statement embedding failed; using fallback statement: {exc}",
                request=request,
            )
            return [
                self._build_group(optimization, index, FALLBACK_STATEMENT, True)
                for index in range(len(optimization.groups))
            ]

        results = []
        for index, group in enumerate(optimization.groups):
            group_embeddings = np.vstack(
                [embedding_by_id[participant_id] for participant_id in group]
            )
            statement = select_diffusion_statement(
                group_embeddings,
                diffusion_embeddings,
                self.diffusion_statements,
            )
            results.append(self._build_group(optimization, index, statement, False))
        return results

    @staticmethod
    def _build_group(
        optimization: DiversityOptimizationResult,
        index: int,
        diffusion_statement: str,
        fallback_used: bool,
    ) -> TextMatchGroup:
        return TextMatchGroup(
            participant_ids=optimization.groups[index],
            diversity_level=optimization.diversity_levels[index],
            diffusion_statement=diffusion_statement,
            fallback_used=fallback_used,
            assigned_target=optimization.assigned_targets[index],
            achieved_diversity=optimization.achieved_diversities[index],
        )

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
        # Without embeddings there is no distance matrix, so no target can be
        # assigned and no diversity reported -- every numeric field stays None.
        shuffled = participant_ids[:]
        random.Random(seed).shuffle(shuffled)
        groups = _allocate_indices(shuffled, group_sizes)
        return [
            TextMatchGroup(
                participant_ids=group,
                diversity_level="unknown",
                diffusion_statement=FALLBACK_STATEMENT,
                fallback_used=True,
                assigned_target=None,
                achieved_diversity=None,
            )
            for group in groups
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


def _target_objective(
    scores: Sequence[float],
    assigned_targets: Sequence[float],
) -> float:
    achieved = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(assigned_targets, dtype=np.float64)
    return -float(np.mean(np.square(achieved - targets)))
