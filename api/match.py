"""
Diversity-maximizing group matching on binary answer masks.

Port of the Dart frankly_match library. See frankly_match.dart for full docs.
"""

from collections import deque


def hamming(s1: str, s2: str) -> int:
    if len(s1) != len(s2):
        return abs(len(s1) - len(s2))
    return sum(a != b for a, b in zip(s1, s2))


def _bucket_samples(samples: dict[str, str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for pid, mask in samples.items():
        buckets.setdefault(mask, []).append(pid)
    return buckets


def _distance_matrix(keys: list[str]) -> dict[str, int]:
    return {k1 + k2: hamming(k1, k2) for k1 in keys for k2 in keys}


def _create_graph(buckets: dict, distances: dict) -> dict[str, list[str]]:
    keys = list(buckets)
    return {k1: [k2 for k2 in keys if distances[k1 + k2] == 1] for k1 in keys}


def _get_next_cluster(buckets: dict, G: dict, size: int) -> list[str]:
    cluster: list[str] = []
    by_size = lambda k: -len(buckets[k])
    sorted_keys = sorted(buckets, key=by_size)

    node = None
    Q: deque[str] = deque([sorted_keys[0]])
    visited = {sorted_keys[0]}

    while len(cluster) < size:
        if node is not None and buckets[node]:
            cluster.append(buckets[node].pop())
        else:
            node = Q.popleft() if Q else next(k for k in sorted_keys if k not in visited)
            visited.add(node)
            for nb in sorted(G[node], key=by_size):
                if nb not in visited:
                    Q.append(nb)
                    visited.add(nb)
    return cluster


def group_match(participant_responses: dict[str, str], target_group_size: int) -> list[list[str]]:
    """BFS cluster-then-compose diversity matching. See openapi.yaml for full semantics."""
    n = len(participant_responses)
    if n <= target_group_size or n <= 1:
        return [list(participant_responses)]

    buckets = _bucket_samples(participant_responses)
    distances = _distance_matrix(list(buckets))
    G = _create_graph(buckets, distances)

    cluster_size, remainder = divmod(n, target_group_size)
    clusters = [
        _get_next_cluster(buckets, G, cluster_size + (1 if i < remainder else 0))
        for i in range(target_group_size)
    ]

    groups = [[c.pop() for c in clusters] for _ in range(cluster_size)]
    for i in range(remainder):
        groups[i % len(groups)].append(clusters[i].pop())

    return groups
