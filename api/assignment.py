"""Minimum-cost assignment (the Hungarian algorithm), in numpy.

Used to give every table's statement slots distinct comments: rows are slots,
columns are comments, and the solver picks one column per row, never the same
column twice, minimising the total cost.
"""

from __future__ import annotations

import numpy as np


def min_cost_assignment(cost: np.ndarray) -> np.ndarray:
    """Column assigned to each row, minimising the summed cost.

    cost is rows x columns with rows <= columns; every row gets a distinct
    column. This is the O(rows^2 * columns) shortest-augmenting-path form of
    the Hungarian algorithm with row and column potentials: each row is added
    in turn, and the cheapest reassignment chain that frees a column for it is
    found with a Dijkstra-like scan over reduced costs.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost must be a two-dimensional matrix")
    n, m = cost.shape
    if n > m:
        raise ValueError("there must be at least as many columns as rows")
    if not np.isfinite(cost).all():
        raise ValueError("cost must be finite; use a large value to forbid a pairing")

    # 1-indexed; index 0 is a sentinel column/row.
    u = np.zeros(n + 1)  # row potentials
    v = np.zeros(m + 1)  # column potentials
    owner = np.zeros(m + 1, dtype=np.int64)  # row holding each column, 0 = free
    way = np.zeros(m + 1, dtype=np.int64)  # previous column on the augmenting path

    for row in range(1, n + 1):
        owner[0] = row
        column = 0
        slack = np.full(m + 1, np.inf)
        visited = np.zeros(m + 1, dtype=bool)
        while True:
            visited[column] = True
            current_row = owner[column]
            reduced = cost[current_row - 1] - u[current_row] - v[1:]
            open_ = ~visited[1:]
            tighter = open_ & (reduced < slack[1:])
            slack[1:][tighter] = reduced[tighter]
            way[1:][tighter] = column
            next_column = int(np.argmin(np.where(open_, slack[1:], np.inf))) + 1
            delta = slack[next_column]
            done = np.flatnonzero(visited)
            u[owner[done]] += delta
            v[done] -= delta
            slack[1:][open_] -= delta
            column = next_column
            if owner[column] == 0:
                break
        # Flip the augmenting path back to the sentinel.
        while column:
            previous = way[column]
            owner[column] = owner[previous]
            column = previous

    assignment = np.empty(n, dtype=np.int64)
    for column in range(1, m + 1):
        if owner[column]:
            assignment[owner[column] - 1] = column - 1
    return assignment
