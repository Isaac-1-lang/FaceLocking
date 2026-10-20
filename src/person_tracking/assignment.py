"""Optimal rectangular assignment (Hungarian / Jonker-Volgenant style).

The tracker only ever handles a handful of people, so a dependency-free
O(n^2 m) implementation is faster than importing SciPy and keeps the
requirements unchanged.
"""
import numpy as np

_BIG = 1e9


def linear_assignment(cost):
    """Minimum-cost matching. Returns sorted ``(row, col)`` pairs; every row of the
    smaller dimension is matched. Non-finite costs are treated as very large, so
    callers must still gate the returned pairs by their own thresholds."""
    cost = np.array(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.size == 0:
        return []
    cost[~np.isfinite(cost)] = _BIG
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j]]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)
