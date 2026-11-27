"""Electrode sets for group deletion: compact clusters and shape-matched rotated sets.

Used to ask whether deleting a given electrode group (e.g. the frontal-polar band) costs more
than deleting other spatially contiguous groups of the same size, rather than scattered ones.

- knn_clusters: each electrode plus its k-1 nearest neighbours on the 2-D montage.
- rotated_sets: a reference set rotated about a centre point and snapped one-to-one to the
  nearest electrodes (optimal assignment), so the copy keeps the shape and the position at the
  cap edge.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def centre_clusters(xy, k: int) -> list[list[int]]:
    """xy (C, 2) electrode positions -> C lists: cluster c = electrode c and its k-1 nearest
    neighbours (Euclidean, ties broken by channel index), sorted by index."""
    xy = np.asarray(xy, dtype=float)                                      # (C, 2)
    C = len(xy)
    if not 1 <= k <= C:
        raise ValueError(f"k must be in [1, {C}], got {k}")
    D = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)        # (C, C)
    idx = np.arange(C)
    return [sorted(np.lexsort((idx, D[c]))[:k].tolist()) for c in range(C)]


def knn_clusters(xy, k: int) -> list[list[int]]:
    """Distinct clusters of centre_clusters, in order of first appearance."""
    seen, out = set(), []
    for cl in centre_clusters(xy, k):
        key = tuple(cl)
        if key not in seen:
            seen.add(key)
            out.append(cl)
    return out


def rotated_sets(xy, ref_idx, degrees, centre=(0.0, 0.0)) -> dict[int, list[int]]:
    """Rotate the electrodes ref_idx about centre by each angle (degrees, counter-clockwise) and
    assign the rotated points one-to-one to electrodes, minimising the summed distance.

    Returns {angle: sorted electrode indices}; angle 0 returns ref_idx itself.
    """
    xy = np.asarray(xy, dtype=float)                                      # (C, 2)
    ref_idx = list(ref_idx)
    c = np.asarray(centre, dtype=float)
    out = {}
    for deg in degrees:
        t = np.deg2rad(deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        P = (xy[ref_idx] - c) @ R.T + c                                    # (k, 2)
        D = np.linalg.norm(P[:, None, :] - xy[None, :, :], axis=-1)        # (k, C)
        _, cols = linear_sum_assignment(D)
        out[int(deg)] = sorted(cols.tolist())
    return out


def disjoint_from(sets, ref_idx) -> list[list[int]]:
    """The sets that share no electrode with ref_idx."""
    ref = set(ref_idx)
    return [list(s) for s in sets if not ref & set(s)]
