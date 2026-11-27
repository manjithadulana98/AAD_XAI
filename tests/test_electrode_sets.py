"""Electrode sets for group deletion (aad_xai.xai.electrode_sets)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aad_xai.models.sgat import DTU_MONTAGE_PATH
from aad_xai.xai.electrode_sets import centre_clusters, disjoint_from, knn_clusters, rotated_sets

FP_NAMES = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]


@pytest.fixture(scope="module")
def montage():
    d = pd.read_csv(DTU_MONTAGE_PATH).sort_values("channel_index")
    return d["electrode_name"].tolist(), d[["x", "y"]].to_numpy()


def test_centre_clusters_on_a_line():
    xy = np.c_[np.arange(6, dtype=float), np.zeros(6)]                     # 6 electrodes on a line
    cl = centre_clusters(xy, 3)
    assert len(cl) == 6
    assert cl[0] == [0, 1, 2]                                              # edge: one-sided
    assert cl[2] == [1, 2, 3]                                              # interior: symmetric
    assert cl[5] == [3, 4, 5]
    assert all(c in s and len(s) == 3 for c, s in enumerate(cl))


def test_ties_broken_by_index():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])                   # 1 and 2 equidistant from 0
    assert centre_clusters(xy, 2)[0] == [0, 1]


def test_knn_clusters_deduplicates():
    xy = np.c_[np.arange(6, dtype=float), np.zeros(6)]
    cl = knn_clusters(xy, 3)
    assert cl == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_k_out_of_range():
    with pytest.raises(ValueError):
        centre_clusters(np.zeros((4, 2)), 5)


def test_rotation_of_a_square():
    xy = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [2.0, 0.0]])
    rs = rotated_sets(xy, [0], [0, 90, 180, 270])
    assert rs == {0: [0], 90: [1], 180: [2], 270: [3]}
    rs2 = rotated_sets(xy, [0, 1], [90])                                   # one-to-one assignment
    assert rs2[90] == [1, 2]


def test_dtu_montage_sets(montage):
    names, xy = montage
    fp = [names.index(n) for n in FP_NAMES]
    cl = knn_clusters(xy, 8)
    assert len(cl) == 55 and all(len(c) == 8 for c in cl)
    assert len(disjoint_from(cl, fp)) == 40
    rs = rotated_sets(xy, fp, range(0, 360, 45), centre=xy[names.index("Cz")])
    assert rs[0] == sorted(fp)
    assert sorted(names[i] for i in rs[180]) == sorted(["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8"])
    dis = disjoint_from([rs[a] for a in rs if a], fp)
    assert len(dis) == 5 and all(len(s) == 8 for s in dis)
