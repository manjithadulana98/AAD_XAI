"""Cluster-matched group deletion analysis (scripts/analyse_cluster_deletion.py) on synthetic outputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aad_xai.models.sgat import DTU_MONTAGE_PATH
from aad_xai.xai.electrode_sets import centre_clusters, knn_clusters, rotated_sets

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyse_cluster_deletion",
                                               _ROOT / "scripts" / "analyse_cluster_deletion.py")
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)

SEEDS = ["42", "1", "2", "3", "4"]
FP_NAMES = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]


def _config():
    d = pd.read_csv(DTU_MONTAGE_PATH).sort_values("channel_index")
    names, xy = d["electrode_name"].tolist(), d[["x", "y"]].to_numpy()
    fp = [names.index(n) for n in FP_NAMES]
    rest = [c for c in range(64) if c not in fp]
    rng = np.random.default_rng(13)
    return {"seeds": [int(s) for s in SEEDS], "montage": names, "fp_names": FP_NAMES,
            "scatter": [rng.choice(rest, 8, replace=False).tolist() for _ in range(20)],
            "clusters": knn_clusters(xy, 8), "centre_clusters": centre_clusters(xy, 8),
            "bands": {str(a): s for a, s in rotated_sets(xy, fp, range(0, 360, 45),
                                                         centre=xy[names.index("Cz")]).items()}}


CFG = _config()


def _deletion(rng, fp_extra, contig, band_extra):
    """One model's deletion record. Accuracy after deletion: scattered 0.62, compact clusters
    0.62 - contig, bands 0.62 - contig - band_extra, FP 0.62 - contig - fp_extra (plus noise)."""
    n = lambda: 0.004 * rng.standard_normal()
    base = 0.62 - contig
    return {"intact": [0.70, 1.0],
            "fp": [base - fp_extra + n(), 0.5],
            "scatter": [[0.62 + n(), 0.8] for _ in CFG["scatter"]],
            "clusters": [[base + n(), 0.7] for _ in CFG["clusters"]],
            "bands": {a: [base - band_extra + n(), 0.6] for a in CFG["bands"] if a != "0"}}


def _files(tmp_path, fp_extra=0.03, contig=0.02, band_extra=0.0, agree=1.0, r11_shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    res, old = [], []
    for si in range(18):
        for k in range(3):
            fold, subj = f"sgso_g{si % 6}_k{k}", f"S{si + 1}"
            dels = {s: _deletion(rng, fp_extra, contig, band_extra) for s in SEEDS}
            dels9 = {s: _deletion(rng, fp_extra, contig, band_extra) for s in SEEDS}
            res.append({"fold_id": fold, "subject": subj, "n_windows": 40,
                        "reproduction": {s: {"agree": agree, "max_abs_diff": 0.0} for s in SEEDS},
                        "reproduction_r9": {s: {"agree": 1.0, "max_abs_diff": 0.0} for s in SEEDS},
                        "deletion": dels, "deletion_r9": dels9})
            sh = lambda v: [v[0] + r11_shift, v[1]]
            old.append({"fold_id": fold, "subject": subj, "n_windows": 40,
                        "group_deletion": {s: {"intact": d["intact"], "fp": sh(d["fp"]),
                                               "random": [sh(q) for q in d["scatter"]]} for s, d in dels.items()},
                        "group_deletion_r9": {s: {"intact": d["intact"], "fp": sh(d["fp"]),
                                                  "random": [sh(q) for q in d["scatter"]]}
                                              for s, d in dels9.items()}})
    pc, p11 = tmp_path / "cd.json", tmp_path / "r11.json"
    pc.write_text(json.dumps({"commit": "synthetic", "config": CFG, "results": res}))
    p11.write_text(json.dumps({"commit": "synthetic", "results": old}))
    return pc, p11


def _run(tmp_path, **kw):
    return ac.analyse(*_files(tmp_path, **kw), out=lambda *_: None)


def test_families():
    fam = ac.families(CFG)
    assert len(fam["clusters"]) == 40
    assert fam["bands"] == ["90", "135", "180", "225", "270"]


@pytest.mark.parametrize("c1,c2,expected", [(True, True, "specific"), (False, False, "not specific"),
                                            (True, False, "mixed"), (False, True, "mixed")])
def test_outcome(c1, c2, expected):
    assert ac.outcome(c1, c2) == expected


def test_specific(tmp_path):
    r = _run(tmp_path, fp_extra=0.03)
    assert r["round 11"]["outcome"] == "specific"
    assert r["round 9"]["outcome"] == "specific"
    assert np.mean(r["round 11"]["clusters"]) == pytest.approx(0.03, abs=0.003)
    assert r["C4"]["holds"]                                                # contig = 0.02 > 0


def test_not_specific(tmp_path):
    r = _run(tmp_path, fp_extra=-0.03, contig=-0.02)
    assert r["round 11"]["outcome"] == "not specific"
    assert not r["C4"]["holds"]


def test_mixed_when_bands_cost_more(tmp_path):
    r = _run(tmp_path, fp_extra=0.03, band_extra=0.06)
    assert r["round 11"]["holds_clusters"]
    assert not r["round 11"]["holds_bands"]
    assert r["round 11"]["outcome"] == "mixed"


def test_cluster_map_reports_every_centre(tmp_path):
    r = _run(tmp_path)
    assert len(r["cluster_map"]["round 11"]) == 64
    assert np.mean(r["cluster_map"]["round 11"]) == pytest.approx(0.70 - 0.60, abs=0.003)


def test_gate_reproduction(tmp_path):
    with pytest.raises(AssertionError):
        _run(tmp_path, agree=0.95)


def test_gate_previous_run(tmp_path):
    assert _run(tmp_path, r11_shift=0.004)["gate_diff"] == pytest.approx(0.004)
    with pytest.raises(AssertionError):
        _run(tmp_path, r11_shift=0.01)
