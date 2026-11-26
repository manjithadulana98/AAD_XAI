"""X1-after-eye-removal analysis (scripts/analyse_xai_round11.py) on synthetic outputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyse_xai_round11", _ROOT / "scripts" / "analyse_xai_round11.py")
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)

C = 64
NAMES = [f"E{i}" for i in range(C)]
FP = [0, 1, 2, 3, 4, 5, 6, 7]
for i, n in zip(FP, ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]):
    NAMES[i] = n
SEEDS = ["42", "1", "2", "3", "4"]
EDGES = [[i, (i + 1) % C] for i in range(C)]


def _curve(ks, start, slope):
    return [[k, start - slope * k / max(ks), 0.0] for k in ks]


def _record(rng, fold, subj, fp_occ, fp_acc_drop, r9_fp_acc_drop=0.0, agree=1.0):
    """One subject-fold. fp_occ: extra occlusion score of Fp/AF; fp_acc_drop: extra accuracy
    drop when Fp/AF are deleted (round 11); r9_fp_acc_drop: same for the round-9 models."""
    ks = [0, 1, 8, 64]

    def occ():
        s = rng.standard_normal(C)
        s[FP] += fp_occ
        return s.tolist()

    def gd(extra):
        return {"intact": [0.70, 1.0], "fp": [0.68 - extra, 0.9 - extra],
                "random": [[0.68 + 0.002 * rng.standard_normal(), 0.9] for _ in range(4)]}

    half = {"n_rank": 10, "n_eval": 10,
            "channel_scores": {m: occ() for m in ("occlusion", "ig", "attention")},
            "channel_curves": {mode: {**{m: {"delete": _curve(ks, 0.7, 0.2 + 0.01 * rng.random()),
                                             "keep": _curve(ks, 0.5, -0.1)} for m in ("occlusion", "ig", "attention")},
                                      **{f"random{i}": {"delete": _curve(ks, 0.7, 0.1), "keep": _curve(ks, 0.5, -0.05)}
                                         for i in range(3)}} for mode in ("permute", "zero")},
            "edge_scores": {m: rng.standard_normal(len(EDGES)).tolist() for m in ("attention", "attn_x_grad")},
            "edge_curves": {**{m: _curve(ks, 0.7, 0.1) for m in ("attention", "attn_x_grad")},
                            **{f"random{i}": _curve(ks, 0.7, 0.05) for i in range(3)}},
            "band": {"attribution": rng.standard_normal(8).tolist(),
                     "drops": [[[1, 4], 0.0, float(d)] for d in rng.standard_normal(8)]}}
    return {"fold_id": fold, "subject": subj, "n_windows": 40, "locality": 0.8,
            "variance": rng.uniform(1, 2, C).tolist(), "variance_raw": rng.uniform(1, 2, C).tolist(),
            "reproduction": {s: {"agree": agree, "max_abs_diff": 0.0} for s in SEEDS},
            "reproduction_r9": {s: {"agree": 1.0, "max_abs_diff": 0.0} for s in SEEDS},
            "group_deletion": {s: gd(fp_acc_drop) for s in SEEDS},
            "group_deletion_r9": {s: gd(r9_fp_acc_drop) for s in SEEDS},
            "seed_scores": {s: {m: occ() for m in ("occlusion", "ig", "attention")} for s in SEEDS},
            "halves": [half, half],
            "sanity": {m: {"readout": 0.9, "all": 0.05 * rng.standard_normal()} for m in ("occlusion", "ig", "attention", "band")}}


def write(tmp, r9_occ, r11_occ, r11_drop, r9_drop, seed=0, agree=1.0, drop_one=False):
    rng = np.random.default_rng(seed)
    r9, r11 = [], []
    for g in range(6):
        for k in range(3):
            for j in range(3):
                fold, subj = f"sgso_g{g}_k{k}", f"S{3 * g + j + 1}"
                r9.append(_record(rng, fold, subj, r9_occ, 0.0))
                r11.append(_record(rng, fold, subj, r11_occ, r11_drop, r9_drop, agree))
    if drop_one:
        r11 = r11[1:]
    p9, p11 = tmp / "xai_round9.json", tmp / "xai_round11.json"
    p9.write_text(json.dumps({"commit": "r9", "results": r9}))
    p11.write_text(json.dumps({"commit": "r11", "config": {"montage": NAMES, "fp_names": [NAMES[i] for i in FP]},
                               "edges": EDGES, "results": r11}))
    return str(p9), str(p11)


def test_eye_driven_ranking_is_removed(tmp_path):
    res = ar.analyse(*write(tmp_path, r9_occ=3.0, r11_occ=0.0, r11_drop=0.0, r9_drop=0.05), out=lambda *a: None)
    assert res["E1"]["holds"] and not res["E2"]["holds"]
    assert res["E_outcome"] == "removed"
    assert res["E3"]["E3a"] and not res["E3"]["E3b"]


def test_weaker_but_still_present_ranking_is_reduced(tmp_path):
    res = ar.analyse(*write(tmp_path, r9_occ=3.0, r11_occ=1.0, r11_drop=0.03, r9_drop=0.03, seed=1),
                     out=lambda *a: None)
    assert res["E_outcome"] == "reduced"
    assert not res["E3"]["E3a"] and res["E3"]["E3b"]


def test_unchanged_ranking_is_not_reduced(tmp_path):
    res = ar.analyse(*write(tmp_path, r9_occ=2.0, r11_occ=2.0, r11_drop=0.0, r9_drop=0.0, seed=3),
                     out=lambda *a: None)
    assert res["E_outcome"] == "not reduced"


@pytest.mark.parametrize("kw", [{"agree": 0.95}, {"drop_one": True}])
def test_gates(tmp_path, kw):
    with pytest.raises(AssertionError):
        ar.analyse(*write(tmp_path, 3.0, 0.0, 0.0, 0.0, **kw), out=lambda *a: None)


def test_fp_index_and_outcome_rule():
    M = np.zeros((2, C))
    M[0, FP] = 1.0                                              # Fp/AF above the rest
    M[1, :] = np.arange(C)                                      # Fp/AF (indices 0-7) lowest
    idx = ar.fp_index(M, FP)
    assert idx[0] > 2 and idx[1] < -1
    assert ar.e1_outcome(True, False) == "removed"
    assert ar.e1_outcome(True, True) == "reduced"
    assert ar.e1_outcome(False, True) == "not reduced"
