"""Explanation-transfer analysis (scripts/analyse_xai_transfer.py) on synthetic X1-format outputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyse_xai_transfer", Path(__file__).resolve().parents[1] / "scripts" / "analyse_xai_transfer.py")
axt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(axt)

C, E = 64, 12
EDGES = [[i, (i + 1) % C] for i in range(E)]
SEEDS = ["42", "1", "2", "3", "4"]
BANDS_HZ = [[1, 4], [4, 8], [8, 10], [10, 13], [13, 16], [16, 20], [20, 26], [26, 32]]


def _curve(level):
    return [[k, float(level), 0.0] for k in (0, 8, 64)]


def _record(fold_id, subject, rng, ch_map, var_map, band_drop, faithful=True):
    """One X1-format subject-fold record. ch_map (C,), var_map (C,), band_drop (8,)."""
    noisy = lambda v, s=0.3: (np.asarray(v) + s * rng.standard_normal(np.shape(v))).tolist()
    curves = {}
    for mode in ("permute", "zero"):
        curves[mode] = {m: {"delete": _curve(0.55 if faithful else 0.6), "keep": _curve(0.55 if faithful else 0.5)}
                        for m in ("occlusion", "ig")}
        curves[mode]["attention"] = {"delete": _curve(0.58 if faithful else 0.6), "keep": _curve(0.53)}
        curves[mode].update({f"random{i}": {"delete": _curve(0.6 + 0.001 * rng.standard_normal()),
                                            "keep": _curve(0.5 + 0.001 * rng.standard_normal())}
                             for i in range(3)})
    half = {"n_rank": 10, "n_eval": 10,
            "channel_scores": {m: noisy(ch_map) for m in ("occlusion", "ig", "attention")},
            "channel_curves": curves,
            "edge_scores": {m: noisy(np.linspace(1, 0, E)) for m in ("attention", "attn_x_grad")},
            "edge_curves": {**{m: _curve(0.65) for m in ("attention", "attn_x_grad")},
                            **{f"random{i}": _curve(0.7 + 0.001 * rng.standard_normal()) for i in range(3)}},
            "band": {"attribution": noisy(-np.asarray(band_drop), 0.01),
                     "drops": [[b, 0.0, float(d)] for b, d in zip(BANDS_HZ, noisy(band_drop, 0.01))]}}
    return {"fold_id": fold_id, "subject": subject, "n_windows": 20, "locality": 0.8,
            "variance": np.exp(noisy(var_map, 0.1)).tolist(),
            "reproduction": {s: {"agree": 1.0, "max_abs_diff": 0.0} for s in SEEDS},
            "seed_scores": {s: {m: noisy(ch_map) for m in ("occlusion", "ig", "attention")} for s in SEEDS},
            "halves": [half, half],
            "sanity": {m: {"readout": 0.9, "all": float(0.05 * rng.standard_normal())}
                       for m in ("occlusion", "ig", "attention", "band")}}


DROP = np.array([0.05, 0.17, 0.12, 0.10, 0.02, 0.004, 0.004, 0.01])


def _dataset(n_subj, n_folds, rng, ch_map, var_map, drop=DROP):
    return [_record(f"f{s % 6}_k{k}", f"S{s + 1}", rng, ch_map, var_map, drop)
            for s in range(n_subj) for k in range(n_folds)]


def _write(tmp_path, dtu, kul, dvar=None):
    (tmp_path / "dtu.json").write_text(json.dumps({"results": dtu}))
    dvar = dvar if dvar is not None else [{k: r[k] for k in ("fold_id", "subject", "n_windows", "variance")}
                                          for r in dtu]
    (tmp_path / "dvar.json").write_text(json.dumps({"results": dvar}))
    names = [f"E{i}" for i in range(C)]
    for part, subj in (("A", range(1, 9)), ("B", range(9, 17))):
        d = tmp_path / f"kul{part}" / f"results_xai_kul_{part}"
        d.mkdir(parents=True)
        recs = [r for r in kul if int(r["subject"][1:]) in subj]
        (d / f"xai_kul_{part}.json").write_text(json.dumps(
            {"commit": "test", "part": part, "config": {"montage": names}, "edges": EDGES, "results": recs}))
    return [str(tmp_path / n) for n in ("dtu.json", "dvar.json", "kulA", "kulB")]


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(axt, "N_BOOT", 100)
    monkeypatch.setattr(axt, "N_SPLITS", 50)


# --- helpers -------------------------------------------------------------------------------------
def test_area_and_holm():
    assert axt.area(_curve(0.6)) == pytest.approx(0.6)
    assert np.allclose(axt.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_group_map_ignores_per_subject_scale():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((5, C))
    M2 = M * np.array([1, 10, 0.1, 3, 7])[:, None] + 5
    assert np.allclose(axt.group_map(M), axt.group_map(M2))


def test_partial_spearman_removes_a_shared_covariate():
    rng = np.random.default_rng(1)
    cov = rng.standard_normal(200)
    a, b = cov + 0.3 * rng.standard_normal(200), cov + 0.3 * rng.standard_normal(200)
    assert axt.rho(a, b) > 0.7
    assert abs(axt.partial_spearman(a, b, [cov])) < 0.25
    shared = rng.standard_normal(200)
    a2, b2 = a + 2 * shared, b + 2 * shared
    assert axt.partial_spearman(a2, b2, [cov]) > 0.5


def test_in_degree_counts_edge_targets():
    deg = axt.in_degree([[0, 1], [2, 1], [1, 0]], n=3)
    assert deg.tolist() == [1, 2, 0]


def test_split_half_reliability_extremes():
    rng = np.random.default_rng(2)
    same = np.tile(rng.standard_normal(C), (16, 1))
    assert axt.split_half_reliability(same, n_splits=20) == pytest.approx(1.0)
    noise = rng.standard_normal((16, C))
    assert abs(axt.split_half_reliability(noise, n_splits=200)) < 0.35


def test_transfer_detects_shared_content_beyond_variance():
    rng = np.random.default_rng(3)
    content, var = rng.standard_normal(C), rng.standard_normal(C)
    Md = content + var + 0.5 * rng.standard_normal((18, C))
    Mk = content + var + 0.5 * rng.standard_normal((16, C))
    Vd, Vk = var + 0.2 * rng.standard_normal((18, C)), var + 0.2 * rng.standard_normal((16, C))
    t = axt.transfer(Md, Mk, Vd, Vk, np.zeros(C), n_boot=200)
    assert t["transfers"] and t["partial_ci"][0] > 0 and t["rho"] > 0.7


def test_transfer_not_shown_when_maps_only_follow_variance():
    rng = np.random.default_rng(4)
    var = rng.standard_normal(C)
    Md = var + 0.5 * rng.standard_normal((18, C))
    Mk = var + 0.5 * rng.standard_normal((16, C))
    Vd, Vk = var + 0.1 * rng.standard_normal((18, C)), var + 0.1 * rng.standard_normal((16, C))
    t = axt.transfer(Md, Mk, Vd, Vk, rng.standard_normal(C), n_boot=200)
    assert t["rho"] > 0.7                  # raw maps agree ...
    assert not t["transfers"]              # ... but only through the variance structure


def test_band_transfer_prediction():
    rng = np.random.default_rng(5)
    d = axt.by_subject(_dataset(18, 3, rng, np.zeros(C), np.zeros(C)))
    k = axt.by_subject(_dataset(16, 2, rng, np.zeros(C), np.zeros(C)))
    t = axt.band_transfer(d, k)
    assert t["prediction_holds"] and t["median"] > 0.9 and t["p"] < 0.001
    flipped = axt.by_subject(_dataset(16, 2, rng, np.zeros(C), np.zeros(C), drop=DROP[::-1]))
    t2 = axt.band_transfer(d, flipped)
    assert not t2["prediction_holds"] and t2["median"] < 0


# --- end to end ----------------------------------------------------------------------------------
def test_analyse_end_to_end(tmp_path):
    rng = np.random.default_rng(6)
    content, var = rng.standard_normal(C), rng.standard_normal(C)
    dtu = _dataset(18, 3, rng, content, var)
    kul = _dataset(16, 2, rng, content, var)
    lines = []
    res = axt.analyse(*_write(tmp_path, dtu, kul), out=lines.append)
    assert all(v["replicates"] for v in res["replication"].values())
    assert res["replication"]["R2 channel faithful occlusion"]["kul"]
    assert res["T1"]["prediction_holds"]
    assert set(res["T2"]) == {"occlusion", "ig", "attention"} and set(res["T3"]) == {"attention", "attn_x_grad"}
    assert any("Top-10" in ln for ln in lines)


def test_analyse_rejects_mismatched_dtu_variance(tmp_path):
    rng = np.random.default_rng(7)
    dtu = _dataset(18, 3, rng, np.zeros(C), np.zeros(C))
    kul = _dataset(16, 2, rng, np.zeros(C), np.zeros(C))
    dvar = [{"fold_id": r["fold_id"], "subject": r["subject"], "n_windows": r["n_windows"] + 1,
             "variance": r["variance"]} for r in dtu]
    with pytest.raises(AssertionError, match="variance windows"):
        axt.analyse(*_write(tmp_path, dtu, kul, dvar), out=lambda s: None)


def test_analyse_rejects_missing_kul_fold(tmp_path):
    rng = np.random.default_rng(8)
    dtu = _dataset(18, 3, rng, np.zeros(C), np.zeros(C))
    kul = _dataset(16, 2, rng, np.zeros(C), np.zeros(C))[:-1]
    with pytest.raises(AssertionError):
        axt.analyse(*_write(tmp_path, dtu, kul), out=lambda s: None)
