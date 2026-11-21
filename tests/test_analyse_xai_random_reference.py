"""Architecture-only reference analysis (scripts/analyse_xai_random_reference.py) on synthetic outputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


arr = _load("analyse_xai_random_reference", _ROOT / "scripts" / "analyse_xai_random_reference.py")
fx = _load("test_analyse_xai_transfer_fixture", _ROOT / "tests" / "test_analyse_xai_transfer.py")
C, E, SEEDS = fx.C, fx.E, fx.SEEDS


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(arr, "N_BOOT", 200)
    monkeypatch.setattr(arr.xt, "N_SPLITS", 50)


def _random_records(trained, rng, ch_map, edge_map, n_windows=None):
    noisy = lambda v, s=0.3: (np.asarray(v) + s * rng.standard_normal(np.shape(v))).tolist()
    return [{"fold_id": r["fold_id"], "subject": r["subject"], "n_windows": n_windows or r["n_windows"],
             "rand_seeds": {s: 0 for s in SEEDS}, "rand_acc": {s: 0.5 for s in SEEDS},
             "rand_scores": {s: {m: noisy(ch_map) for m in ("occlusion", "ig", "attention")} for s in SEEDS},
             "rand_edge_scores": {m: noisy(edge_map) for m in ("attention", "attn_x_grad")}}
            for r in trained]


def _write(tmp_path, shared_random, break_windows=False, seed=0):
    """Trained maps share a learned pattern across datasets. Random maps share an architecture
    pattern across datasets (shared_random) or are unrelated between the datasets."""
    rng = np.random.default_rng(seed)
    learned, arch = rng.standard_normal(C), rng.standard_normal(C)
    var_d, var_k = rng.standard_normal(C), rng.standard_normal(C)
    dtu = fx._dataset(18, 3, rng, learned, var_d)
    kul = fx._dataset(16, 2, rng, learned, var_k)
    paths = fx._write(tmp_path, dtu, kul)
    edge_trained = np.linspace(1, 0, E)
    if shared_random:
        maps = {"DTU": (learned, edge_trained), "KUL": (learned, edge_trained)}   # architecture explains all
    else:
        maps = {"DTU": (arch, rng.standard_normal(E)), "KUL": (rng.standard_normal(C), rng.standard_normal(E))}
    for name, trained in (("DTU", dtu), ("KUL", kul)):
        d = tmp_path / f"rand{name}" / f"results_xai_random_{name}"
        d.mkdir(parents=True)
        recs = _random_records(trained, rng, *maps[name],
                               n_windows=21 if (break_windows and name == "KUL") else None)
        (d / f"xai_random_{name}.json").write_text(json.dumps(
            {"commit": "test", "dataset": name, "edges": fx.EDGES, "results": recs}))
    return paths + [str(tmp_path / "randDTU"), str(tmp_path / "randKUL")]


def test_learned_agreement_is_beyond_the_architecture(tmp_path):
    res = arr.analyse(*_write(tmp_path, shared_random=False, seed=1), out=lambda *a: None)
    for m in ("occlusion", "ig", "attention"):
        assert res["A1"][m]["beyond_architecture"], (m, res["A1"][m])
        assert abs(res["A1"][m]["random"]) < 0.4
    assert all(res["A2"][m]["beyond_architecture"] for m in ("attention", "attn_x_grad"))
    assert res["rand_acc"] == {"DTU": 0.5, "KUL": 0.5}


def test_agreement_the_architecture_already_gives_is_not_beyond(tmp_path):
    res = arr.analyse(*_write(tmp_path, shared_random=True, seed=2), out=lambda *a: None)
    for m in ("occlusion", "ig", "attention"):
        assert res["A1"][m]["random"] > 0.5
        assert not res["A1"][m]["beyond_architecture"], (m, res["A1"][m])
        assert res["A1"][m]["trained_vs_random"]["DTU"] > 0.8
    assert not any(res["A2"][m]["beyond_architecture"] for m in ("attention", "attn_x_grad"))


def test_window_mismatch_is_caught(tmp_path):
    with pytest.raises(AssertionError):
        arr.analyse(*_write(tmp_path, shared_random=False, break_windows=True), out=lambda *a: None)


def test_paired_transfer_difference_is_trained_minus_random():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(30)
    Td = base + 0.2 * rng.standard_normal((10, 30))
    Tk = base + 0.2 * rng.standard_normal((12, 30))
    Rd, Rk = rng.standard_normal((10, 30)), rng.standard_normal((12, 30))
    t = arr.paired_transfer(Td, Tk, Rd, Rk, n_boot=100)
    assert t["diff"] == pytest.approx(t["trained"] - t["random"])
    assert t["trained"] > 0.8 and t["beyond_architecture"]
