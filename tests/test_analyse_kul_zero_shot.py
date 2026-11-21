"""The DTU -> KUL zero-shot analysis (scripts/analyse_kul_zero_shot.py) on synthetic notebook outputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from aad_xai.evaluation.decision_windows import corr_from_stats, window_stats

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zs = _load("analyse_kul_zero_shot", _ROOT / "scripts" / "analyse_kul_zero_shot.py")
fake_kul = _load("test_analyse_kul_external_fixture", _ROOT / "tests" / "test_analyse_kul_external.py")

DTU_FOLDS = [f"sgso_g{g}_k{k}" for g in range(6) for k in range(3)]
DTU_SEEDS = [42, 1, 2, 3, 4]


def _trf_arrays(rng, label, eff):
    """K = 2 bands, windows of T = 16 samples; attended envelope = eff * reconstruction + noise."""
    stats, shift_stats, score, shift_score = [], [], [], []
    for y in label:
        P = rng.standard_normal((16, 2))
        env = rng.standard_normal((2, 2, 16))
        env[y] += eff * P.T
        for e, st_l, sc_l in ((env, stats, score), (np.roll(env, 5, -1), shift_stats, shift_score)):
            st = window_stats(P, e)
            r = corr_from_stats(st, 16)
            st_l.append(st); sc_l.append(r[1].mean() - r[0].mean())
    return {"test_score": np.array(score), "test_stats": np.array(stats),
            "test_shift_score": np.array(shift_score), "test_shift_stats": np.array(shift_stats)}


def write_zero_shot(root, kul_a, kul_trf, sgat=0.0, trf=0.0, leak=0.0, seed=0, break_windows=False):
    """Fake zero-shot outputs on the windows of the fake KUL external test."""
    rng = np.random.default_rng(seed)
    d_sg, d_tr = root / "zs_sgat", root / "zs_trf"
    d_sg.mkdir(); d_tr.mkdir()
    info = json.load(open(kul_a / "kul_external_sgat_A.json"))["trial_info"]
    folds = []
    for f in sorted(kul_a.glob("sgso_*.npz")):
        fid = f.stem
        with np.load(f) as z:
            meta = {k: z[k] for k in ("test_subject", "test_trial", "test_start", "test_label", "test_swap_perm")}
        if break_windows and fid == "sgso_g0_k0":
            meta["test_start"] = meta["test_start"] + 1
        sign = 2 * meta["test_label"] - 1
        arrays = dict(meta)
        for fo in DTU_FOLDS:
            for sd in DTU_SEEDS:
                for kind, eff in (("", sgat), ("_shifted", leak), ("_swapped", leak)):
                    d = eff * sign + 2.0 * rng.standard_normal(len(sign))
                    arrays[f"dtu_s{sd}_{fo}_test{kind}_logits"] = np.stack([np.zeros_like(d), d], axis=1)
        np.savez_compressed(d_sg / f"{fid}.npz", **arrays)
        with np.load(kul_trf / f"trf_ea_{fid}.npz") as t:
            tmeta = {k: t[k] for k in ("test_subject", "test_trial", "test_start", "test_label")}
        tarr = dict(tmeta)
        tarr.update(_trf_arrays(rng, tmeta["test_label"], trf))
        tarr["test_decoder_scores"] = (tarr["test_score"][:, None]
                                       + 0.1 * rng.standard_normal((len(tmeta["test_label"]), 18)))
        np.savez_compressed(d_tr / f"trf_zs_{fid}.npz", **tarr)
        folds.append({"fold_id": fid})
    cfg = {"dtu_folds": DTU_FOLDS, "dtu_seeds": DTU_SEEDS}
    json.dump({"commit": "fake", "config": cfg, "trial_info": info, "folds": folds},
              open(d_sg / "kul_zero_shot_sgat.json", "w"))
    json.dump({"commit": "fake", "config": {}, "trial_info": info, "folds": folds,
               "decoders": {fo: {"std": False} for fo in DTU_FOLDS}}, open(d_tr / "kul_zero_shot_trf.json", "w"))
    return d_sg, d_tr


def _run(tmp_path, kul_sgat=1.5, kul_trf=0.5, **kw):
    a, b, t = fake_kul.write_fake(tmp_path, sgat=kul_sgat, trf=kul_trf, seed=kw.get("seed", 0))
    d_sg, d_tr = write_zero_shot(tmp_path, a, t, **kw)
    return zs.analyse(d_sg, d_tr, a, b, t, out=lambda *x: None)


def test_null_zero_shot(tmp_path):
    res = _run(tmp_path, sgat=0.0, trf=0.0, leak=0.0, seed=1)
    assert not res["z1_5s"] and not res["claim"]
    assert all(d > 0 for d in res["z2"]["diffs"])           # KUL-trained models are better


def test_working_zero_shot_beats_null_trf(tmp_path):
    res = _run(tmp_path, sgat=1.0, trf=0.0, leak=0.0, seed=2)
    assert res["z1_5s"] and res["controls_ok"] and res["claim"]
    assert res["z3"]["outcome"] == "better"


def test_leaky_controls_block_the_claim(tmp_path):
    res = _run(tmp_path, sgat=1.0, trf=0.0, leak=1.0, seed=3)
    assert res["z1_5s"] and not res["controls_ok"] and not res["claim"]


def test_network_losing_more_than_trf(tmp_path):
    # KUL-trained: SGAT strong, TRF moderate; zero-shot: SGAT at chance, TRF unchanged.
    res = _run(tmp_path, kul_sgat=1.5, kul_trf=0.5, sgat=0.0, trf=0.5, seed=4)
    assert res["z4"]["diffs"][0] > 0 and res["z4"]["holm"][0] < 0.05
    assert res["z3"]["outcome"] == "worse"


def test_window_mismatch_is_caught(tmp_path):
    with pytest.raises(AssertionError):
        _run(tmp_path, break_windows=True)


def test_per_fold_and_single_models_reported(tmp_path):
    res = _run(tmp_path, sgat=1.0, seed=5)
    assert len(res["per_dtu_fold"]) == 18
    # 90-model averaging beats a single noisy model
    assert np.mean(list(res["per_dtu_fold"].values())) > res["single_mean"]
