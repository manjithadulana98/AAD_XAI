"""Calibration-EA analysis (scripts/analyse_calibration_ea.py) on synthetic notebook outputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyse_calibration_ea", _ROOT / "scripts" / "analyse_calibration_ea.py")
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)

SD_ARMS = ["full", "none", "st10", "st30", "st60", "stall", "os60"]
CT_ARMS = ["full", "none", "st10", "st30", "st60", "stall"]
SEEDS = [42, 1, 2, 3, 4]
SUBJ = [f"S{i}" for i in range(1, 19)]


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(ac, "N_BOOT", 400)


def _logits(diff):
    return np.stack([np.zeros_like(diff), diff], axis=1).astype(np.float32)        # (N, 2)


def write(tmp, strength, trf_strength=0.4, noea_strength=0.6, leak=0.0, gate=1.0, seed=0):
    """strength: {arm: per-window signal of StimulusGAT}; leak: signal in the shifted control."""
    rng = np.random.default_rng(seed)
    sg, tr = tmp / "sgat" / "results_calibration_ea", tmp / "trf" / "results_trf_calibration_ea"
    na, nb = tmp / "noeaA" / "results_loso_catcn_A", tmp / "noeaB" / "results_loso_catcn_B"
    for d in (sg, tr, na, nb):
        d.mkdir(parents=True)
    subj_ability = {s: rng.uniform(0.5, 1.5) for s in SUBJ}
    sd_recs, tr_recs = [], {}
    for g in range(6):
        for k in range(3):
            fid = f"sgso_g{g}_k{k}"
            meta = {"subject": [], "trial": [], "start": [], "label": []}
            for s in SUBJ[3 * g:3 * g + 3]:
                for t in range(10):
                    y = int(rng.integers(2))
                    for w in range(8):
                        meta["subject"].append(s); meta["trial"].append(f"{s}_T{k}{t}")
                        meta["start"].append(w * 320); meta["label"].append(y)
            m = {f"test_{a}": np.array(v) for a, v in meta.items()}
            sign = 2 * m["test_label"] - 1
            ab = np.array([subj_ability[s] for s in m["test_subject"]])
            n = len(sign)
            arrays = dict(m)
            for arm in SD_ARMS:
                for s in SEEDS:
                    arrays[f"{arm}_s{s}_test_logits"] = _logits(strength.get(arm, 0.3) * ab * sign + rng.standard_normal(n))
                    if arm in ("full", "st60"):
                        arrays[f"{arm}_s{s}_test_shifted_logits"] = _logits(leak * sign + rng.standard_normal(n))
                        arrays[f"{arm}_s{s}_test_swapped_logits"] = _logits(rng.standard_normal(n))
            np.savez(sg / f"sd_{fid}.npz", **arrays)
            t_arrays = dict(m)
            for arm in SD_ARMS:
                t_arrays[f"{arm}_score"] = trf_strength * ab * sign + rng.standard_normal(n)
                t_arrays[f"{arm}_stats"] = np.zeros((n, 2, 15, 5))
                if arm in ("full", "st60"):
                    t_arrays[f"{arm}_shift_score"] = rng.standard_normal(n)
            np.savez(tr / f"trf_cal_{fid}.npz", **t_arrays)
            sd_recs.append({"fold_id": fid})
            tr_recs[fid] = {"variant": "raw", "alpha": 1.0}

    ct_recs, noea_recs = [], {"A": [], "B": []}
    n_tr, T = 7, 3200
    for s in SUBJ:
        for k in range(8):
            fid = f"loso_{s}_f{k}"
            tids = np.array([f"{s}_T{k}{j}" for j in range(n_tr)])
            labels = rng.integers(0, 2, n_tr)
            base = {"trial_ids": tids, "trial_lens": np.full(n_tr, T)}
            for name, (win, hop) in {"test5": (320, 160), "test1": (64, 32), "test2": (128, 64)}.items():
                starts = np.arange(0, T - win + 1, hop)
                base[f"{name}_trial"] = np.repeat(tids, len(starts))
                base[f"{name}_start"] = np.tile(starts, n_tr)
                base[f"{name}_label"] = np.repeat(labels, len(starts))
                base[f"{name}_subject"] = np.full(n_tr * len(starts), s)
            cal, no = dict(base), dict(base)
            for name in ("test5", "test1", "test2"):
                sign = 2 * base[f"{name}_label"] - 1
                n = len(sign)
                for arm in CT_ARMS:
                    cal[f"{arm}_s42_{name}_logits"] = _logits(strength.get(arm, 0.3) * subj_ability[s] * sign + rng.standard_normal(n))
                no[f"loso_s42_{name}_logits"] = _logits(noea_strength * subj_ability[s] * sign + rng.standard_normal(n))
            n5 = len(base["test5_label"])
            for arm in ("full", "st60"):
                cal[f"{arm}_s42_test5_shifted_logits"] = _logits(leak * (2 * base["test5_label"] - 1) + rng.standard_normal(n5))
                cal[f"{arm}_s42_test5_swapped_logits"] = _logits(rng.standard_normal(n5))
            np.savez(sg / f"ct_{fid}.npz", **cal)
            part = "A" if int(s[1:]) <= 9 else "B"
            np.savez((na if part == "A" else nb) / f"{fid}.npz", **no)
            rec = {"fold_id": fid, "meta": {"test_subjects": [s]}}
            ct_recs.append(rec)
            noea_recs[part].append(rec)
    (sg / "calibration_ea.json").write_text(json.dumps({
        "commit": "test", "config": {"sd_arms": SD_ARMS, "ct_arms": CT_ARMS, "sd_seeds": SEEDS, "gate_agreement": 0.999},
        "single_talker": {s: {"trial_index": [0, 1], "total_s": 100.0} for s in SUBJ},
        "gate": {"SD": {"agreement": gate}, "CT": {"agreement": 1.0}},
        "folds": {"SD": sd_recs, "CT": ct_recs}}))
    (tr / "trf_calibration_ea.json").write_text(json.dumps({
        "commit": "test", "config": {"gate_agreement": 0.999}, "gate": {"agreement": 1.0, "same_selection": True},
        "folds": tr_recs}))
    for part, d in (("A", na), ("B", nb)):
        (d / f"loso_catcn_{part}.json").write_text(json.dumps({"folds": noea_recs[part]}))
    return [str(tmp / x) for x in ("sgat", "trf", "noeaA", "noeaB")]


def test_calibration_that_matches_full_ea_gives_both_claims(tmp_path):
    res = ac.analyse(*write(tmp_path, {"full": 1.0, "st60": 1.0, "stall": 1.0, "none": 0.4}), out=lambda *a: None)
    assert res["C1"]["outcome"] == "no loss"
    assert res["C2"]["outcome"] == "better"
    assert res["sd_controls_ok"] and res["sd_claim"]
    assert res["ct_verdicts"]["full"] == ["above"] * 6 and res["ct_claim"]
    assert res["L2"]["helps"]
    assert res["sd_table"]["none"]["sgat"][0] < res["sd_table"]["full"]["sgat"][0]


def test_weaker_calibration_is_a_loss_and_breaks_parity(tmp_path):
    res = ac.analyse(*write(tmp_path, {"full": 1.0, "st60": 0.15}, noea_strength=0.6, seed=1), out=lambda *a: None)
    assert res["C1"]["outcome"] == "loss" and not res["sd_claim"]
    assert not res["ct_claim"]
    assert not res["L2"]["helps"]


def test_leaky_control_blocks_the_claims(tmp_path):
    res = ac.analyse(*write(tmp_path, {"full": 1.0, "st60": 1.0}, leak=0.5, seed=2), out=lambda *a: None)
    assert res["C1"]["outcome"] == "no loss"
    assert not res["sd_controls_ok"] and not res["sd_claim"]
    assert not res["ct_controls_ok"] and not res["ct_claim"]


def test_failed_gate_stops_the_analysis(tmp_path):
    with pytest.raises(AssertionError):
        ac.analyse(*write(tmp_path, {"full": 1.0, "st60": 1.0}, gate=0.99, seed=3), out=lambda *a: None)


def test_c1_outcome_rule():
    assert ac.c1_outcome([-0.004, 0.0, 0.01, -0.002], [1, 1, 1, 1]) == "no loss"
    assert ac.c1_outcome([-0.006, 0.0, 0.01, 0.0], [1, 1, 1, 1]) == "loss"           # drop beyond the margin
    assert ac.c1_outcome([-0.004, 0.0, 0.0, 0.0], [0.01, 1, 1, 1]) == "loss"         # significant drop
    assert ac.c1_outcome([0.02, 0.0, 0.0, 0.0], [0.01, 1, 1, 1]) == "no loss"


def test_verdict():
    assert ac.verdict(0.70, (0.71, 0.75)) == "above"
    assert ac.verdict(0.70, (0.65, 0.69)) == "below"
    assert ac.verdict(0.70, (0.68, 0.72)) == "level"
