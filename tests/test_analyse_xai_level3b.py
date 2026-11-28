"""Level-3b analysis (scripts/analyse_xai_level3b.py) on synthetic outputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from aad_xai.xai import planted_eval as pe

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyse_xai_level3b", _ROOT / "scripts" / "analyse_xai_level3b.py")
al = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(al)

GRID = [-40.0, -35.0, -32.5, -30.0, -27.5, -25.0, -22.5, -20.0]


def _metrics(auroc, decoy, attn_auroc=0.6, gt=0.2):
    m = {"band_attr_rank": 3, "band_stop_rank": 1, "edge_gt_drop": gt, "edge_random_drop": 0.02}
    for c in pe.CH_METHODS:
        m[f"auroc_{c}"] = attn_auroc if c == "attention" else auroc
        m[f"decoy_rank_{c}"] = decoy
    for c in pe.EDGE_METHODS:
        m[f"edge_auroc_{c}"] = 0.6
    return m


def _rec(cond, snr, seed, stage, acc, val=0.8, auroc=0.95, decoy=40):
    return {"condition": cond, "snr_db": snr, "seed": seed, "stage": stage, "test_acc": acc, "val": val,
            "metrics": _metrics(auroc, decoy, gt=0.2 + 0.01 * seed)}


def _acc_curve(snr, cross):
    """Accuracy 0.5 below the crossing SNR, 0.75 at it, 1.0 at -20 dB and 0.9 in between."""
    if snr < cross:
        return 0.5
    return 1.0 if snr == -20.0 else (0.75 if snr == cross else 0.9)


def _part(cond, cross, *, unlearned_main=0, decoy_main=40, null=False, notebook_main=None):
    rs = [_rec(cond, s, seed, "sweep", _acc_curve(s, cross)) for s in GRID for seed in range(3)]
    for seed in range(3, 10):
        bad = seed - 3 < unlearned_main
        rs.append(_rec(cond, cross, seed, "main", 0.52 if bad else 0.74, val=0.5 if bad else 0.8,
                       auroc=0.5 if bad else 0.95, decoy=decoy_main if not bad else 3))
    if null:
        rs += [{"condition": "null", "snr_db": None, "seed": s, "stage": "null", "test_acc": 0.5, "val": 0.5,
                "metrics": {f"decoy_rank_{m}": 30 for m in pe.CH_METHODS}} for s in range(3)]
    return rs, {cond: cross if notebook_main is None else notebook_main}


def _files(tmp_path, a_kw=None, b_kw=None):
    ra, ma = _part("delta_T7", -30.0, null=True, **(a_kw or {}))
    rb, mb = _part("theta_P8", -32.5, **(b_kw or {}))
    l3 = [_rec(c, -20.0, s, "main", 1.0) for c in ("delta_T7", "theta_P8") for s in range(10)]
    paths = []
    for name, obj in (("a.json", {"commit": "A", "main_snr_db": ma, "config": {"part": "A"}, "results": ra}),
                      ("b.json", {"commit": "B", "main_snr_db": mb, "config": {"part": "B"}, "results": rb}),
                      ("l3.json", {"commit": "L3", "main_snr_db": -20.0, "results": l3})):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        paths.append(p)
    return paths


def _run(tmp_path, **kw):
    return al.analyse(*_files(tmp_path, **kw), out=lambda *_: None)


def test_main_snr_and_regime(tmp_path):
    r = _run(tmp_path)
    assert r["main_snr"] == {"delta_T7": -30.0, "theta_P8": -32.5}
    assert r["regime"] == {"delta_T7": "realistic", "theta_P8": "realistic"}
    assert r["learned"] == {"delta_T7": 10, "theta_P8": 10}


def test_verdicts_and_level3_comparison(tmp_path):
    r = _run(tmp_path)
    v = r["verdicts_all"]
    assert v["electrodes"]["ig"]["pass"]
    assert not v["electrodes"]["attention"]["pass"]
    assert all(x["level3"] == x["level3b"] for x in r["vs_level3"].values())


def test_unlearned_seeds_give_model_failure(tmp_path):
    r = _run(tmp_path, a_kw={"unlearned_main": 3})                        # 3 of 10 seeds learn nothing
    assert r["learned"]["delta_T7"] == 7
    assert not r["verdicts_all"]["decoy"]["ig"]["pass"]                   # unlearned seeds put Cz in the top 6
    assert r["verdicts_learned"]["decoy"]["ig"]["pass"]
    assert r["failure_kind"]["decoy ig"] == "model failure"
    assert r["failure_kind"]["electrodes attention"] == "method failure"


def test_gate_notebook_choice_must_match(tmp_path):
    with pytest.raises(AssertionError):
        _run(tmp_path, a_kw={"notebook_main": -27.5})


def test_parts_must_be_labelled(tmp_path):
    a, b, l3 = _files(tmp_path)
    with pytest.raises(AssertionError):
        al.analyse(b, a, l3, out=lambda *_: None)
