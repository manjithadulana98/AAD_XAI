"""Pre-registered planted-benchmark evaluation (aad_xai.xai.planted_eval)."""
from __future__ import annotations

import pytest

from aad_xai.xai import planted_eval as pe


def test_select_main_snr_lowest_reaching():
    acc = {-40.0: [0.5, 0.52, 0.51], -35.0: [0.6, 0.72, 0.75], -30.0: [0.9, 0.95, 0.92]}
    assert pe.select_main_snr(acc) == -35.0


def test_select_main_snr_non_monotone_takes_lowest():
    acc = {-40.0: [0.71, 0.72, 0.70], -35.0: [0.6, 0.65, 0.66], -30.0: [0.9, 0.9, 0.9]}
    assert pe.select_main_snr(acc) == -40.0


def test_select_main_snr_none_reaches():
    assert pe.select_main_snr({-40.0: [0.5, 0.6, 0.69], -30.0: [0.6, 0.6, 0.6]}) is None


def test_is_learned():
    assert pe.is_learned(0.60)
    assert not pe.is_learned(0.59)


@pytest.mark.parametrize("accs,expected", [([0.95, 0.97, 0.99], "near ceiling"),
                                           ([0.7, 0.72, 0.8, 0.5], "realistic"),
                                           ([0.52, 0.5, 0.84, 0.55], "spread")])
def test_regime(accs, expected):
    assert pe.regime(accs) == expected


def _rec(acc, auroc=0.95, decoy=40, attr_rank=1, stop_rank=1, gt=0.2, rnd=0.02, edge_auroc=0.95, val=0.9):
    m = {"band_attr_rank": attr_rank, "band_stop_rank": stop_rank, "edge_gt_drop": gt, "edge_random_drop": rnd}
    for c in pe.CH_METHODS:
        m[f"auroc_{c}"] = auroc
        m[f"decoy_rank_{c}"] = decoy
    for c in pe.EDGE_METHODS:
        m[f"edge_auroc_{c}"] = edge_auroc
    return {"test_acc": acc, "val": val, "metrics": m}


def test_verdicts_all_pass():
    main = {"a": [_rec(0.75, gt=0.2 + 0.01 * i) for i in range(10)],
            "b": [_rec(0.72, gt=0.3 + 0.01 * i) for i in range(10)]}
    v = pe.verdicts(main)
    assert all(x["V1"] and x["V2"] and x["V3"] for x in v["validity"].values())
    assert all(x["pass"] for x in v["electrodes"].values())
    assert all(x["pass"] for x in v["decoy"].values())
    assert v["bands"]["pass"]
    assert v["edges"] is not None
    assert all(x["pass"] for x in v["edges"].values())


def test_verdicts_failures():
    main = {"a": [_rec(0.75, auroc=0.6, attr_rank=3)] * 9 + [_rec(0.75, decoy=4)],
            "b": [_rec(0.65, gt=0.0, rnd=0.02)] * 10}
    v = pe.verdicts(main)
    assert not v["validity"]["b"]["V1"]
    assert not v["validity"]["b"]["V3"]
    assert v["edges"] is None                                             # V3 fails -> edges not scored
    assert not v["electrodes"]["ig"]["pass"]
    assert v["decoy"]["ig"]["best_rank"] == 4
    assert not v["decoy"]["ig"]["pass"]
    assert not v["bands"]["pass"]


def test_v3_needs_two_seeds():
    v = pe.verdicts({"a": [_rec(0.8)]})
    assert v["validity"]["a"]["p3"] == 1.0


@pytest.mark.parametrize("all_pass,learned_pass,expected", [(True, True, "pass"), (False, True, "model failure"),
                                                            (False, False, "method failure")])
def test_failure_kind(all_pass, learned_pass, expected):
    assert pe.failure_kind(all_pass, learned_pass) == expected
