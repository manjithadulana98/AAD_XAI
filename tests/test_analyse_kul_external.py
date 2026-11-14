"""The pre-registered KUL analysis (scripts/analyse_kul_external.py) on synthetic notebook outputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from aad_xai.data.negatives import partner_trial_permutation, same_stratum_other_content_partners
from aad_xai.evaluation.decision_windows import corr_from_stats, window_stats

_SPEC = importlib.util.spec_from_file_location(
    "analyse_kul_external", Path(__file__).resolve().parents[1] / "scripts" / "analyse_kul_external.py")
ak = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ak)

N5 = 60                      # 5 s windows per trial (300 s trials; real KUL ~78)
WIN = 5 * 64


def _trial_info():
    info = {}
    for s in range(1, 17):
        for k in range(8):
            exp, part = 1 + k // 4, 1 + k % 4
            ear = "R" if (part + s) % 2 else "L"
            left, right = (3 - exp, exp) if ear == "R" else (exp, 3 - exp)
            info[f"S{s}_T{k:03d}"] = {"subject": f"S{s}", "experiment": exp, "part": part, "story_part": part,
                                      "condition": "dry" if (part + s // 2) % 2 else "hrtf", "attended_ear": ear,
                                      "attended_track": exp, "left_track": left, "right_track": right}
    return info


def _grid(trials, info, n, step):
    subj = np.array([info[t]["subject"] for t in trials for _ in range(n)])
    trial = np.array([t for t in trials for _ in range(n)])
    start = np.array([j * step for _ in trials for j in range(n)])
    label = np.array([int(info[t]["attended_ear"] == "R") for t in trials for _ in range(n)])
    return subj, trial, start, label


def write_fake(root: Path, sgat=0.0, trf=0.0, leak=0.0, seed=0, break_swap=False):
    """Fake outputs of parts A, B and the TRF: SGAT logit difference = sgat * (2y - 1) + noise;
    controls = leak * (2y - 1) + noise; TRF attended envelope = trf * prediction + noise."""
    rng = np.random.default_rng(seed)
    info = _trial_info()
    dirs = {p: root / f"sgat_{p}" for p in "AB"}
    for d in list(dirs.values()) + [root / "trf"]:
        d.mkdir(parents=True)
    folds, trf_folds = {p: [] for p in "AB"}, {}
    for g in range(8):
        test_subj = [f"S{2 * g + 1}", f"S{2 * g + 2}"]
        for k, held in enumerate(((1, 2), (3, 4))):
            fid = f"sgso_g{g}_k{k}"
            trials = [t for t, i in info.items() if i["subject"] in test_subj and i["story_part"] in held]
            subj, trial, start, label = _grid(trials, info, N5, WIN)
            keys = {t: (info[t]["subject"], info[t]["experiment"], info[t]["story_part"]) for t in trials}
            perm = partner_trial_permutation(trial, start, same_stratum_other_content_partners(keys))
            if break_swap:
                perm = np.arange(len(trial))
            base = {"test_subject": subj, "test_trial": trial, "test_start": start, "test_label": label,
                    "val_label": rng.integers(0, 2, 20), "test_swap_perm": perm}
            for w in (1, 2):
                s_, t_, st_, l_ = _grid(trials, info, N5 * 5 // w, w * 64)
                base.update({f"test{w}s_subject": s_, f"test{w}s_trial": t_, f"test{w}s_start": st_,
                             f"test{w}s_label": l_})
            sign = 2 * label - 1
            for p, seeds in ak.SEEDS.items():
                arrays = dict(base)
                models = {}
                for sd in seeds:
                    m = f"kul_s{sd}"

                    def lg(eff, n_sign):
                        d = eff * n_sign + rng.standard_normal(len(n_sign))
                        return np.stack([np.zeros_like(d), d], axis=1)
                    arrays[f"{m}_test_logits"] = lg(sgat, sign)
                    arrays[f"{m}_test_shifted_logits"] = lg(leak, sign)
                    arrays[f"{m}_test_swapped_logits"] = lg(leak, sign)
                    for w in (1, 2):
                        sw = 2 * base[f"test{w}s_label"] - 1
                        arrays[f"{m}_test{w}s_logits"] = lg(sgat, sw)
                        arrays[f"{m}_test{w}s_shifted_logits"] = lg(leak, sw)
                    models[m] = {"best_epoch": 4, "val_balanced": 0.6}
                np.savez_compressed(dirs[p] / f"{fid}.npz", **arrays)
                folds[p].append({"fold_id": fid, "meta": {"test_subjects": test_subj}, "models": models})
            # TRF: K = 2 bands, short windows of T = 16 samples carry the statistics
            stats, shift_stats, score, shift_score = [], [], [], []
            for y in label:
                P = rng.standard_normal((16, 2))
                env = rng.standard_normal((2, 2, 16))
                env[y] += trf * P.T
                for e, st_l, sc_l in ((env, stats, score), (np.roll(env, 5, -1), shift_stats, shift_score)):
                    st = window_stats(P, e)
                    r = corr_from_stats(st, 16)
                    st_l.append(st); sc_l.append(r[1].mean() - r[0].mean())
            np.savez_compressed(root / "trf" / f"trf_ea_{fid}.npz", test_subject=subj, test_trial=trial,
                                test_start=start, test_label=label, test_score=np.array(score),
                                test_stats=np.array(stats), test_shift_score=np.array(shift_score),
                                test_shift_stats=np.array(shift_stats))
            trf_folds[fid] = {"val_acc": 0.6}
    for p in "AB":
        json.dump({"commit": "fake", "part": p, "trial_info": info, "folds": folds[p]},
                  open(dirs[p] / f"kul_external_sgat_{p}.json", "w"))
    json.dump({"commit": "fake", "trial_info": info, "folds": trf_folds}, open(root / "trf" / "kul_external_trf.json", "w"))
    return dirs["A"], dirs["B"], root / "trf"


def _run(tmp_path, **kw):
    return ak.analyse(*write_fake(tmp_path, **kw), out=lambda *a: None)


def test_null_outputs(tmp_path):
    res = _run(tmp_path, sgat=0.0, trf=0.0, leak=0.0, seed=1)
    assert not res["h1"]
    assert res["h2"] == "no difference"
    assert not res["claim"]


def test_strong_model_weak_baseline(tmp_path):
    res = _run(tmp_path, sgat=1.5, trf=0.0, leak=0.0, seed=2)
    assert res["h1"] and res["h2"] == "better" and res["controls_ok"] and res["claim"]


def test_leaky_controls_block_the_claim(tmp_path):
    res = _run(tmp_path, sgat=1.5, trf=0.0, leak=1.5, seed=3)
    assert res["h1"] and res["h2"] == "better"
    assert not res["controls_ok"] and not res["claim"]


def test_stronger_baseline_is_worse(tmp_path):
    res = _run(tmp_path, sgat=0.0, trf=1.0, leak=0.0, seed=4)
    assert res["h2"] == "worse" and not res["claim"]


def test_swap_integrity_check(tmp_path):
    with pytest.raises(AssertionError):
        _run(tmp_path, break_swap=True)


@pytest.mark.parametrize("diffs, ps, expected", [
    ([0.03, 0.02, 0.01, 0.0], [0.01, 0.2, 0.5, 1.0], "better"),
    ([0.03, 0.02, 0.01, -0.006], [0.01, 0.2, 0.5, 0.9], "no difference"),   # a drop > 0.5 points blocks "better"
    ([-0.03, 0.0, 0.0, 0.0], [0.01, 1.0, 1.0, 1.0], "worse"),
    ([0.03, -0.02, 0.0, 0.0], [0.01, 0.02, 1.0, 1.0], "mixed"),
    ([0.03, 0.02, 0.0, 0.0], [0.06, 0.2, 1.0, 1.0], "no difference"),
])
def test_h2_outcome_rules(diffs, ps, expected):
    assert ak.h2_outcome(diffs, ps) == expected


def test_holm():
    assert ak.holm([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.09, 0.09, 0.5])
