"""Calibration-recording alignment instead of transductive EA (exploratory, DTU).

usage: python -I scripts/analyse_calibration_ea.py <sgat dir> <trf dir> <no-EA CT dir A> <no-EA CT dir B>

Inputs: the outputs of notebooks/kaggle_run_dtu_calibration_ea.py (sgat dir: parts SD and CT),
notebooks/kaggle_run_dtu_trf_calibration_ea.py (trf dir) and the no-EA run of the CA-TCN
protocol (kernels loso-ca-tcn-protocol-a / -b). Written and tested on synthetic outputs before
any calibration result existed; the rules are not to be changed after the runs.

Arms: full = EA from the scored test EEG (transductive, as reported so far); st10 / st30 / st60 /
stall = EA from the first 10 / 30 / 60 s or all of the subject's single-talker trials; os60 =
the first 60 s of the subject's unscored two-talker trials (SD only); none = no test alignment.

Story-disjoint (SD; 18 subjects; StimulusGAT = 5-seed logit-averaged ensemble; decisions over
L = 5/10/20/40 s by the sign of the summed 5 s scores, for both models):
- Gates: both notebooks reproduced the original runs (recorded in their JSON files).
- C1 (primary): st60 minus full, per-subject Wilcoxon, Holm over 4 lengths. "no loss" iff the
  mean difference is >= -0.5 points at every length and no length has a Holm-significant drop;
  otherwise "loss".
- C2: st60 StimulusGAT minus st60 TRF, the H2 outcome rule of the KUL test (Holm over 4).
- Controls (st60 ensemble, shifted and swapped envelopes): subject Wilcoxon vs 0.5 p >= 0.05 at
  every length and no fold >= 0.6 at 5 s.
- Claim "calibration EA keeps the story-disjoint result" iff C1 "no loss", C2 "better", controls
  clean.
CA-TCN protocol (CT; seed 42; accuracy per fold, mean over a subject's 8 folds, mean over
subjects; 1 s and 2 s scored directly, L >= 5 from the 5 s grid with 50% overlap by summed
logit differences, as analyse_loso.py):
- L1: verdict vs the published CA-TCN mean per length from the 95% subject-bootstrap CI ("above"
  if CA-TCN < CI, "below" if CA-TCN > CI, else "level"). Claim "parity with CA-TCN without
  transductive alignment" iff st60 is "level" or "above" at every length where full is, and the
  st60 controls are clean (5 s, subject Wilcoxon vs 0.5 p >= 0.05 and no subject >= 0.6).
- L2: st60 minus the no-EA run, per-subject Wilcoxon, Holm over the 6 lengths; calibration EA
  "helps" iff positive with Holm p < 0.05 at some length.
- Descriptive: every arm at every length on both protocols; st60 minus full on CT.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aad_xai.evaluation.decision_windows import group_windows, summed_score_decisions  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("analyse_kul_external",
                                               Path(__file__).resolve().parent / "analyse_kul_external.py")
ak = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ak)
holm, _wil, h2_outcome = ak.holm, ak._wil, ak.h2_outcome

SD_LENGTHS = [5, 10, 20, 40]
CT_LENGTHS = [1, 2, 5, 10, 25, 50]
SF, SUB, WINDOW_S = 64, 5 * 64, 5
PRIMARY = "st60"
MAX_DROP = 0.005
CATCN = {1: 0.583, 2: 0.634, 5: 0.703, 10: 0.767, 25: 0.858, 50: 0.925}
N_BOOT = 10000


def _one(d, pattern):
    hits = glob.glob(str(Path(d) / "**" / pattern), recursive=True)
    assert len(hits) == 1, (d, pattern, hits)
    return Path(hits[0])


class Outputs:
    def __init__(self, sgat_dir, trf_dir, noea_a, noea_b):
        self.sg_json_path = _one(sgat_dir, "calibration_ea.json")
        self.sg = json.load(open(self.sg_json_path))
        self.tr_json_path = _one(trf_dir, "trf_calibration_ea.json")
        self.tr = json.load(open(self.tr_json_path))
        self.sd_dir, self.tr_dir = self.sg_json_path.parent, self.tr_json_path.parent
        self.sd_fids = [r["fold_id"] for r in self.sg["folds"]["SD"]]
        self.ct_folds = self.sg["folds"]["CT"]
        self.seeds = self.sg["config"]["sd_seeds"]
        self.noea = {}
        for d in (noea_a, noea_b):
            self.noea.update({Path(p).stem: Path(p) for p in glob.glob(str(Path(d) / "**" / "loso_*_f*.npz"),
                                                                        recursive=True)})
        self.noea_folds = {}
        for d in (noea_a, noea_b):
            for js in glob.glob(str(Path(d) / "**" / "loso_catcn_[AB].json"), recursive=True):
                self.noea_folds.update({f["fold_id"]: f for f in json.load(open(js))["folds"]})
        self._cache = {}

    def _load(self, path):
        if path not in self._cache:
            with np.load(path) as z:
                self._cache[path] = {k: z[k] for k in z.files}
        return self._cache[path]

    def sd(self, fid):
        return self._load(self.sd_dir / f"sd_{fid}.npz")

    def trf(self, fid):
        return self._load(self.tr_dir / f"trf_cal_{fid}.npz")

    def ct(self, fid):
        return self._load(self.sd_dir / f"ct_{fid}.npz")

    def checks(self):
        g = self.sg["gate"]
        assert g["SD"]["agreement"] >= self.sg["config"]["gate_agreement"], "SD gate failed"
        assert g["CT"]["agreement"] >= self.sg["config"]["gate_agreement"], "CT gate failed"
        assert self.tr["gate"]["agreement"] >= self.tr["config"]["gate_agreement"] and self.tr["gate"]["same_selection"]
        assert len(self.sd_fids) == 18 and sorted(self.tr["folds"]) == sorted(self.sd_fids)
        assert len(self.ct_folds) == 144
        for fid in self.sd_fids:
            s, t = self.sd(fid), self.trf(fid)
            for k in ("subject", "trial", "start", "label"):
                assert np.array_equal(s[f"test_{k}"], t[f"test_{k}"]), (fid, k)
        assert set(self.noea) >= {r["fold_id"] for r in self.ct_folds}, "no-EA outputs missing folds"

    # --- story-disjoint -------------------------------------------------------------------------
    def _subject_acc(self, per_fold, L, fold_acc=None):
        """per_fold(fid) -> (scores (N,), z with test_* meta); per-subject accuracy over L s."""
        counts = defaultdict(lambda: [0, 0])
        for fid in self.sd_fids:
            score, z = per_fold(fid)
            groups = group_windows(z["test_subject"], z["test_trial"], z["test_start"], z["test_label"], L // WINDOW_S)
            dec = summed_score_decisions(score, groups)
            c_f = n_f = 0
            for (s, y, _), d in zip(groups, dec):
                counts[str(s)][0] += int(d == y); counts[str(s)][1] += 1
                c_f += int(d == y); n_f += 1
            if fold_acc is not None:
                fold_acc.append(c_f / n_f)
        subj = sorted(counts, key=lambda s: int(s[1:]))
        return np.array([counts[s][0] / counts[s][1] for s in subj])

    def sgat(self, arm, L=5, kind="", fold_acc=None):
        def per_fold(fid):
            z = self.sd(fid)
            return np.mean([np.diff(z[f"{arm}_s{s}_test{kind}_logits"], axis=1)[:, 0] for s in self.seeds], axis=0), z
        return self._subject_acc(per_fold, L, fold_acc)

    def trf_acc(self, arm, L=5, kind=""):
        return self._subject_acc(lambda fid: (self.trf(fid)[f"{arm}_{kind}score"], self.trf(fid)), L)

    # --- CA-TCN protocol ------------------------------------------------------------------------
    @staticmethod
    def _fold_acc(z, name, L, kind="logits"):
        if L in (1, 2):
            return float(np.mean(z[f"{name}_test{L}_{kind}"].argmax(1) == z[f"test{L}_label"]))
        diff = np.diff(z[f"{name}_test5_{kind}"], axis=1)[:, 0]
        where = {(str(t), int(s)): i for i, (t, s) in enumerate(zip(z["test5_trial"], z["test5_start"]))}
        lab = {str(t): int(y) for t, y in zip(z["test5_trial"], z["test5_label"])}
        n_samp, step, ok = L * SF, L * SF // 2, []
        for tid, n in zip(z["trial_ids"], z["trial_lens"]):
            tid = str(tid)
            if n < n_samp:
                continue
            for j in range(int((n - n_samp) / step) + 1):
                idx = [where[(tid, j * step + q * SUB)] for q in range(L // 5)]
                ok.append(int((diff[idx].sum() > 0) == bool(lab[tid])))
        return float(np.mean(ok)) if ok else np.nan

    def ct_subject(self, arm, L, kind="logits", run="cal"):
        acc = defaultdict(list)
        for f in self.ct_folds:
            if run == "cal":
                a = self._fold_acc(self.ct(f["fold_id"]), f"{arm}_s42", L, kind)
            else:
                a = self._fold_acc(self._load(self.noea[f["fold_id"]]), "loso_s42", L, kind)
            if not np.isnan(a):
                acc[f["meta"]["test_subjects"][0]].append(a)
        subj = sorted(acc, key=lambda s: int(s[1:]))
        return np.array([np.mean(acc[s]) for s in subj])

    def check_noea_windows(self):
        for f in self.ct_folds:
            a, b = self.ct(f["fold_id"]), self._load(self.noea[f["fold_id"]])
            for k in ("test5_trial", "test5_start", "test5_label", "test1_label", "test2_label", "trial_ids"):
                assert np.array_equal(a[k], b[k]), (f["fold_id"], k)


def boot_ci(x, n=None, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n or N_BOOT, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def verdict(ref, ci):
    return "above" if ref < ci[0] else "below" if ref > ci[1] else "level"


def c1_outcome(diffs, holm_ps, alpha=0.05):
    drop = any(d < 0 and p < alpha for d, p in zip(diffs, holm_ps))
    return "no loss" if all(d >= -MAX_DROP for d in diffs) and not drop else "loss"


def _paired(rows, out, label):
    """rows: [(L, a, b)] -> Wilcoxon of a - b, Holm; prints and returns diffs, p, holm."""
    ps = [_wil(a - b) for _, a, b in rows]
    hp = holm(ps)
    diffs = [float(np.mean(a - b)) for _, a, b in rows]
    out(f"  {label}")
    for (L, a, b), d, p, q in zip(rows, diffs, ps, hp):
        out(f"   {L:>2} s  {a.mean():.4f} vs {b.mean():.4f}  diff {d:+.4f}  higher {int((a > b).sum())}/{len(a)} "
            f"lower {int((a < b).sum())}/{len(a)}  p={p:.3g}  Holm {q:.3g}")
    return {"diffs": diffs, "p": ps, "holm": hp}


def analyse(sgat_dir, trf_dir, noea_a, noea_b, out=print) -> dict:
    O = Outputs(sgat_dir, trf_dir, noea_a, noea_b)
    O.checks()
    O.check_noea_windows()
    res = {}
    g = O.sg["gate"]
    out(f"commits: StimulusGAT {O.sg['commit']}; TRF {O.tr['commit']}")
    out(f"gates: SD {g['SD']['agreement']:.5f}, CT {g['CT']['agreement']:.5f}, TRF {O.tr['gate']['agreement']:.5f} "
        f"(same TRF selection in every fold: {O.tr['gate']['same_selection']})")
    st = O.sg["single_talker"]
    tot = [v["total_s"] for v in st.values()]
    out(f"single-talker EEG per subject: {min(tot):.0f}-{max(tot):.0f} s in "
        f"{min(len(v['trial_index']) for v in st.values())}-{max(len(v['trial_index']) for v in st.values())} trials; "
        f"first at recording position {min(v['trial_index'][0] for v in st.values())}-"
        f"{max(v['trial_index'][0] for v in st.values())}")

    out("\n=== Story-disjoint: accuracy by arm (18 subjects) ===")
    arms = O.sg["config"]["sd_arms"]
    res["sd_table"] = {}
    out(f"  {'arm':6s}  " + "  ".join(f"SGAT {L:>2}s" for L in SD_LENGTHS) + "   " +
        "  ".join(f"TRF {L:>2}s" for L in SD_LENGTHS))
    for arm in arms:
        sg = [O.sgat(arm, L).mean() for L in SD_LENGTHS]
        tr = [O.trf_acc(arm, L).mean() for L in SD_LENGTHS]
        res["sd_table"][arm] = {"sgat": sg, "trf": tr}
        out(f"  {arm:6s}  " + "  ".join(f"{a:9.4f}" for a in sg) + "   " + "  ".join(f"{a:8.4f}" for a in tr))

    out("\n=== C1 (primary): StimulusGAT st60 minus full ===")
    res["C1"] = _paired([(L, O.sgat(PRIMARY, L), O.sgat("full", L)) for L in SD_LENGTHS], out, "st60 vs full")
    res["C1"]["outcome"] = c1_outcome(res["C1"]["diffs"], res["C1"]["holm"])
    out(f"  outcome: {res['C1']['outcome']}")

    out("\n=== C2: StimulusGAT st60 minus TRF st60 ===")
    res["C2"] = _paired([(L, O.sgat(PRIMARY, L), O.trf_acc(PRIMARY, L)) for L in SD_LENGTHS], out,
                        "SGAT vs TRF, both st60")
    res["C2"]["outcome"] = h2_outcome(res["C2"]["diffs"], res["C2"]["holm"])
    out(f"  outcome: {res['C2']['outcome']}")
    full_c2 = _paired([(L, O.sgat("full", L), O.trf_acc("full", L)) for L in SD_LENGTHS], out,
                      "(descriptive) SGAT vs TRF, both full")
    res["C2_full"] = full_c2

    out("\n=== Controls (StimulusGAT st60 ensemble) ===")
    res["sd_controls_ok"] = True
    for kind in ("_shifted", "_swapped"):
        fa = []
        a5 = O.sgat(PRIMARY, 5, kind, fold_acc=fa)
        pk = [_wil(a5 - 0.5)] + [_wil(O.sgat(PRIMARY, L, kind) - 0.5) for L in SD_LENGTHS[1:]]
        ok = min(pk) >= 0.05 and max(fa) < 0.6
        res["sd_controls_ok"] &= ok
        out(f"  {kind[1:]:8s} 5 s {a5.mean():.3f}; subject p at 5/10/20/40 s: " + ", ".join(f"{p:.2g}" for p in pk)
            + f"; max fold @5 s {max(fa):.3f} -> {'clean' if ok else 'NOT clean'}")
    tsh = O.trf_acc(PRIMARY, 5, "shift_")
    out(f"  (descriptive) TRF st60 shift 5 s {tsh.mean():.3f}, p {_wil(tsh - 0.5):.2g}")
    res["sd_claim"] = bool(res["C1"]["outcome"] == "no loss" and res["C2"]["outcome"] == "better"
                           and res["sd_controls_ok"])
    out(f"\nCLAIM 'calibration EA keeps the story-disjoint result': {res['sd_claim']}")

    out("\n=== CA-TCN protocol: accuracy by arm (18 subjects, seed 42) ===")
    ct_arms = O.sg["config"]["ct_arms"]
    res["ct_table"], res["ct_verdicts"] = {}, {}
    out(f"  {'arm':6s}  " + "  ".join(f"{L:>2}s mean [CI] verdict".rjust(30) for L in CT_LENGTHS))
    for arm in ct_arms + ["noEA"]:
        cells, row, ver = [], [], []
        for L in CT_LENGTHS:
            a = O.ct_subject(arm, L, run="noea" if arm == "noEA" else "cal")
            ci = boot_ci(a)
            v = verdict(CATCN[L], ci)
            row.append(float(a.mean())); ver.append(v)
            cells.append(f"{a.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {v}".rjust(30))
        res["ct_table"][arm], res["ct_verdicts"][arm] = row, ver
        out(f"  {arm:6s}  " + "  ".join(cells))
    out("  CA-TCN: " + ", ".join(f"{L} s {v:.3f}" for L, v in CATCN.items()))

    good = ("level", "above")
    need = [L for L, v in zip(CT_LENGTHS, res["ct_verdicts"]["full"]) if v in good]
    keeps = all(res["ct_verdicts"][PRIMARY][CT_LENGTHS.index(L)] in good for L in need)
    out("\n=== L1 / controls (st60, 5 s) ===")
    res["ct_controls_ok"] = True
    for kind in ("shifted_logits", "swapped_logits"):
        a = O.ct_subject(PRIMARY, 5, kind)
        p = _wil(a - 0.5)
        ok = p >= 0.05 and a.max() < 0.6
        res["ct_controls_ok"] &= ok
        out(f"  {kind.split('_')[0]:8s} mean {a.mean():.3f}  p={p:.3g}  max subject {a.max():.3f} -> "
            f"{'clean' if ok else 'NOT clean'}")
    res["ct_claim"] = bool(keeps and res["ct_controls_ok"])
    out(f"  full is level/above at {need} s; st60 keeps it at all of them: {keeps}")
    out(f"\nCLAIM 'parity with CA-TCN without transductive alignment': {res['ct_claim']}")

    out("\n=== L2: CA-TCN protocol, st60 minus no-EA run ===")
    res["L2"] = _paired([(L, O.ct_subject(PRIMARY, L), O.ct_subject(None, L, run="noea")) for L in CT_LENGTHS],
                        out, "st60 vs no EA")
    res["L2"]["helps"] = any(d > 0 and q < 0.05 for d, q in zip(res["L2"]["diffs"], res["L2"]["holm"]))
    out(f"  calibration EA helps: {res['L2']['helps']}")
    res["ct_st60_vs_full"] = _paired([(L, O.ct_subject(PRIMARY, L), O.ct_subject("full", L)) for L in CT_LENGTHS],
                                     out, "(descriptive) st60 vs full")
    return res


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    analyse(*sys.argv[1:5])
