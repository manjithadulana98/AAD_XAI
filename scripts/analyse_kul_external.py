"""Pre-registered analysis of the KUL external held-out test (docs/preregistration_kul_external_test.md).

usage: python -I scripts/analyse_kul_external.py <sgat_part_A_dir> <sgat_part_B_dir> <trf_dir>

Inputs are the downloaded outputs of the Kaggle notebooks kaggle_run_kul_external_sgat.py
(parts A and B) and kaggle_run_kul_external_trf.py. Written and tested on synthetic outputs
before any KUL result existed; the decision rules are not to be changed after the runs.

Confirmatory (section 6):
- H1: 5-seed ensemble, 5 s, per-subject accuracy > 0.5 (one-sided Wilcoxon, p < 0.05).
- H2: ensemble minus TRF + EA at 5/10/20/40 s, two-sided per-subject Wilcoxon, Holm over 4.
  Outcome, in this order: "mixed" if Holm-significant in both directions; "better" if
  positive with Holm p < 0.05 at >= 1 length and the mean difference is >= -0.5 points at
  every length; "worse" if negative with Holm p < 0.05 at >= 1 length; else "no difference".
- Controls (ensemble, shifted and swapped envelopes): subject Wilcoxon vs 0.5 p >= 0.05 at
  5/10/20/40 s, and no fold >= 0.6 at 5 s.
- Claim "StimulusGAT generalises to KUL" iff H1, H2 == "better", and both controls pass.
Decision rules over L s: StimulusGAT, sign of the summed 5 s logit differences of
consecutive windows within a trial; TRF, correlation recomputed over the whole window
(its standard rule; the summed-score variant is printed as a descriptive check).
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aad_xai.evaluation.decision_windows import (group_windows, recomputed_corr_decisions,  # noqa: E402
                                                 summed_score_decisions)

SEEDS = {"A": [42, 1, 2], "B": [3, 4]}
LENGTHS = [5, 10, 20, 40]
WINDOW_S, SFREQ = 5, 64
META = ("test_subject", "test_trial", "test_start", "test_label", "val_label", "test_swap_perm",
        "test1s_subject", "test1s_trial", "test1s_label", "test2s_subject", "test2s_trial", "test2s_label")


def _wil(d, alternative="two-sided"):
    d = np.asarray(d, dtype=float)
    return float(wilcoxon(d, alternative=alternative).pvalue) if np.any(d != 0) else 1.0


def holm(ps):
    order, out, run = np.argsort(ps), [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def h2_outcome(mean_diffs, holm_ps, alpha=0.05, max_drop=0.005):
    """Pre-registered classification of the ensemble-minus-TRF comparison."""
    pos = any(d > 0 and p < alpha for d, p in zip(mean_diffs, holm_ps))
    neg = any(d < 0 and p < alpha for d, p in zip(mean_diffs, holm_ps))
    if pos and neg:
        return "mixed"
    if pos and all(d >= -max_drop for d in mean_diffs):
        return "better"
    if neg:
        return "worse"
    return "no difference"


class Runs:
    """Loaded outputs, with integrity checks across parts A, B and the TRF."""

    def __init__(self, dir_a, dir_b, dir_trf):
        self.dirs = {"A": Path(dir_a), "B": Path(dir_b)}
        self.trf_dir = Path(dir_trf)
        self.json = {p: self._json(d, f"kul_external_sgat_{p}.json") for p, d in self.dirs.items()}
        self.trf_json = self._json(self.trf_dir, "kul_external_trf.json")
        self.fids = [f["fold_id"] for f in self.json["A"]["folds"]]
        assert len(self.fids) == 16, len(self.fids)
        assert [f["fold_id"] for f in self.json["B"]["folds"]] == self.fids, "parts A and B have different folds"
        assert sorted(self.trf_json["folds"]) == sorted(self.fids), "TRF has different folds"
        assert self.json["A"]["trial_info"] == self.json["B"]["trial_info"] == self.trf_json["trial_info"]
        self.info = self.json["A"]["trial_info"]
        self._cache = {}
        self.members = [(p, f"kul_s{s}") for p in ("A", "B") for s in SEEDS[p]]
        for fid in self.fids:
            a, b, t = self.npz("A", fid), self.npz("B", fid), self.trf(fid)
            for k in META:
                assert np.array_equal(a[k], b[k]), (fid, k)
            for k in ("test_subject", "test_trial", "test_start", "test_label"):
                assert np.array_equal(a[k], t[k]), ("trf", fid, k)
            for p, m in self.members:
                assert f"{m}_test_logits" in self.npz(p, fid), (p, fid, m)
            perm = a["test_swap_perm"]
            parts = np.array([self.info[str(x)]["story_part"] for x in a["test_trial"]])
            subj = a["test_subject"]
            assert np.all(parts[perm] != parts) and np.all(subj[perm] == subj), (fid, "swap pairs")
        subs = {str(s) for fid in self.fids for s in self.npz("A", fid)["test_subject"]}
        self.subjects = sorted(subs, key=lambda s: int(s[1:]))
        assert len(self.subjects) == 16, self.subjects

    @staticmethod
    def _json(d, name):
        hits = glob.glob(str(d / "**" / name), recursive=True)
        assert len(hits) == 1, (d, name, hits)
        return json.load(open(hits[0]))

    def _load(self, d, name):
        if (d, name) not in self._cache:
            hits = glob.glob(str(d / "**" / f"{name}.npz"), recursive=True)
            assert len(hits) == 1, (d, name, len(hits))
            with np.load(hits[0], allow_pickle=False) as z:
                self._cache[(d, name)] = {k: z[k] for k in z.files}
        return self._cache[(d, name)]

    def npz(self, part, fid):
        return self._load(self.dirs[part], fid)

    def trf(self, fid):
        return self._load(self.trf_dir, f"trf_ea_{fid}")

    def score(self, members, fid, grid="test", kind=""):
        """Mean over members of the logit difference (B - A), one per window of ``grid``."""
        return np.mean([np.diff(self.npz(p, fid)[f"{m}_{grid}{kind}_logits"], axis=1)[:, 0]
                        for p, m in members], axis=0)

    def _acc(self, counts):
        return np.array([counts[s][0] / counts[s][1] for s in self.subjects])

    def sgat(self, members=None, L=5, grid="test", kind="", fold_acc=None, where=None):
        """Per-subject accuracy (S1..S16) of the logit-averaged members, decisions over L s.
        ``where(info) -> bool`` restricts to trials whose info matches."""
        members = members or self.members
        n_per = L // WINDOW_S if grid == "test" else 1     # 1 s / 2 s grids: one decision per window
        counts = defaultdict(lambda: [0, 0])
        for fid in self.fids:
            z = self.npz("A", fid)
            groups = group_windows(z[f"{grid}_subject"], z[f"{grid}_trial"], z[f"{grid}_start"],
                                   z[f"{grid}_label"], n_per)
            if where is not None:
                groups = [gr for gr in groups if where(self.info[str(z[f"{grid}_trial"][gr[2][0]])])]
            dec = summed_score_decisions(self.score(members, fid, grid, kind), groups)
            c_f = n_f = 0
            for (s, y, _), d in zip(groups, dec):
                counts[str(s)][0] += int(d == y); counts[str(s)][1] += 1
                c_f += int(d == y); n_f += 1
            if fold_acc is not None:
                fold_acc.append(c_f / n_f)
        return self._acc(counts)

    def trf_acc(self, L=5, kind="", rule="recomputed"):
        counts = defaultdict(lambda: [0, 0])
        for fid in self.fids:
            t = self.trf(fid)
            groups = group_windows(t["test_subject"], t["test_trial"], t["test_start"], t["test_label"],
                                   L // WINDOW_S)
            if rule == "recomputed":
                dec = recomputed_corr_decisions(t[f"test_{kind}stats"], WINDOW_S * SFREQ, groups)
            else:
                dec = summed_score_decisions(t[f"test_{kind}score"], groups)
            for (s, y, _), d in zip(groups, dec):
                counts[str(s)][0] += int(d == y); counts[str(s)][1] += 1
        return self._acc(counts)

    def pick_rate(self, side_of):
        """Per-subject share of 5 s test windows where the ensemble picks the candidate for
        which ``side_of(info, picked_right) -> bool`` holds."""
        counts = defaultdict(lambda: [0, 0])
        for fid in self.fids:
            z = self.npz("A", fid)
            right = self.score(self.members, fid) > 0
            for s, tid, r in zip(z["test_subject"], z["test_trial"], right):
                counts[str(s)][0] += int(side_of(self.info[str(tid)], bool(r)))
                counts[str(s)][1] += 1
        return self._acc(counts)


def analyse(dir_a, dir_b, dir_trf, out=print):
    R = Runs(dir_a, dir_b, dir_trf)
    res = {}
    out(f"commits: A {R.json['A']['commit']}; B {R.json['B']['commit']}; TRF {R.trf_json['commit']}")
    out(f"{len(R.fids)} folds, {len(R.subjects)} subjects; test windows identical across A, B and the TRF; "
        "swap pairs same subject, other part")

    out("\n=== H1: 5-seed ensemble above chance at 5 s ===")
    e5 = R.sgat()
    res["h1_p"] = _wil(e5 - 0.5, "greater")
    res["h1"] = res["h1_p"] < 0.05
    out(f"  ensemble 5 s {e5.mean():.4f} (min {e5.min():.3f}, max {e5.max():.3f}); "
        f"subjects > 0.5: {int((e5 > 0.5).sum())}/16; one-sided p = {res['h1_p']:.3g} -> H1 {res['h1']}")

    out("\n=== H2: ensemble vs TRF + EA (Holm over 4 lengths) ===")
    rows = [(L, R.sgat(L=L), R.trf_acc(L=L)) for L in LENGTHS]
    ps = [_wil(a - b) for _, a, b in rows]
    hp = holm(ps)
    diffs = [float(np.mean(a - b)) for _, a, b in rows]
    for (L, a, b), d, p, q in zip(rows, diffs, ps, hp):
        out(f"  {L:>2} s  SGAT {a.mean():.4f}  TRF {b.mean():.4f}  diff {d:+.4f}  "
            f"SGAT higher {int((a > b).sum())}/16 lower {int((a < b).sum())}/16  p={p:.3g}  Holm {q:.3g}")
    res["h2"] = h2_outcome(diffs, hp)
    res["h2_diffs"], res["h2_holm"] = diffs, hp
    out(f"  H2 outcome (pre-registered): {res['h2']}")

    out("\n=== Controls (ensemble) ===")
    res["controls_ok"] = True
    for kind in ("_shifted", "_swapped"):
        fa = []
        a5 = R.sgat(kind=kind, fold_acc=fa)
        pk = [_wil(a5 - 0.5)] + [_wil(R.sgat(L=L, kind=kind) - 0.5) for L in LENGTHS[1:]]
        ok = min(pk) >= 0.05 and max(fa) < 0.6
        res["controls_ok"] &= ok
        res[f"control{kind}"] = {"acc5": float(a5.mean()), "p": pk, "max_fold": float(max(fa)), "ok": ok}
        out(f"  {kind[1:]:8s} 5 s {a5.mean():.3f}; subject p at 5/10/20/40 s: "
            + ", ".join(f"{p:.2g}" for p in pk) + f"; max fold @5 s {max(fa):.3f} -> {'clean' if ok else 'NOT clean'}")
    out(f"  controls: {'clean' if res['controls_ok'] else 'NOT clean'}")

    res["claim"] = bool(res["h1"] and res["h2"] == "better" and res["controls_ok"])
    out(f"\nCLAIM 'StimulusGAT generalises to KUL' (H1 and H2 better and controls clean): {res['claim']}")

    out("\n=== Descriptive (pre-specified, no decision attached) ===")
    t1 = R.pick_rate(lambda i, right: (i["right_track"] if right else i["left_track"]) == 1)
    left = R.pick_rate(lambda i, right: not right)
    out(f"  narrator bias: track-1 pick rate {t1.mean():.3f} (p vs 0.5 = {_wil(t1 - 0.5):.3g})")
    out(f"  side bias: left pick rate {left.mean():.3f} (p vs 0.5 = {_wil(left - 0.5):.3g})")
    dry = R.sgat(where=lambda i: i["condition"] == "dry")
    hrtf = R.sgat(where=lambda i: i["condition"] == "hrtf")
    out(f"  condition, 5 s: dry {dry.mean():.4f}  hrtf {hrtf.mean():.4f}  (paired p = {_wil(dry - hrtf):.3g})")
    singles = {L: np.mean([R.sgat([m], L=L).mean() for m in R.members]) for L in (5, 10)}
    for w in (1, 2):
        singles[w] = np.mean([R.sgat([m], grid=f"test{w}s").mean() for m in R.members])
    ens_short = {w: R.sgat(grid=f"test{w}s").mean() for w in (1, 2)}
    out("  single-model mean: " + ", ".join(f"{L} s {singles[L]:.4f}" for L in (1, 2, 5, 10))
        + f"; ensemble 1 s {ens_short[1]:.4f}, 2 s {ens_short[2]:.4f}")
    out("  seeds at 5 s: " + ", ".join(f"{m} {R.sgat([(p, m)]).mean():.4f}" for p, m in R.members))
    out("  TRF summed-score rule: " + ", ".join(f"{L} s {R.trf_acc(L=L, rule='summed').mean():.4f}"
                                                 for L in LENGTHS[1:]))
    tsh = R.trf_acc(kind="shift_")
    out(f"  TRF shift control 5 s {tsh.mean():.3f} (p vs 0.5 = {_wil(tsh - 0.5):.3g})")
    val = {p: np.mean([m["val_balanced"] for f in R.json[p]["folds"] for m in f["models"].values()]) for p in "AB"}
    ep = [m["best_epoch"] for p in "AB" for f in R.json[p]["folds"] for m in f["models"].values()]
    out(f"  mean class-balanced val: A {val['A']:.4f}, B {val['B']:.4f}; best epoch median {np.median(ep):.0f} "
        f"(range {min(ep)}-{max(ep)}); TRF mean val {np.mean([f['val_acc'] for f in R.trf_json['folds'].values()]):.4f}")
    t5 = R.trf_acc()
    out("  per subject 5 s (SGAT ens / TRF): " + ", ".join(
        f"{s} {a:.3f}/{b:.3f}" for s, a, b in zip(R.subjects, e5, t5)))
    return res


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    analyse(*sys.argv[1:4])
