"""DTU-trained models on KUL without KUL training (exploratory, preregistration section 8).

usage: python -I scripts/analyse_kul_zero_shot.py <zs_sgat_dir> <zs_trf_dir> <kul_sgat_A_dir> <kul_sgat_B_dir> <kul_trf_dir>

Inputs: downloaded outputs of notebooks/kaggle_run_kul_zero_shot_sgat.py and
kaggle_run_kul_zero_shot_trf.py, plus the KUL external test outputs (the KUL-trained
StimulusGAT parts A and B and the KUL-trained TRF + EA) for the paired comparisons.
Written and tested on synthetic outputs before any zero-shot result existed; the rules below
are not to be changed after the runs.

Units: the 16 KUL subjects; accuracy over decision windows of 5/10/20/40 s (StimulusGAT: sign
of the summed 5 s logit differences within a trial; TRF: correlation recomputed over the
window). Zero-shot StimulusGAT = logits of all 90 DTU round-9 models averaged.

- Z1 (primary): zero-shot StimulusGAT above 0.5, one-sided Wilcoxon, Holm over the 4 lengths.
- Controls (zero-shot ensemble, shifted and swapped envelopes): two-sided Wilcoxon vs 0.5
  p >= 0.05 at every length, and no KUL fold >= 0.6 at 5 s.
- Claim "DTU-trained StimulusGAT decodes KUL" iff Z1 holds at 5 s (Holm p < 0.05) and both
  controls are clean.
- Z2, transfer gap: KUL-trained 5-seed ensemble minus zero-shot, two-sided Wilcoxon, Holm
  over 4. Also reported: share of the above-chance accuracy kept, (zs - 0.5) / (kul - 0.5).
- Z3, zero-shot StimulusGAT vs zero-shot TRF (18 DTU decoders, reconstructions averaged):
  two-sided Wilcoxon, Holm over 4, classified with the KUL external test's H2 rule.
- Z4, does the network lose more than the linear model? Per subject, (KUL-trained minus
  zero-shot) for StimulusGAT minus the same for the TRF; two-sided Wilcoxon, Holm over 4.
- Descriptive, no decision: zero-shot TRF vs 0.5 and its shift control; spread of the 18
  per-DTU-fold 5-seed ensembles and the single-model mean; accuracy by condition (dry /
  HRTF), attended track (reader) and attended ear; right-ear and track-1 pick rates.
"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aad_xai.evaluation.decision_windows import (group_windows, recomputed_corr_decisions,  # noqa: E402
                                                 summed_score_decisions)

_SPEC = importlib.util.spec_from_file_location("analyse_kul_external",
                                               Path(__file__).resolve().parent / "analyse_kul_external.py")
ak = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ak)

LENGTHS = ak.LENGTHS
WINDOW_S, SFREQ = ak.WINDOW_S, ak.SFREQ
N_MODELS = 90
N_DTU_DECODERS = 18
_wil, holm, h2_outcome = ak._wil, ak.holm, ak.h2_outcome


class ZeroShot:
    """Zero-shot outputs, checked against the KUL external test (same windows, same swaps)."""

    def __init__(self, zs_sgat, zs_trf, R):
        self.R = R
        self._gcache = {}
        self.sg_dir, self.tr_dir = Path(zs_sgat), Path(zs_trf)
        self.sg_json = R._json(self.sg_dir, "kul_zero_shot_sgat.json")
        self.tr_json = R._json(self.tr_dir, "kul_zero_shot_trf.json")
        cfg = self.sg_json["config"]
        self.by_fold = {f: [f"dtu_s{s}_{f}" for s in cfg["dtu_seeds"]] for f in cfg["dtu_folds"]}
        self.members = [m for ms in self.by_fold.values() for m in ms]
        assert len(self.members) == N_MODELS, len(self.members)
        assert len(self.tr_json["decoders"]) == N_DTU_DECODERS, len(self.tr_json["decoders"])
        assert self.sg_json["trial_info"] == self.tr_json["trial_info"] == R.info, "trial info differs"
        assert sorted(f["fold_id"] for f in self.sg_json["folds"]) == sorted(R.fids)
        assert sorted(f["fold_id"] for f in self.tr_json["folds"]) == sorted(R.fids)
        for fid in R.fids:
            z, k, t, kt = self.npz(fid), R.npz("A", fid), self.trf(fid), R.trf(fid)
            for key in ("test_subject", "test_trial", "test_start", "test_label"):
                assert np.array_equal(z[key], k[key]), ("zero-shot SGAT windows differ", fid, key)
                assert np.array_equal(t[key], kt[key]), ("zero-shot TRF windows differ", fid, key)
            assert np.array_equal(z["test_swap_perm"], k["test_swap_perm"]), ("swap differs", fid)
            for m in self.members:
                for kind in ("", "_shifted", "_swapped"):
                    assert f"{m}_test{kind}_logits" in z, (fid, m, kind)
            assert t["test_decoder_scores"].shape == (len(t["test_label"]), N_DTU_DECODERS), fid

    def npz(self, fid):
        return self.R._load(self.sg_dir, fid)

    def trf(self, fid):
        return self.R._load(self.tr_dir, f"trf_zs_{fid}")

    def score(self, members, fid, kind=""):
        """Mean over members of the logit difference (B - A), one per 5 s window."""
        z = self.npz(fid)
        return np.mean([np.diff(z[f"{m}_test{kind}_logits"], axis=1)[:, 0] for m in members], axis=0)

    def _groups(self, fid, L, where):
        """Decision groups of fold ``fid`` (the zero-shot SGAT and TRF windows are identical)."""
        z = self.npz(fid)
        if (fid, L) not in self._gcache:
            self._gcache[(fid, L)] = group_windows(z["test_subject"], z["test_trial"], z["test_start"],
                                                   z["test_label"], L // WINDOW_S)
        groups = self._gcache[(fid, L)]
        if where is not None:
            groups = [g for g in groups if where(self.R.info[str(z["test_trial"][g[2][0]])])]
        return groups

    def _tally(self, per_fold, fold_acc=None):
        counts = defaultdict(lambda: [0, 0])
        for groups, dec in per_fold:
            c_f = 0
            for (s, y, _), d in zip(groups, dec):
                counts[str(s)][0] += int(d == y); counts[str(s)][1] += 1
                c_f += int(d == y)
            if fold_acc is not None:
                fold_acc.append(c_f / max(len(groups), 1))
        # NaN for a subject with no window in this selection (only possible with ``where``).
        return np.array([counts[s][0] / counts[s][1] if counts[s][1] else np.nan for s in self.R.subjects])

    def sgat(self, members=None, L=5, kind="", fold_acc=None, where=None):
        """Per-subject accuracy (S1..S16) of the logit-averaged DTU models."""
        members = members or self.members
        per_fold = []
        for fid in self.R.fids:
            groups = self._groups(fid, L, where)
            per_fold.append((groups, summed_score_decisions(self.score(members, fid, kind), groups)))
        return self._tally(per_fold, fold_acc)

    def trf_acc(self, L=5, kind=""):
        """Per-subject accuracy of the DTU TRF ensemble (recomputed-correlation rule)."""
        per_fold = []
        for fid in self.R.fids:
            t = self.trf(fid)
            groups = self._groups(fid, L, None)
            per_fold.append((groups, recomputed_corr_decisions(t[f"test_{kind}stats"], WINDOW_S * SFREQ, groups)))
        return self._tally(per_fold)

    def trf_decoder_acc(self, j):
        """Per-subject 5 s accuracy of DTU decoder j alone."""
        per_fold = []
        for fid in self.R.fids:
            t = self.trf(fid)
            groups = self._groups(fid, 5, None)
            per_fold.append((groups, summed_score_decisions(t["test_decoder_scores"][:, j], groups)))
        return self._tally(per_fold)

    def pick_rate(self, side_of):
        """Per-subject share of 5 s windows where the zero-shot ensemble picks the candidate for
        which ``side_of(info, picked_right) -> bool`` holds."""
        counts = defaultdict(lambda: [0, 0])
        for fid in self.R.fids:
            z = self.npz(fid)
            right = self.score(self.members, fid) > 0
            for s, tid, r in zip(z["test_subject"], z["test_trial"], right):
                counts[str(s)][0] += int(side_of(self.R.info[str(tid)], bool(r)))
                counts[str(s)][1] += 1
        return np.array([counts[s][0] / counts[s][1] for s in self.R.subjects])


def _paired(out, name, rows, res_key, res, classify=False):
    """rows: [(L, a, b)] per-subject arrays; two-sided Wilcoxon of a - b, Holm over the lengths."""
    ps = [_wil(a - b) for _, a, b in rows]
    hp = holm(ps)
    diffs = [float(np.mean(a - b)) for _, a, b in rows]
    for (L, a, b), d, p, q in zip(rows, diffs, ps, hp):
        out(f"  {L:>2} s  {a.mean():.4f} vs {b.mean():.4f}  diff {d:+.4f}  higher {int((a > b).sum())}/16 "
            f"lower {int((a < b).sum())}/16  p={p:.3g}  Holm {q:.3g}")
    res[res_key] = {"diffs": diffs, "p": ps, "holm": hp}
    if classify:
        res[res_key]["outcome"] = h2_outcome(diffs, hp)
        out(f"  outcome: {res[res_key]['outcome']}")


def analyse(zs_sgat, zs_trf, dir_a, dir_b, dir_trf, out=print):
    R = ak.Runs(dir_a, dir_b, dir_trf)
    Z = ZeroShot(zs_sgat, zs_trf, R)
    res = {}
    out(f"commits: zero-shot SGAT {Z.sg_json['commit']}; zero-shot TRF {Z.tr_json['commit']}; "
        f"KUL A {R.json['A']['commit']}")
    out(f"{len(R.fids)} KUL folds, {len(R.subjects)} subjects, {len(Z.members)} DTU models, "
        f"{N_DTU_DECODERS} DTU decoders; windows and swaps identical to the KUL external test")

    zs = {L: Z.sgat(L=L) for L in LENGTHS}
    kul = {L: R.sgat(L=L) for L in LENGTHS}
    out("\n=== Z1: zero-shot StimulusGAT (90 DTU models) above chance, Holm over 4 lengths ===")
    ps = [_wil(zs[L] - 0.5, "greater") for L in LENGTHS]
    hp = holm(ps)
    for L, p, q in zip(LENGTHS, ps, hp):
        out(f"  {L:>2} s  {zs[L].mean():.4f} (min {zs[L].min():.3f}, max {zs[L].max():.3f}); "
            f"subjects > 0.5: {int((zs[L] > 0.5).sum())}/16; one-sided p={p:.3g}  Holm {q:.3g}")
    res["z1"] = {"acc": [float(zs[L].mean()) for L in LENGTHS], "p": ps, "holm": hp}
    res["z1_5s"] = hp[0] < 0.05

    out("\n=== Controls (zero-shot ensemble) ===")
    res["controls_ok"] = True
    for kind in ("_shifted", "_swapped"):
        fa = []
        a5 = Z.sgat(kind=kind, fold_acc=fa)
        pk = [_wil(a5 - 0.5)] + [_wil(Z.sgat(L=L, kind=kind) - 0.5) for L in LENGTHS[1:]]
        ok = min(pk) >= 0.05 and max(fa) < 0.6
        res["controls_ok"] &= ok
        res[f"control{kind}"] = {"acc5": float(a5.mean()), "p": pk, "max_fold": float(max(fa)), "ok": ok}
        out(f"  {kind[1:]:8s} 5 s {a5.mean():.3f}; subject p at 5/10/20/40 s: "
            + ", ".join(f"{p:.2g}" for p in pk) + f"; max fold @5 s {max(fa):.3f} -> {'clean' if ok else 'NOT clean'}")
    res["claim"] = bool(res["z1_5s"] and res["controls_ok"])
    out(f"\nCLAIM 'DTU-trained StimulusGAT decodes KUL' (Z1 at 5 s and controls clean): {res['claim']}")

    out("\n=== Z2: transfer gap, KUL-trained ensemble vs zero-shot ===")
    _paired(out, "gap", [(L, kul[L], zs[L]) for L in LENGTHS], "z2", res)
    kept = [float((zs[L].mean() - 0.5) / (kul[L].mean() - 0.5)) if kul[L].mean() > 0.5 else float("nan")
            for L in LENGTHS]
    res["z2"]["kept"] = kept
    out("  share of above-chance accuracy kept: " + ", ".join(f"{L} s {k:.2f}" for L, k in zip(LENGTHS, kept)))

    zt = {L: Z.trf_acc(L=L) for L in LENGTHS}
    kt = {L: R.trf_acc(L=L) for L in LENGTHS}
    out("\n=== Z3: zero-shot StimulusGAT vs zero-shot TRF + EA ===")
    _paired(out, "zs", [(L, zs[L], zt[L]) for L in LENGTHS], "z3", res, classify=True)

    out("\n=== Z4: loss from moving dataset, StimulusGAT minus TRF ===")
    out("  (each value: KUL-trained minus zero-shot; positive diff = the network loses more)")
    _paired(out, "drop", [(L, kul[L] - zs[L], kt[L] - zt[L]) for L in LENGTHS], "z4", res)

    out("\n=== Descriptive (no decision attached) ===")
    tsh = Z.trf_acc(kind="shift_")
    out("  zero-shot TRF: " + ", ".join(f"{L} s {zt[L].mean():.4f} (p vs 0.5 {_wil(zt[L] - 0.5):.2g})"
                                         for L in LENGTHS) + f"; shift control 5 s {tsh.mean():.3f} "
        f"(p {_wil(tsh - 0.5):.2g})")
    dec = [Z.trf_decoder_acc(j).mean() for j in range(N_DTU_DECODERS)]
    out(f"  single DTU decoders at 5 s: mean {np.mean(dec):.4f} (range {min(dec):.3f}-{max(dec):.3f})")
    per_fold = {f: Z.sgat(ms).mean() for f, ms in Z.by_fold.items()}
    out(f"  18 per-DTU-fold 5-seed ensembles at 5 s: mean {np.mean(list(per_fold.values())):.4f} "
        f"(range {min(per_fold.values()):.3f}-{max(per_fold.values()):.3f})")
    singles = [Z.sgat([m]).mean() for m in Z.members]
    out(f"  single DTU models at 5 s: mean {np.mean(singles):.4f} (range {min(singles):.3f}-{max(singles):.3f})")
    res["per_dtu_fold"] = per_fold
    res["single_mean"] = float(np.mean(singles))
    for name, key, values in (("condition", "condition", ("dry", "hrtf")),
                              ("attended track", "attended_track", (1, 2)),
                              ("attended ear", "attended_ear", ("L", "R"))):
        a, b = (Z.sgat(where=lambda i, k=key, v=v: i[k] == v) for v in values)
        ok = np.isfinite(a) & np.isfinite(b)
        out(f"  {name}, 5 s: {values[0]} {np.nanmean(a):.4f}  {values[1]} {np.nanmean(b):.4f}  "
            f"(paired p = {_wil(a[ok] - b[ok]):.3g}, {int(ok.sum())} subjects with both)")
    right = Z.pick_rate(lambda i, r: r)
    t1 = Z.pick_rate(lambda i, r: (i["right_track"] if r else i["left_track"]) == 1)
    out(f"  right-ear pick rate {right.mean():.3f} (p vs 0.5 = {_wil(right - 0.5):.3g}); "
        f"track-1 pick rate {t1.mean():.3f} (p = {_wil(t1 - 0.5):.3g})")
    out("  per subject 5 s (zero-shot SGAT / KUL-trained SGAT / zero-shot TRF): " + ", ".join(
        f"{s} {a:.3f}/{b:.3f}/{c:.3f}" for s, a, b, c in zip(R.subjects, zs[5], kul[5], zt[5])))
    return res


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    analyse(*sys.argv[1:6])
