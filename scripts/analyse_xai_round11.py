"""X1 after eye removal: are the frontal-polar electrodes still first in the round-11 models?

usage: python -I scripts/analyse_xai_round11.py <round-9 xai_round9.json> <round-11 xai_round11.json>

Inputs: the X1 output of notebooks/kaggle_run_dtu_xai_round9.py (round 9) and the output of
notebooks/kaggle_run_dtu_xai_round11.py (round 11, which also holds the round-9 group deletion).
The rules are in the header of the round-11 notebook; written and tested on synthetic outputs
before any round-11 explanation existed, not to be changed after the run.

Subject = unit (n = 18); each subject's 3 folds pooled.
- Gate: every round-11 and round-9 model reproduces its saved test decisions (>= 0.99) and both
  files hold the same 54 subject-folds with the same window counts.
- E1 (primary): frontal-polar occlusion index (mean z of the 8 Fp/AF electrodes in the subject's
  seed-averaged occlusion map) lower in round 11 than in round 9; one-sided paired Wilcoxon.
- E2: round-11 index above 0; one-sided Wilcoxon. Outcome: "removed" (E1, not E2), "reduced"
  (E1 and E2), "not reduced" (not E1).
- E3: frontal-polar reliance = mean accuracy over the random 8-electrode deletions minus the
  accuracy with the 8 Fp/AF electrodes deleted (= the extra drop caused by Fp/AF), per subject
  over folds and seeds. E3a: round 11 lower than round 9; E3b: round 11 above 0. One-sided
  Wilcoxon, p < 0.05 each. Margin version descriptive.
- Descriptive: X1 rules on both rounds, Fp/AF index for IG and attention, top-10 electrodes,
  occlusion group map vs the log-variance group map of the model's own input.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location("analyse_xai_transfer",
                                               Path(__file__).resolve().parent / "analyse_xai_transfer.py")
ax = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ax)

ALPHA = 0.05
CH_METHODS = ax.CH_METHODS


def fp_index(M: np.ndarray, fp_idx) -> np.ndarray:
    """(n_subj, C) maps -> (n_subj,) mean z-score of the Fp/AF electrodes in each subject's map."""
    Z = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-12)
    return Z[:, list(fp_idx)].mean(axis=1)


def e1_outcome(e1: bool, e2: bool) -> str:
    if not e1:
        return "not reduced"
    return "reduced" if e2 else "removed"


def reliance(bs, key: str, which: int = 0) -> np.ndarray:
    """(n_subj,) mean over folds and seeds of mean(random-set metric) - Fp/AF-set metric.
    which: 0 = accuracy, 1 = margin. key: "group_deletion" (round 11) or "group_deletion_r9"."""
    out = []
    for recs in bs.values():
        v = [np.mean([r_[which] for r_ in g["random"]]) - g["fp"][which]
             for r in recs for g in r[key].values()]
        out.append(float(np.mean(v)))
    return np.array(out)


def check(r9, r11) -> None:
    key = lambda res: {(r["fold_id"], r["subject"]): r["n_windows"] for r in res}
    assert len(r11) == 54 and len(ax.by_subject(r11)) == 18, (len(r11), len(ax.by_subject(r11)))
    assert key(r9) == key(r11), "round-9 X1 and round-11 files hold different subject-folds or windows"
    for r in r11:
        for k in ("reproduction", "reproduction_r9"):
            assert min(v["agree"] for v in r[k].values()) >= 0.99, (r["fold_id"], r["subject"], k)


def analyse(r9_path, r11_path, out=print) -> dict:
    j9, j11 = json.load(open(r9_path)), json.load(open(r11_path))
    r9, r11 = j9["results"], j11["results"]
    check(r9, r11)
    names, fp_names = j11["config"]["montage"], j11["config"]["fp_names"]
    fp = [names.index(n) for n in fp_names]
    bs9, bs11 = ax.by_subject(r9), ax.by_subject(r11)
    assert list(bs9) == list(bs11)
    rep = min(v["agree"] for r in r11 for k in ("reproduction", "reproduction_r9") for v in r[k].values())
    out(f"commits: round 9 X1 {j9.get('commit')}; round 11 {j11.get('commit')}")
    out(f"gate: {len(r11)} subject-folds, {len(bs11)} subjects, windows match X1; min reproduction {rep:.4f}")
    res = {}

    out("\n=== E1 / E2 (primary): frontal-polar occlusion index (mean z of Fp/AF, per subject) ===")
    i9 = fp_index(ax.channel_subject_maps(bs9, "occlusion"), fp)
    i11 = fp_index(ax.channel_subject_maps(bs11, "occlusion"), fp)
    p1 = ax.one_sided(i11, i9, "less")
    p2 = ax.one_sided(i11, np.zeros_like(i11), "greater")
    e1, e2 = p1 < ALPHA, p2 < ALPHA
    res["E1"] = {"round9": i9.tolist(), "round11": i11.tolist(), "p": p1, "holds": bool(e1)}
    res["E2"] = {"p": p2, "holds": bool(e2)}
    res["E_outcome"] = e1_outcome(e1, e2)
    out(f"  round 9 {i9.mean():+.2f} (median {np.median(i9):+.2f}); round 11 {i11.mean():+.2f} "
        f"(median {np.median(i11):+.2f}); lower in {int((i11 < i9).sum())}/{len(i9)}")
    out(f"  E1 round 11 < round 9: p = {p1:.3g} -> {'holds' if e1 else 'not shown'}")
    out(f"  E2 round 11 > 0:       p = {p2:.3g} -> {'holds' if e2 else 'not shown'}")
    out(f"  OUTCOME: frontal-polar occlusion ranking {res['E_outcome'].upper()}")

    out("\n=== E3: frontal-polar reliance by group deletion (extra accuracy drop from deleting Fp/AF) ===")
    a9, a11 = reliance(bs11, "group_deletion_r9"), reliance(bs11, "group_deletion")
    m9, m11 = reliance(bs11, "group_deletion_r9", 1), reliance(bs11, "group_deletion", 1)
    p3a = ax.one_sided(a11, a9, "less")
    p3b = ax.one_sided(a11, np.zeros_like(a11), "greater")
    res["E3"] = {"round9": a9.tolist(), "round11": a11.tolist(), "p_a": p3a, "p_b": p3b,
                 "E3a": bool(p3a < ALPHA), "E3b": bool(p3b < ALPHA),
                 "margin_round9": m9.tolist(), "margin_round11": m11.tolist()}
    out(f"  accuracy points: round 9 {100 * a9.mean():+.2f}, round 11 {100 * a11.mean():+.2f}; "
        f"lower in {int((a11 < a9).sum())}/{len(a9)}")
    out(f"  E3a round 11 < round 9: p = {p3a:.3g} -> {'holds' if p3a < ALPHA else 'not shown'}")
    out(f"  E3b round 11 > 0:       p = {p3b:.3g} -> {'holds' if p3b < ALPHA else 'not shown'}")
    out(f"  (descriptive) margin: round 9 {m9.mean():+.3f}, round 11 {m11.mean():+.3f}")
    for tag, key in (("round 9", "group_deletion_r9"), ("round 11", "group_deletion")):
        g = [gd for r in r11 for gd in r[key].values()]
        out(f"  (descriptive) {tag}: accuracy intact {np.mean([x['intact'][0] for x in g]):.3f}, Fp/AF deleted "
            f"{np.mean([x['fp'][0] for x in g]):.3f}, 8 random deleted "
            f"{np.mean([np.mean([q[0] for q in x['random']]) for x in g]):.3f}")

    out("\n=== Descriptive: Fp/AF index for the other channel methods ===")
    res["fp_index"] = {}
    for m in CH_METHODS:
        j9_, j11_ = (fp_index(ax.channel_subject_maps(b, m), fp) for b in (bs9, bs11))
        res["fp_index"][m] = {"round9": float(j9_.mean()), "round11": float(j11_.mean())}
        out(f"  {m:10s} round 9 {j9_.mean():+.2f}  round 11 {j11_.mean():+.2f}")

    out("\n=== Descriptive: X1 rules, round 9 vs round 11 ===")
    v9, v11 = ax.x1_verdicts(r9), ax.x1_verdicts(r11)
    res["x1"] = {k: {"round9": v9[k], "round11": v11[k]} for k in v9}
    for k in v9:
        out(f"  {k:40s} round 9 {'pass' if v9[k] else 'fail':4s}  round 11 {'pass' if v11[k] else 'fail':4s}")
    for tag, bs in (("round 9", bs9), ("round 11", bs11)):
        ch, ed, sa, st, ba = (ax.channel_faithfulness(bs), ax.edge_faithfulness(bs), ax.sanity(bs),
                              ax.stability(bs), ax.band_rule(bs))
        a = ch["areas"]
        out(f"  {tag}: delete areas occlusion {a['occlusion_delete']:.3f} IG {a['ig_delete']:.3f} attention "
            f"{a['attention_delete']:.3f} random {a['random_delete']:.3f}; edges {ed['attention']['area']:.3f} / "
            f"{ed['attn_x_grad']['area']:.3f} vs {ed['attention']['random']:.3f}; sanity " + ", ".join(
                f"{m} {sa[m]['median_rho']:+.2f}" for m in sa) + "; stability " + ", ".join(
                f"{m} {st[m]['median_rho']:.2f}" for m in st) + f"; band rho {ba['median_rho']:+.2f}")
        out("    band-stop margin drop: " + ", ".join(
            f"{b.split()[0]} {d:+.3f}" for b, d in zip(ax.BANDS, ba["drops"])))

    out("\n=== Descriptive: top-10 electrodes (z-scored per subject, averaged) and variance ===")
    V9 = np.log(np.array([np.mean([r["variance_raw"] for r in recs], axis=0) for recs in bs11.values()]))
    V11 = np.log(np.array([np.mean([r["variance"] for r in recs], axis=0) for recs in bs11.values()]))
    res["vs_variance"] = {}
    for m in CH_METHODS:
        for tag, bs, V in (("round 9", bs9, V9), ("round 11", bs11, V11)):
            g = ax.group_map(ax.channel_subject_maps(bs, m))
            rv = ax.rho(g, ax.group_map(V))
            res["vs_variance"][f"{m} {tag}"] = rv
            out(f"  {m:10s} {tag:8s}: " + ", ".join(f"{names[i]} {g[i]:+.2f}" for i in np.argsort(-g)[:10])
                + f"   (vs own log-variance rho {rv:+.2f})")
    for tag, V in (("round 9 (uncleaned)", V9), ("round 11 (cleaned)", V11)):
        g = ax.group_map(V)
        out(f"  log-variance {tag}: " + ", ".join(f"{names[i]} {g[i]:+.2f}" for i in np.argsort(-g)[:10]))
    return res


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    analyse(*sys.argv[1:3])
