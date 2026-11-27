"""Is the frontal-polar reliance specific? Cluster-matched group deletion.

usage: python scripts/analyse_cluster_deletion.py <cluster_deletion.json> <xai_round11.json>

Inputs: the output of notebooks/kaggle_run_dtu_cluster_deletion.py and the output of
notebooks/kaggle_run_dtu_xai_round11.py (for the reproduction gate). The rules are in the header
of the cluster-deletion notebook; written and tested on synthetic outputs before any cluster
deletion existed, not to be changed after the run.

Subject = unit (n = 18); each subject's 3 folds and 5 seeds pooled.
- Gate: every model reproduces its saved test decisions (>= 0.99); the FP and SCATTER accuracies
  reproduce the round-11 X1 run (mean absolute difference <= 0.005), for both rounds.
- Reliance of FP over a family = mean accuracy with a family set deleted - accuracy with FP
  deleted. Families: CLUSTERS (compact clusters sharing no electrode with FP) and BANDS (the
  rotated FP bands sharing no electrode with FP).
- C1 (primary, round 11): reliance over CLUSTERS > 0. C2 (round 11): reliance over BANDS > 0.
  One-sided Wilcoxon, p < 0.05 each. "specific" if both, "not specific" if neither, else "mixed".
- C3: the same on round 9. C4 (round 11): SCATTER mean accuracy - CLUSTERS mean accuracy > 0.
- Descriptive: margins, share of CLUSTERS that cost less than FP, cluster-deletion map, bands.
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
GATE_REPRO = 0.99
GATE_E3 = 0.005
ROUNDS = (("round 11", "deletion"), ("round 9", "deletion_r9"))


def families(config) -> dict[str, list[int]]:
    """Indices into a record's "clusters" list and the band angles of the comparison families."""
    fp = {config["montage"].index(n) for n in config["fp_names"]}
    return {"clusters": [i for i, s in enumerate(config["clusters"]) if not fp & set(s)],
            "bands": [a for a, s in config["bands"].items() if a != "0" and not fp & set(s)]}


def family_value(d: dict, fam: str, members, which: int = 0) -> float:
    """Mean metric (0 accuracy, 1 margin) over the family's sets in one deletion record."""
    if fam == "clusters":
        return float(np.mean([d["clusters"][i][which] for i in members]))
    if fam == "bands":
        return float(np.mean([d["bands"][a][which] for a in members]))
    if fam == "scatter":
        return float(np.mean([q[which] for q in d["scatter"]]))
    raise ValueError(fam)


def per_subject(bs, key: str, fn) -> np.ndarray:
    """(n_subj,) mean over folds and seeds of fn(deletion record)."""
    return np.array([np.mean([fn(d) for r in recs for d in r[key].values()]) for recs in bs.values()])


def reliance(bs, key: str, fam: str, members, which: int = 0) -> np.ndarray:
    """(n_subj,) family-mean metric minus the metric with FP deleted (= extra drop from FP)."""
    return per_subject(bs, key, lambda d: family_value(d, fam, members, which) - d["fp"][which])


def outcome(c1: bool, c2: bool) -> str:
    if c1 and c2:
        return "specific"
    return "not specific" if not (c1 or c2) else "mixed"


def check(res, r11) -> float:
    """Gate. Returns the mean absolute FP/SCATTER accuracy difference to the round-11 X1 run."""
    assert len(res) == 54 and len(ax.by_subject(res)) == 18, (len(res), len(ax.by_subject(res)))
    old = {(r["fold_id"], r["subject"]): r for r in r11}
    assert set(old) == {(r["fold_id"], r["subject"]) for r in res}, "different subject-folds"
    diffs = []
    for r in res:
        o = old[(r["fold_id"], r["subject"])]
        assert r["n_windows"] == o["n_windows"], (r["fold_id"], r["subject"])
        for k in ("reproduction", "reproduction_r9"):
            assert min(v["agree"] for v in r[k].values()) >= GATE_REPRO, (r["fold_id"], r["subject"], k)
        for new_key, old_key in (("deletion", "group_deletion"), ("deletion_r9", "group_deletion_r9")):
            for s, d in r[new_key].items():
                g = o[old_key][s]
                diffs.append(abs(d["fp"][0] - g["fp"][0]))
                diffs += [abs(a[0] - b[0]) for a, b in zip(d["scatter"], g["random"], strict=True)]
    m = float(np.mean(diffs))
    assert m <= GATE_E3, f"FP / SCATTER accuracies do not reproduce the round-11 X1 run ({m:.4f})"
    return m


def test_pair(bs, key, fam_idx) -> dict:
    a = reliance(bs, key, "clusters", fam_idx["clusters"])
    b = reliance(bs, key, "bands", fam_idx["bands"])
    z = np.zeros_like(a)
    p1, p2 = ax.one_sided(a, z, "greater"), ax.one_sided(b, z, "greater")
    return {"clusters": a.tolist(), "bands": b.tolist(), "p_clusters": p1, "p_bands": p2,
            "holds_clusters": bool(p1 < ALPHA), "holds_bands": bool(p2 < ALPHA),
            "outcome": outcome(p1 < ALPHA, p2 < ALPHA)}


def analyse(cd_path, r11_path, out=print) -> dict:
    jc, j11 = json.load(open(cd_path)), json.load(open(r11_path))
    cfg, res = jc["config"], jc["results"]
    diff = check(res, j11["results"])
    bs = ax.by_subject(res)
    names = cfg["montage"]
    fam = families(cfg)
    rep = min(v["agree"] for r in res for k in ("reproduction", "reproduction_r9") for v in r[k].values())
    out(f"commit {jc.get('commit')}; gate: {len(res)} subject-folds, {len(bs)} subjects; min reproduction "
        f"{rep:.4f}; FP/SCATTER vs round-11 X1 mean |diff| {diff:.4f}")
    out(f"families: {len(fam['clusters'])} compact clusters, bands at "
        + ", ".join(f"{a} deg ({'/'.join(names[i] for i in cfg['bands'][a][:3])}...)" for a in fam["bands"]))
    result = {"gate_diff": diff}

    for tag, key in ROUNDS:
        t = test_pair(bs, key, fam)
        result[tag] = t
        a, b = np.array(t["clusters"]), np.array(t["bands"])
        label = "C1 / C2 (primary)" if key == "deletion" else "C3 (secondary)"
        out(f"\n=== {label}, {tag}: extra accuracy drop from deleting FP over a matched family ===")
        out(f"  over compact clusters: {100 * a.mean():+.2f} points (median {100 * np.median(a):+.2f}), "
            f"> 0 in {int((a > 0).sum())}/{len(a)}; p = {t['p_clusters']:.3g} -> "
            f"{'holds' if t['holds_clusters'] else 'not shown'}")
        out(f"  over rotated bands:    {100 * b.mean():+.2f} points (median {100 * np.median(b):+.2f}), "
            f"> 0 in {int((b > 0).sum())}/{len(b)}; p = {t['p_bands']:.3g} -> "
            f"{'holds' if t['holds_bands'] else 'not shown'}")
        out(f"  OUTCOME: the frontal-polar reliance is {t['outcome'].upper()}")
        sc = reliance(bs, key, "scatter", None)
        out(f"  (reference) over scattered sets: {100 * sc.mean():+.2f} points")
        ma = reliance(bs, key, "clusters", fam["clusters"], 1)
        mb = reliance(bs, key, "bands", fam["bands"], 1)
        out(f"  (descriptive) margin: over clusters {ma.mean():+.3f}, over bands {mb.mean():+.3f}")

    out("\n=== C4, round 11: do contiguous deletions cost more than scattered ones? ===")
    c4 = per_subject(bs, "deletion", lambda d: family_value(d, "scatter", None)
                     - family_value(d, "clusters", fam["clusters"]))
    p4 = ax.one_sided(c4, np.zeros_like(c4), "greater")
    result["C4"] = {"values": c4.tolist(), "p": p4, "holds": bool(p4 < ALPHA)}
    out(f"  scattered minus clustered accuracy: {100 * c4.mean():+.2f} points, > 0 in {int((c4 > 0).sum())}/"
        f"{len(c4)}; p = {p4:.3g} -> {'holds' if p4 < ALPHA else 'not shown'}")

    for tag, key in ROUNDS:
        out(f"\n=== Descriptive, {tag} ===")
        acc = lambda f: per_subject(bs, key, f)
        intact, fpd = acc(lambda d: d["intact"][0]), acc(lambda d: d["fp"][0])
        out(f"  accuracy intact {intact.mean():.3f}; deleted: FP {fpd.mean():.3f}, scattered "
            f"{acc(lambda d: family_value(d, 'scatter', None)).mean():.3f}, clusters "
            f"{acc(lambda d: family_value(d, 'clusters', fam['clusters'])).mean():.3f}, bands "
            f"{acc(lambda d: family_value(d, 'bands', fam['bands'])).mean():.3f}")
        share = acc(lambda d: np.mean([d["clusters"][i][0] > d["fp"][0] for i in fam["clusters"]]))
        out(f"  share of compact clusters that cost less than FP: median {np.median(share):.2f} "
            f"(range {share.min():.2f}-{share.max():.2f})")
        out("  band drops: " + ", ".join(
            f"{a} deg {100 * (intact - acc(lambda d, a=a: d['bands'][a][0])).mean():.1f}"
            for a in sorted(cfg["bands"], key=int) if a != "0") + f"; FP {100 * (intact - fpd).mean():.1f} points")
        cl_index = {tuple(s): i for i, s in enumerate(cfg["clusters"])}
        centre = [cl_index[tuple(s)] for s in cfg["centre_clusters"]]
        drops = np.array([(intact - acc(lambda d, i=i: d["clusters"][i][0])).mean() for i in centre])   # (64,)
        result.setdefault("cluster_map", {})[tag] = drops.tolist()
        order = np.argsort(-drops)
        out("  cluster-deletion map, top 10 centres (accuracy points): "
            + ", ".join(f"{names[i]} {100 * drops[i]:.1f}" for i in order[:10]))
        fp_centres = [names.index(n) for n in cfg["fp_names"]]
        out(f"  FP deletion {100 * (intact - fpd).mean():.1f} points; it costs more than "
            f"{int((drops < (intact - fpd).mean()).sum())}/64 centred clusters "
            f"(clusters centred on Fp/AF electrodes: best {100 * drops[fp_centres].max():.1f})")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    analyse(*sys.argv[1:3])
