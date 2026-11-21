"""Architecture-only reference for the DTU -> KUL explanation transfer (exploratory).

usage: python -I scripts/analyse_xai_random_reference.py <DTU xai_round9.json> <DTU dtu_variance.json>
       <KUL xai dir A> <KUL xai dir B> <random DTU dir> <random KUL dir>

Inputs: the X1 / explanation-transfer outputs (trained models) and the outputs of
notebooks/kaggle_run_xai_random_reference.py (fully randomised copies of the same models, same
windows). Written and tested on synthetic outputs before any random-model map existed; the rules
are not to be changed after the run.

Rules (subject = unit; group maps by the X1 recipe; bootstrap = 2000 resamples of subjects within
each dataset, the same resamples for the trained and the random maps):
- A1, channels (occlusion, IG, attention): D = partial Spearman of the trained DTU and KUL group
  maps minus the same for the random-model maps (controls: both datasets' log-variance maps and the
  electrode in-degree, as in T2). The trained agreement is "beyond the architecture" iff the 95%
  bootstrap CI of D lies above 0. No correction over the 3 methods (exploratory).
- A2, edges (attention, attention x gradient): the same with the raw Spearman over directed edges.
- Descriptive, no decision: random-model accuracy; split-half reliability of the random maps;
  per dataset, Spearman between the trained and the random group maps (how much of a trained map
  the architecture alone reproduces).
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location("analyse_xai_transfer",
                                               Path(__file__).resolve().parent / "analyse_xai_transfer.py")
xt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(xt)

CH_METHODS, EDGE_METHODS = xt.CH_METHODS, xt.EDGE_METHODS
N_BOOT = 2000


def load_random(d, dataset) -> dict:
    hits = glob.glob(str(Path(d) / "**" / f"xai_random_{dataset}.json"), recursive=True)
    assert len(hits) == 1, (d, hits)
    js = json.load(open(hits[0]))
    assert js["dataset"] == dataset
    return js


def check_match(trained, rand) -> None:
    """Same (fold, subject) records with the same window counts."""
    a = {(r["fold_id"], r["subject"]): r["n_windows"] for r in trained}
    b = {(r["fold_id"], r["subject"]): r["n_windows"] for r in rand}
    assert a == b, "random-model records differ from the trained-model records"


def random_channel_maps(bs, method) -> np.ndarray:
    """(n_subj, C): per subject, the mean over folds and random seeds of the all-window scores."""
    return np.array([np.mean([r["rand_scores"][k][method] for r in recs for k in r["rand_scores"]], axis=0)
                     for recs in bs.values()])


def random_edge_maps(bs, method) -> np.ndarray:
    """(n_subj, E): per subject, the mean over folds of the half-averaged edge scores."""
    return np.array([np.mean([r["rand_edge_scores"][method] for r in recs], axis=0) for recs in bs.values()])


def paired_transfer(Td, Tk, Rd, Rk, covs=None, n_boot=None, seed=0) -> dict:
    """Trained vs random cross-dataset agreement on the same bootstrap resamples.
    Td, Rd (n_dtu, K); Tk, Rk (n_kul, K); covs = (Vd, Vk, deg) for the partial Spearman, else raw."""
    n_boot = n_boot or N_BOOT

    def agree(Md, Mk, id_, ik):
        gd, gk = xt.group_map(Md[id_]), xt.group_map(Mk[ik])
        if covs is None:
            return xt.rho(gd, gk)
        Vd, Vk, deg = covs
        return xt.partial_spearman(gd, gk, [xt.group_map(Vd[id_]), xt.group_map(Vk[ik]), deg])

    def stats(id_, ik):
        t, r = agree(Td, Tk, id_, ik), agree(Rd, Rk, id_, ik)
        return t, r, t - r

    est = stats(np.arange(len(Td)), np.arange(len(Tk)))
    rng = np.random.default_rng(seed)
    boot = np.array([stats(rng.integers(0, len(Td), len(Td)), rng.integers(0, len(Tk), len(Tk)))
                     for _ in range(n_boot)])
    ci = lambda j: np.percentile(boot[:, j], [2.5, 97.5]).tolist()
    return {"trained": est[0], "trained_ci": ci(0), "random": est[1], "random_ci": ci(1),
            "diff": est[2], "diff_ci": ci(2), "beyond_architecture": bool(ci(2)[0] > 0)}


def analyse(dtu_xai_path, dtu_var_path, kul_a, kul_b, rand_dtu, rand_kul, out=print) -> dict:
    dtu = json.load(open(dtu_xai_path))["results"]
    dvar = json.load(open(dtu_var_path))["results"]
    kul, edges, _ = xt.load_kul(kul_a, kul_b)
    xt.check_kul(kul)
    xt.check_dtu_variance(dtu, dvar)
    rd, rk = load_random(rand_dtu, "DTU"), load_random(rand_kul, "KUL")
    check_match(dtu, rd["results"])
    check_match(kul, rk["results"])
    assert rd["edges"] == rk["edges"] == edges, "edge lists differ"
    bs_d, bs_k = xt.by_subject(dtu), xt.by_subject(kul)
    rbs_d, rbs_k = xt.by_subject(rd["results"]), xt.by_subject(rk["results"])
    assert list(bs_d) == list(rbs_d) and list(bs_k) == list(rbs_k)
    res = {}
    out(f"commits: random DTU {rd['commit']}, random KUL {rk['commit']}; {len(rd['results'])} DTU and "
        f"{len(rk['results'])} KUL subject-folds, windows identical to the trained-model runs")

    accs = {n: np.array([a for r in js["results"] for a in r["rand_acc"].values()]) for n, js in
            (("DTU", rd), ("KUL", rk))}
    res["rand_acc"] = {n: float(a.mean()) for n, a in accs.items()}
    out("random-model accuracy: " + "; ".join(f"{n} mean {a.mean():.3f} (range {a.min():.3f}-{a.max():.3f})"
                                              for n, a in accs.items()))

    Vd, Vk, deg = xt.variance_subject_maps(dvar), xt.variance_subject_maps(kul), xt.in_degree(edges)
    out("\n=== A1: channel maps, trained vs architecture-only agreement (partial Spearman) ===")
    res["A1"] = {}
    for m in CH_METHODS:
        Td, Tk = xt.channel_subject_maps(bs_d, m), xt.channel_subject_maps(bs_k, m)
        Rd, Rk = random_channel_maps(rbs_d, m), random_channel_maps(rbs_k, m)
        t = paired_transfer(Td, Tk, Rd, Rk, covs=(Vd, Vk, deg))
        t["trained_vs_random"] = {"DTU": xt.rho(xt.group_map(Td), xt.group_map(Rd)),
                                  "KUL": xt.rho(xt.group_map(Tk), xt.group_map(Rk))}
        t["random_reliability"] = {"DTU": xt.split_half_reliability(Rd), "KUL": xt.split_half_reliability(Rk, seed=1)}
        t["random_raw_rho"] = xt.rho(xt.group_map(Rd), xt.group_map(Rk))
        res["A1"][m] = t
        out(f"  {m:10s} trained {t['trained']:+.2f} {np.round(t['trained_ci'], 2).tolist()}  random "
            f"{t['random']:+.2f} {np.round(t['random_ci'], 2).tolist()}  diff {t['diff']:+.2f} "
            f"{np.round(t['diff_ci'], 2).tolist()} -> "
            f"{'BEYOND the architecture' if t['beyond_architecture'] else 'not beyond the architecture'}")
        out(f"             random raw rho {t['random_raw_rho']:+.2f}; trained vs random map DTU "
            f"{t['trained_vs_random']['DTU']:+.2f} KUL {t['trained_vs_random']['KUL']:+.2f}; random reliability "
            f"DTU {t['random_reliability']['DTU']:.2f} KUL {t['random_reliability']['KUL']:.2f}")

    out("\n=== A2: edge maps, trained vs architecture-only agreement (Spearman) ===")
    res["A2"] = {}
    n_boot_edges = max(N_BOOT // 4, 1)
    for m in EDGE_METHODS:
        Td, Tk = xt.edge_subject_maps(bs_d, m), xt.edge_subject_maps(bs_k, m)
        Rd, Rk = random_edge_maps(rbs_d, m), random_edge_maps(rbs_k, m)
        t = paired_transfer(Td, Tk, Rd, Rk, n_boot=n_boot_edges)
        t["trained_vs_random"] = {"DTU": xt.rho(xt.group_map(Td), xt.group_map(Rd)),
                                  "KUL": xt.rho(xt.group_map(Tk), xt.group_map(Rk))}
        res["A2"][m] = t
        out(f"  {m:12s} trained {t['trained']:+.2f} {np.round(t['trained_ci'], 2).tolist()}  random "
            f"{t['random']:+.2f} {np.round(t['random_ci'], 2).tolist()}  diff {t['diff']:+.2f} "
            f"{np.round(t['diff_ci'], 2).tolist()} -> "
            f"{'BEYOND the architecture' if t['beyond_architecture'] else 'not beyond the architecture'} "
            f"({n_boot_edges} resamples); trained vs random map DTU {t['trained_vs_random']['DTU']:+.2f} "
            f"KUL {t['trained_vs_random']['KUL']:+.2f}")
    return res


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    analyse(*sys.argv[1:7])
