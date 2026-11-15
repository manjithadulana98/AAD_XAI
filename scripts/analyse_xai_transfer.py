"""Explanation transfer from DTU to KUL (exploratory; rules in the header of
notebooks/kaggle_run_kul_xai_transfer.py, written before any KUL explanation was computed).

usage: python -I scripts/analyse_xai_transfer.py <DTU xai_round9.json> <DTU dtu_variance.json> <KUL dir A> <KUL dir B>

Replication: the X1 rules (levels 1, 2, 4, H2, bands) applied unchanged to DTU (must reproduce
the X1 entry in RESULTS_LOG) and to KUL. Transfer: band-stop profiles, channel group maps (raw and
partial Spearman controlling for the log-variance maps of both datasets and the electrode
in-degree), edge group maps. Subject = unit throughout.
"""
from __future__ import annotations

import glob
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

CH_METHODS = ("occlusion", "ig", "attention")
EDGE_METHODS = ("attention", "attn_x_grad")
BANDS = ("delta 1-4", "theta 4-8", "alpha1 8-10", "alpha2 10-13", "beta1 13-16", "beta2 16-20",
         "beta3 20-26", "gamma1 26-32")
N_BOOT, N_SPLITS = 2000, 1000


# --- small helpers -------------------------------------------------------------------------
def area(curve) -> float:
    """Mean height of the accuracy curve [(k, acc, margin), ...] over its points (trapezoid)."""
    v = np.array([c[1] for c in curve], dtype=float)
    return float((v.sum() - 0.5 * v[0] - 0.5 * v[-1]) / (len(v) - 1))


def holm(ps) -> np.ndarray:
    ps = np.asarray(ps, dtype=float)
    out, run = np.empty_like(ps), 0.0
    for r, i in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def one_sided(a, b, alternative: str) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return 1.0 if not np.any(d) else float(wilcoxon(d, alternative=alternative).pvalue)


def rho(a, b) -> float:
    r = spearmanr(a, b).correlation
    return 0.0 if np.isnan(r) else float(r)


def by_subject(results) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in results:
        out[r["subject"]].append(r)
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))


# --- X1 rules ------------------------------------------------------------------------------
def channel_faithfulness(bs, mode: str = "permute") -> dict:
    """Per-subject deletion / keep areas for the 3 methods and the mean of the random rankings."""
    A = {(m, k): [] for m in CH_METHODS + ("random",) for k in ("delete", "keep")}
    for recs in bs.values():
        for kind in ("delete", "keep"):
            vals = defaultdict(list)
            for r in recs:
                for h in r["halves"]:
                    cur = h["channel_curves"][mode]
                    for m in CH_METHODS:
                        vals[m].append(area(cur[m][kind]))
                    vals["random"].append(np.mean([area(cur[k][kind]) for k in cur if k.startswith("random")]))
            for m, v in vals.items():
                A[(m, kind)].append(float(np.mean(v)))
    p_del = holm([one_sided(A[(m, "delete")], A[("random", "delete")], "less") for m in CH_METHODS])
    p_keep = holm([one_sided(A[(m, "keep")], A[("random", "keep")], "greater") for m in CH_METHODS])
    out = {"areas": {f"{m}_{k}": float(np.mean(v)) for (m, k), v in A.items()}}
    for m, hd, hk in zip(CH_METHODS, p_del, p_keep):
        out[m] = {"holm_delete": float(hd), "holm_keep": float(hk), "pass": bool(hd < 0.05 and hk < 0.05)}
    out["h2"] = {o: one_sided(A[("attention", "delete")], A[(o, "delete")], "greater") for o in ("occlusion", "ig")}
    out["h2_supported"] = bool(all(p < 0.05 for p in out["h2"].values()))
    return out


def edge_faithfulness(bs) -> dict:
    E = {m: [] for m in EDGE_METHODS + ("random",)}
    for recs in bs.values():
        vals = defaultdict(list)
        for r in recs:
            for h in r["halves"]:
                cur = h["edge_curves"]
                for m in EDGE_METHODS:
                    vals[m].append(area(cur[m]))
                vals["random"].append(np.mean([area(cur[k]) for k in cur if k.startswith("random")]))
        for m, v in vals.items():
            E[m].append(float(np.mean(v)))
    hp = holm([one_sided(E[m], E["random"], "less") for m in EDGE_METHODS])
    return {m: {"area": float(np.mean(E[m])), "random": float(np.mean(E["random"])), "holm": float(q),
                "pass": bool(q < 0.05)} for m, q in zip(EDGE_METHODS, hp)}


def band_profiles(bs) -> tuple[np.ndarray, np.ndarray]:
    """(n_subj, 8) band-power IG attribution and (n_subj, 8) band-stop margin drop, per subject."""
    att = np.array([np.mean([h["band"]["attribution"] for r in recs for h in r["halves"]], axis=0)
                    for recs in bs.values()])
    drp = np.array([np.mean([[d[2] for d in h["band"]["drops"]] for r in recs for h in r["halves"]], axis=0)
                    for recs in bs.values()])
    return att, drp


def band_rule(bs) -> dict:
    att, drp = band_profiles(bs)
    rhos = [rho(a, d) for a, d in zip(att, drp)]
    return {"median_rho": float(np.median(rhos)), "pass": bool(np.median(rhos) >= 0.7),
            "attribution": att.mean(0).tolist(), "drops": drp.mean(0).tolist()}


def sanity(bs) -> dict:
    out = {}
    for m in CH_METHODS + ("band",):
        full = [np.mean([list(r["sanity"][m].values())[-1] for r in recs]) for recs in bs.values()]
        out[m] = {"median_rho": float(np.median(full)), "pass": bool(abs(np.median(full)) < 0.3)}
    return out


def stability(bs) -> dict:
    out = {}
    for m in CH_METHODS:
        per_subj = []
        for recs in bs.values():
            rs = [rho(r["seed_scores"][a][m], r["seed_scores"][b][m])
                  for r in recs for a, b in itertools.combinations(sorted(r["seed_scores"]), 2)]
            per_subj.append(np.median(rs))
        out[m] = {"median_rho": float(np.median(per_subj)), "pass": bool(np.median(per_subj) >= 0.7)}
    return out


def x1_verdicts(results) -> dict[str, bool]:
    """The X1 verdicts (True = the rule's pass / supported outcome)."""
    bs = by_subject(results)
    ch, ed, sa, st = channel_faithfulness(bs), edge_faithfulness(bs), sanity(bs), stability(bs)
    v = {f"R1 sanity {m}": sa[m]["pass"] for m in CH_METHODS + ("band",)}
    v.update({f"R2 channel faithful {m}": ch[m]["pass"] for m in CH_METHODS})
    v.update({f"R3 edge faithful {m}": ed[m]["pass"] for m in EDGE_METHODS})
    v["R4 H2 attention least faithful"] = ch["h2_supported"]
    v["R5 band attribution matches band-stop"] = band_rule(bs)["pass"]
    v.update({f"R6 seed-stable {m}": st[m]["pass"] for m in CH_METHODS})
    return v


# --- maps and transfer ------------------------------------------------------------------------
def channel_subject_maps(bs, method: str) -> np.ndarray:
    """(n_subj, C): per subject, the mean over folds and seeds of the all-window channel scores."""
    return np.array([np.mean([r["seed_scores"][k][method] for r in recs for k in r["seed_scores"]], axis=0)
                     for recs in bs.values()])


def edge_subject_maps(bs, method: str) -> np.ndarray:
    """(n_subj, E): per subject, the mean over folds and halves of the edge scores (faith seed)."""
    return np.array([np.mean([h["edge_scores"][method] for r in recs for h in r["halves"]], axis=0)
                     for recs in bs.values()])


def variance_subject_maps(var_results) -> np.ndarray:
    """(n_subj, C): per subject, log of the mean over folds of the raw window variance."""
    bs = by_subject(var_results)
    return np.log(np.array([np.mean([r["variance"] for r in recs], axis=0) for recs in bs.values()]))


def group_map(M: np.ndarray) -> np.ndarray:
    """X1 recipe: z-score each subject's map, then average over subjects. (n_subj, K) -> (K,)"""
    Z = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-12)
    return Z.mean(axis=0)


def partial_spearman(a, b, covariates) -> float:
    """Spearman correlation of a and b after removing the linear effect of the ranked covariates."""
    ra, rb = rankdata(a), rankdata(b)
    X = np.column_stack([np.ones(len(ra))] + [rankdata(c) for c in covariates])
    res_a = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    res_b = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    if res_a.std() < 1e-12 or res_b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(res_a, res_b)[0, 1])


def in_degree(edges, n: int = 64) -> np.ndarray:
    """(n,) number of directed edges (i attends to j) that end at each node j."""
    deg = np.zeros(n)
    for _, j in edges:
        deg[j] += 1
    return deg


def split_half_reliability(M: np.ndarray, n_splits: int | None = None, seed: int = 0) -> float:
    """Mean Spearman between group maps of random subject halves, Spearman-Brown corrected."""
    n_splits = n_splits or N_SPLITS
    rng = np.random.default_rng(seed)
    n = len(M)
    rs = []
    for _ in range(n_splits):
        p = rng.permutation(n)
        rs.append(rho(group_map(M[p[:n // 2]]), group_map(M[p[n // 2:]])))
    r = float(np.mean(rs))
    return 2 * r / (1 + r) if r > -1 else r


def transfer(Md, Mk, Vd=None, Vk=None, deg=None, n_boot: int | None = None, seed: int = 0) -> dict:
    """DTU vs KUL group-map agreement with subject-bootstrap CIs (subjects resampled within each
    dataset). With Vd / Vk / deg, also the partial Spearman controlling for the two variance group
    maps and the in-degree. Md (n_dtu, K), Mk (n_kul, K); Vd (n_dtu, K), Vk (n_kul, K); deg (K,)."""
    has_cov = Vd is not None
    n_boot = n_boot or N_BOOT

    def stats(id_, ik):
        gd, gk = group_map(Md[id_]), group_map(Mk[ik])
        out = [rho(gd, gk)]
        if has_cov:
            out.append(partial_spearman(gd, gk, [group_map(Vd[id_]), group_map(Vk[ik]), deg]))
        return out

    est = stats(np.arange(len(Md)), np.arange(len(Mk)))
    rng = np.random.default_rng(seed)
    boot = np.array([stats(rng.integers(0, len(Md), len(Md)), rng.integers(0, len(Mk), len(Mk)))
                     for _ in range(n_boot)])
    rel_d, rel_k = split_half_reliability(Md, seed=seed), split_half_reliability(Mk, seed=seed + 1)
    out = {"rho": est[0], "rho_ci": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
           "rel_dtu": rel_d, "rel_kul": rel_k,
           "ceiling": float(np.sqrt(max(rel_d, 0) * max(rel_k, 0)))}
    if has_cov:
        out.update({"partial": est[1], "partial_ci": np.percentile(boot[:, 1], [2.5, 97.5]).tolist()})
        out["transfers"] = bool(out["partial_ci"][0] > 0)
    return out


def band_transfer(bs_d, bs_k) -> dict:
    """T1: each KUL subject's band-stop profile vs the DTU group profile."""
    _, drp_d = band_profiles(bs_d)
    _, drp_k = band_profiles(bs_k)
    ref = drp_d.mean(0)
    rhos = np.array([rho(p, ref) for p in drp_k])
    top2 = [BANDS[i] for i in np.argsort(-drp_k.mean(0))[:2]]
    return {"rhos": rhos.tolist(), "median": float(np.median(rhos)),
            "p": one_sided(rhos, np.zeros_like(rhos), "greater"),
            "group_rho": rho(ref, drp_k.mean(0)), "kul_top2": top2,
            "prediction_holds": set(top2) == {"theta 4-8", "alpha1 8-10"},
            "dtu_profile": ref.tolist(), "kul_profile": drp_k.mean(0).tolist()}


# --- loading -----------------------------------------------------------------------------------
def load_kul(dir_a, dir_b) -> tuple[list, list, list]:
    results, edges, commits = [], None, []
    for d, part in ((dir_a, "A"), (dir_b, "B")):
        hits = glob.glob(str(Path(d) / "**" / f"xai_kul_{part}.json"), recursive=True)
        assert len(hits) == 1, (d, hits)
        js = json.load(open(hits[0]))
        assert js["part"] == part
        edges = js["edges"] if edges is None else edges
        assert js["edges"] == edges, "edge lists differ between parts"
        results += js["results"]
        commits.append(js["commit"])
    return results, edges, commits


def check_kul(results) -> None:
    bs = by_subject(results)
    assert len(results) == 32 and len(bs) == 16, (len(results), len(bs))
    assert all(len(v) == 2 for v in bs.values()), "every KUL subject must appear in its 2 folds"
    assert len({(r["fold_id"], r["subject"]) for r in results}) == 32
    assert min(v["agree"] for r in results for v in r["reproduction"].values()) >= 0.99


def check_dtu_variance(xai, var) -> None:
    a = {(r["fold_id"], r["subject"]): r["n_windows"] for r in xai}
    b = {(r["fold_id"], r["subject"]): r["n_windows"] for r in var}
    assert a == b, "DTU variance windows differ from the X1 windows"


# --- report ------------------------------------------------------------------------------------
def analyse(dtu_xai_path, dtu_var_path, kul_a, kul_b, out=print) -> dict:
    dtu = json.load(open(dtu_xai_path))["results"]
    dvar = json.load(open(dtu_var_path))["results"]
    kul, edges, commits = load_kul(kul_a, kul_b)
    check_kul(kul)
    check_dtu_variance(dtu, dvar)
    bs_d, bs_k = by_subject(dtu), by_subject(kul)
    out(f"KUL commits {commits}; {len(kul)} KUL subject-folds ({len(bs_k)} subjects), {len(dtu)} DTU "
        f"subject-folds ({len(bs_d)} subjects); min KUL reproduction "
        f"{min(v['agree'] for r in kul for v in r['reproduction'].values()):.4f}; DTU variance windows match X1")
    res = {}

    out("\n=== Replication: the X1 rules on DTU (must match RESULTS_LOG) and on KUL ===")
    vd, vk = x1_verdicts(dtu), x1_verdicts(kul)
    res["replication"] = {k: {"dtu": vd[k], "kul": vk[k], "replicates": vd[k] == vk[k]} for k in vd}
    for k in vd:
        out(f"  {k:40s} DTU {'pass' if vd[k] else 'fail':4s}  KUL {'pass' if vk[k] else 'fail':4s}  "
            f"{'replicates' if vd[k] == vk[k] else 'DIFFERS'}")
    for name, bs in (("DTU", bs_d), ("KUL", bs_k)):
        ch, ed = channel_faithfulness(bs), edge_faithfulness(bs)
        sa, st, ba = sanity(bs), stability(bs), band_rule(bs)
        a = ch["areas"]
        out(f"  {name}: delete areas occlusion {a['occlusion_delete']:.3f} IG {a['ig_delete']:.3f} attention "
            f"{a['attention_delete']:.3f} random {a['random_delete']:.3f}; keep "
            f"{a['occlusion_keep']:.3f} / {a['ig_keep']:.3f} / {a['attention_keep']:.3f} / {a['random_keep']:.3f}; "
            f"H2 p {ch['h2']['occlusion']:.3g} / {ch['h2']['ig']:.3g}")
        out(f"       edges attention {ed['attention']['area']:.3f} attn_x_grad {ed['attn_x_grad']['area']:.3f} "
            f"random {ed['attention']['random']:.3f}; sanity rho " + ", ".join(
                f"{m} {sa[m]['median_rho']:+.3f}" for m in sa) + "; stability " + ", ".join(
                f"{m} {st[m]['median_rho']:.3f}" for m in st) + f"; band rule rho {ba['median_rho']:+.2f}")

    out("\n=== T1: band reliance by band-stop (KUL subject profiles vs the DTU group profile) ===")
    t1 = band_transfer(bs_d, bs_k)
    res["T1"] = t1
    out(f"  per-subject Spearman median {t1['median']:+.2f} (one-sided p = {t1['p']:.3g}); group profiles "
        f"{t1['group_rho']:+.2f}; KUL top-2 {t1['kul_top2']} -> prediction "
        f"{'holds' if t1['prediction_holds'] else 'fails'}")
    for b, d, k in zip(BANDS, t1["dtu_profile"], t1["kul_profile"]):
        out(f"    {b:13s} margin drop DTU {d:+.4f}  KUL {k:+.4f}")

    out("\n=== T2: channel group maps, DTU vs KUL ===")
    Vd, Vk, deg = variance_subject_maps(dvar), variance_subject_maps(kul), in_degree(edges)
    tv = transfer(Vd, Vk)
    res["T2_variance"] = tv
    out(f"  reference, log-variance maps: rho {tv['rho']:+.2f} {np.round(tv['rho_ci'], 2).tolist()}; "
        f"reliability DTU {tv['rel_dtu']:.2f} KUL {tv['rel_kul']:.2f}")
    res["T2"] = {}
    for m in CH_METHODS:
        Md, Mk = channel_subject_maps(bs_d, m), channel_subject_maps(bs_k, m)
        t = transfer(Md, Mk, Vd, Vk, deg)
        own = [rho(group_map(Md), group_map(Vd)), rho(group_map(Mk), group_map(Vk)),
               rho(group_map(Md), deg), rho(group_map(Mk), deg)]
        t["vs_own_variance"], t["vs_degree"] = own[:2], own[2:]
        res["T2"][m] = t
        out(f"  {m:10s} rho {t['rho']:+.2f} {np.round(t['rho_ci'], 2).tolist()}  partial {t['partial']:+.2f} "
            f"{np.round(t['partial_ci'], 2).tolist()} -> {'TRANSFERS' if t['transfers'] else 'not shown'}; ceiling "
            f"{t['ceiling']:.2f} (rel DTU {t['rel_dtu']:.2f}, KUL {t['rel_kul']:.2f}); map vs own variance "
            f"DTU {own[0]:+.2f} KUL {own[1]:+.2f}; vs in-degree DTU {own[2]:+.2f} KUL {own[3]:+.2f}")

    out("\n=== T3: edge group maps, DTU vs KUL (descriptive) ===")
    res["T3"] = {}
    for m in EDGE_METHODS:
        Md, Mk = edge_subject_maps(bs_d, m), edge_subject_maps(bs_k, m)
        assert Md.shape[1] == Mk.shape[1] == len(edges), (Md.shape, Mk.shape, len(edges))
        t = transfer(Md, Mk, n_boot=max(N_BOOT // 4, 1))
        res["T3"][m] = t
        out(f"  {m:12s} rho {t['rho']:+.2f} {np.round(t['rho_ci'], 2).tolist()}; ceiling {t['ceiling']:.2f}")

    out("\n=== Top-10 electrodes per method (descriptive; z-scored per subject, averaged) ===")
    names = json.load(open(glob.glob(str(Path(kul_a) / "**" / "xai_kul_A.json"), recursive=True)[0]))["config"]["montage"]
    for m in CH_METHODS:
        for name, bs in (("DTU", bs_d), ("KUL", bs_k)):
            g = group_map(channel_subject_maps(bs, m))
            out(f"  {m:10s} {name}: " + ", ".join(f"{names[i]} {g[i]:+.2f}" for i in np.argsort(-g)[:10]))
    for name, V in (("DTU", Vd), ("KUL", Vk)):
        g = group_map(V)
        out(f"  log-variance {name}: " + ", ".join(f"{names[i]} {g[i]:+.2f}" for i in np.argsort(-g)[:10]))
    return res


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    analyse(*sys.argv[1:5])
