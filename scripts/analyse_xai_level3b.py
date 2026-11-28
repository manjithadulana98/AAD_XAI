"""Level 3b: the planted-ground-truth benchmark at realistic accuracy.

usage: python scripts/analyse_xai_level3b.py <xai_level3b_A.json> <xai_level3b_B.json> <xai_level3.json>

Inputs: the outputs of notebooks/kaggle_run_xai_level3b.py, parts A (delta_T7 + null) and B
(theta_P8), and the level-3 output (ceiling run) for comparison. The rules are in the notebook
header; written and tested on synthetic outputs before any run, not to be changed after it.
- Gate: at -20 dB (the level-3 regime) the median sweep accuracy is >= 0.99 in each condition.
- Main SNR per condition: lowest sweep SNR with median accuracy >= 0.70 (recomputed here and
  checked against the notebook's choice).
- Verdicts (planted_eval.verdicts) on all seeds (primary) and on learned seeds only
  (validation >= 0.60; secondary). Method failures that disappear on learned seeds are
  labelled "model failure".
- Regime label per condition; sweep, null and level-3 comparison are descriptive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aad_xai.xai import planted_eval as pe  # noqa: E402

GATE_SNR, GATE_ACC = -20.0, 0.99


def load_parts(path_a, path_b):
    ja, jb = json.load(open(path_a)), json.load(open(path_b))
    assert ja["config"]["part"] == "A" and jb["config"]["part"] == "B", "parts swapped or missing"
    main_snr = {**(ja["main_snr_db"] or {}), **(jb["main_snr_db"] or {})}
    return ja["results"] + jb["results"], main_snr, (ja.get("commit"), jb.get("commit"))


def sweep_acc(results, cond):
    acc = {}
    for r in results:
        if r["condition"] == cond and r["stage"] == "sweep":
            acc.setdefault(float(r["snr_db"]), []).append(r["test_acc"])
    return dict(sorted(acc.items()))


def main_records(results, main_snr: dict) -> dict:
    return {c: [r for r in results if r["condition"] == c and s is not None and r["snr_db"] == s]
            for c, s in main_snr.items()}


def _fmt_verdicts(v, out):
    for c, x in v["validity"].items():
        out(f"    {c}: n={v['n'][c]} median acc {x['median_acc']:.3f}; V1 {x['V1']}, V2 {x['V2']}, "
            f"V3 {x['V3']} (p = {x['p3']:.3g})")
    for m, x in v["electrodes"].items():
        out(f"    electrodes {m:18s} " + "  ".join(f"{c} {a:.3f}" for c, a in x["median_auroc"].items())
            + f" -> {'PASS' if x['pass'] else 'fail'}; decoy best rank {v['decoy'][m]['best_rank']} -> "
            + ("PASS" if v["decoy"][m]["pass"] else "fail"))
    out("    band attribution first: " + "  ".join(f"{c} {s:.2f}" for c, s in v["bands"]["share_first"].items())
        + f" -> {'PASS' if v['bands']['pass'] else 'fail'}")
    if v["edges"] is None:
        out("    edges: V3 does not hold in every condition -> edge methods not scored")
    else:
        for m, x in v["edges"].items():
            out(f"    edges {m:12s} " + "  ".join(f"{c} {a:.3f}" for c, a in x["median_auroc"].items())
                + f" -> {'PASS' if x['pass'] else 'fail'}")


def _passes(v) -> dict:
    """Flat {test: bool} for comparing verdict sets."""
    d = {f"electrodes {m}": x["pass"] for m, x in v["electrodes"].items()}
    d.update({f"decoy {m}": x["pass"] for m, x in v["decoy"].items()})
    d["band attribution"] = v["bands"]["pass"]
    if v["edges"] is not None:
        d.update({f"edges {m}": x["pass"] for m, x in v["edges"].items()})
    return d


def analyse(path_a, path_b, path_l3, out=print) -> dict:
    results, main_snr, commits = load_parts(path_a, path_b)
    conds = [c for c in ("delta_T7", "theta_P8")]
    res = {"commits": commits}
    out(f"commits: A {commits[0]}; B {commits[1]}; runs {len(results)}")

    for c in conds:
        acc = sweep_acc(results, c)
        assert GATE_SNR in acc and np.median(acc[GATE_SNR]) >= GATE_ACC, (c, acc.get(GATE_SNR))
        chosen = pe.select_main_snr(acc)
        nb = main_snr.get(c)
        assert (chosen is None and nb is None) or (nb is not None and float(nb) == chosen), (c, chosen, nb)
        main_snr[c] = chosen
    out("gate: -20 dB reproduces the level-3 ceiling in both conditions; main SNR matches the notebook")

    out("\n=== Sweep (descriptive): median test accuracy, AUROC and decoy rank, seeds 0-2 ===")
    for c in conds:
        for snr, accs in sweep_acc(results, c).items():
            rs = [r for r in results if r["condition"] == c and r["stage"] == "sweep" and float(r["snr_db"]) == snr]
            out(f"  {c:9s} {snr:+6.1f} dB acc {np.median(accs):.3f} ({', '.join(f'{a:.2f}' for a in accs)})  " + "  ".join(
                f"{m} {np.median([r['metrics'][f'auroc_{m}'] for r in rs]):.2f}/{int(np.median([r['metrics'][f'decoy_rank_{m}'] for r in rs]))}"
                for m in pe.CH_METHODS))
    res["main_snr"] = main_snr
    out("  (each method: median AUROC / median decoy rank)")
    out("  main SNR: " + ", ".join(f"{c} {s} dB" for c, s in main_snr.items()))

    valid = {c: s for c, s in main_snr.items() if s is not None}
    if not valid:
        out("\nNo condition reaches 0.70: no pass/fail claim.")
        return res
    main = main_records(results, valid)
    out("\n=== Regime at the main SNR ===")
    res["regime"], res["learned"] = {}, {}
    for c, rs in main.items():
        accs = [r["test_acc"] for r in rs]
        learned = [pe.is_learned(r["val"]) for r in rs]
        res["regime"][c], res["learned"][c] = pe.regime(accs), int(sum(learned))
        out(f"  {c}: {len(rs)} seeds, test accuracy " + ", ".join(f"{a:.2f}" for a in sorted(accs))
            + f"; learned {sum(learned)}/{len(rs)}; regime {pe.regime(accs).upper()}")

    out("\n=== Verdicts, all seeds (PRIMARY; level-3 rules) ===")
    v_all = pe.verdicts(main)
    _fmt_verdicts(v_all, out)
    res["verdicts_all"] = v_all
    if len(valid) < 2:
        out("  NOTE: only one condition has a main SNR; level 3 required both, so these are partial.")

    out("\n=== Verdicts, learned seeds only (secondary) ===")
    main_l = {c: [r for r in rs if pe.is_learned(r["val"])] for c, rs in main.items()}
    main_l = {c: rs for c, rs in main_l.items() if rs}
    v_l = pe.verdicts(main_l) if main_l else None
    res["verdicts_learned"] = v_l
    if v_l is None:
        out("  no learned seeds")
    else:
        _fmt_verdicts(v_l, out)
        p_all, p_l = _passes(v_all), _passes(v_l)
        res["failure_kind"] = {k: pe.failure_kind(p_all[k], p_l.get(k, False)) for k in p_all}
        out("  labels: " + ", ".join(f"{k} {v}" for k, v in res["failure_kind"].items()))

    out("\n=== Compared with level 3 (ceiling, -20 dB; descriptive) ===")
    j3 = json.load(open(path_l3))
    s3 = j3["main_snr_db"]
    main3 = {c: [r for r in j3["results"] if r["condition"] == c and r["snr_db"] == s3] for c in conds}
    p3, pn = _passes(pe.verdicts(main3)), _passes(v_all)
    res["vs_level3"] = {k: {"level3": p3.get(k), "level3b": pn.get(k)} for k in sorted(set(p3) | set(pn))}
    for k, x in res["vs_level3"].items():
        tag = "" if x["level3"] == x["level3b"] else "   <- CHANGED"
        out(f"  {k:28s} level 3 {x['level3']!s:5s} level 3b {x['level3b']!s:5s}{tag}")

    nul = [r for r in results if r["condition"] == "null"]
    if nul:
        out(f"\nNull: accuracy {[round(r['test_acc'], 3) for r in nul]}; decoy ranks "
            + str({m: [r['metrics'][f'decoy_rank_{m}'] for r in nul] for m in pe.CH_METHODS}))
    return res


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    analyse(*sys.argv[1:4])
