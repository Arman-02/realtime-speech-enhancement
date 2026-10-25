"""Statistics (PLAN.md §4.6, §5.1.6, §8): paired Wilcoxon signed-rank tests with Holm correction, and
MOS aggregation of listening-test responses with rater screening.

uv run python -m ans.stats mos responses/*.json [--index ../web/public/listening/index.json] [--results R]
uv run python -m ans.stats wilcoxon --csv exports/per_utterance.csv --set vbd --metric pesq \
    --a "CRN-128 (KD)" --b CRN-128

A claim "A is better than B" goes into the README only if the Holm-corrected p < 0.05.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------
# Paired tests
# ------------------------------------------------------------------------------------------------


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided paired Wilcoxon signed-rank test of ``a - b`` (pairs with a NaN are dropped).

    Matches ``scipy.stats.wilcoxon(a, b)`` (``zero_method="wilcox"``, default method)."""
    from scipy.stats import wilcoxon as _w

    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = x - y
    out = {
        "n": len(d),
        "median_diff": float(np.median(d)) if len(d) else math.nan,
        "mean_diff": float(np.mean(d)) if len(d) else math.nan,
        "statistic": math.nan,
        "p_wilcoxon": math.nan,
    }
    if len(d) < 2 or not np.any(d != 0):
        out["p_wilcoxon"] = 1.0 if len(d) else math.nan
        return out
    res = _w(x, y)
    out["statistic"], out["p_wilcoxon"] = float(res.statistic), float(res.pvalue)
    return out


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values (NaNs are left as NaN and not counted)."""
    p = np.asarray(pvals, np.float64)
    idx = [i for i in np.argsort(p) if np.isfinite(p[i])]
    m = len(idx)
    adj = np.full(len(p), math.nan)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def compare_pairs(
    scores: dict[str, dict[str, float]], pairs: list[tuple[str, str]], metric: str, set_name: str, family: str
) -> list[dict]:
    """``scores[model][utt_id] -> value``; paired tests for ``pairs`` with Holm correction within the family."""
    out = []
    for a, b in pairs:
        if a not in scores or b not in scores:
            continue
        common = sorted(set(scores[a]) & set(scores[b]))
        if len(common) < 2:
            continue
        r = wilcoxon([scores[a][u] for u in common], [scores[b][u] for u in common])
        out.append({"a": a, "b": b, "metric": metric, "set": set_name, "family": family, **r})
    for r, p in zip(out, holm([r["p_wilcoxon"] for r in out]), strict=True):
        r["p_holm"] = p
    return out


# ------------------------------------------------------------------------------------------------
# MOS (listening test, §3.6 / §5.1.6)
# ------------------------------------------------------------------------------------------------


def _mean_ci(v: Sequence[float]) -> dict:
    x = np.asarray(v, np.float64)
    if len(x) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    ci = 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0
    return {"mean": round(float(x.mean()), 3), "ci95": round(float(ci), 3), "n": len(x)}


def mos(
    responses: list[dict],
    reference: str = "clean",
    anchor: str = "noisy",
    max_inversions: float = 0.3,
) -> dict:
    """MOS ± 95 % CI per condition. Raters who score the hidden reference below the low anchor on more than
    ``max_inversions`` of their trials are excluded. Practice trials are never part of an export."""
    raters, excluded = [], []
    per_cond: dict[str, list[float]] = defaultdict(list)
    per_rater_cond: dict[str, dict[str, list[float]]] = {}
    for resp in responses:
        rid = str(resp.get("rater_id", f"rater{len(raters) + len(excluded)}"))
        by_trial: dict[str, dict[str, float]] = defaultdict(dict)
        for r in resp.get("ratings", []):
            if str(r.get("trial", "")).startswith("practice"):
                continue
            by_trial[str(r["trial"])][str(r["condition"])] = float(r["score"])
        judged = [t for t in by_trial.values() if reference in t and anchor in t]
        inversions = sum(t[reference] < t[anchor] for t in judged)
        frac = inversions / len(judged) if judged else 1.0
        info = {"rater_id": rid, "trials": len(by_trial), "inversion_rate": round(frac, 3)}
        if frac > max_inversions or not by_trial:
            excluded.append(info)
            continue
        raters.append(info)
        per_rater_cond[rid] = defaultdict(list)
        for t in by_trial.values():
            for cond, score in t.items():
                per_cond[cond].append(score)
                per_rater_cond[rid][cond].append(score)
    conditions = {c: _mean_ci(v) for c, v in sorted(per_cond.items())}
    for c in conditions:  # rater-level CI (means of per-rater means) is the conservative one
        rater_means = [float(np.mean(d[c])) for d in per_rater_cond.values() if d.get(c)]
        conditions[c]["rater_level"] = _mean_ci(rater_means)
    return {
        "conditions": conditions,
        "raters": len(raters),
        "excluded": excluded,
        "screening": f"excluded if {reference} < {anchor} on > {max_inversions:.0%} of trials",
    }


def mos_pairwise(responses: list[dict], pairs: list[tuple[str, str]]) -> list[dict]:
    """Paired Wilcoxon on per-(rater, trial) scores between conditions (Holm within the given pairs)."""
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for resp in responses:
        rid = str(resp.get("rater_id"))
        for r in resp.get("ratings", []):
            if not str(r.get("trial", "")).startswith("practice"):
                scores[str(r["condition"])][f"{rid}/{r['trial']}"] = float(r["score"])
    return compare_pairs(scores, pairs, "mos", "listening", "mos")


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------


def _load_scores(csv_path: Path, set_name: str, metric: str) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("set", "vbd") == set_name and row.get(metric) not in (None, ""):
                scores[row["model"]][row["id"]] = float(row[metric])
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mos", help="aggregate listening-test response JSON files")
    m.add_argument("files", nargs="+")
    m.add_argument("--max-inversions", type=float, default=0.3)
    m.add_argument("--results", help="also write the table into this results.json under 'mos'")
    w = sub.add_parser("wilcoxon", help="paired test between two models from a per-utterance CSV")
    w.add_argument("--csv", required=True)
    w.add_argument("--set", default="vbd")
    w.add_argument("--metric", default="pesq")
    w.add_argument("--a", required=True)
    w.add_argument("--b", required=True)
    args = ap.parse_args()

    if args.cmd == "mos":
        responses = [json.loads(Path(f).read_text()) for f in args.files]
        table = mos(responses, max_inversions=args.max_inversions)
        conds = list(table["conditions"])
        anchors = [c for c in conds if c not in ("clean", "noisy")]
        table["significance"] = mos_pairwise(
            responses, [(a, b) for i, a in enumerate(anchors) for b in anchors[i + 1 :]]
        )
        print(f"raters kept: {table['raters']}, excluded: {len(table['excluded'])}")
        for c, v in table["conditions"].items():
            print(f"{c:<10s} MOS {v['mean']:.2f} ± {v['ci95']:.2f} (n={v['n']})")
        if args.results:
            p = Path(args.results)
            res = json.loads(p.read_text())
            res["mos"] = table
            p.write_text(json.dumps(res, indent=2) + "\n")
    else:
        scores = _load_scores(Path(args.csv), args.set, args.metric)
        r = compare_pairs(scores, [(args.a, args.b)], args.metric, args.set, "cli")
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
