"""Does the external bench's min_possession_s drive any decision? Rank methods at 0/2/5/8 s.

    python scripts/min_possession_sensitivity.py                      # val matches, seeds 0-4
    python scripts/min_possession_sensitivity.py --seeds 0 1 2 --values 0 5

External labels (DFL, Metrica) change possession on almost every touch; harness.bench_external
maps them onto the challenge's control rule with external.align_labels(labels, min_possession_s),
merging H-A-H possessions shorter than min_possession_s (0 = no merging). 5 s was chosen on the
public window's possession rate, which may no longer tune or select anything. It cannot be
re-derived without that window, but it can be tested for whether it MATTERS: if the ranking of
methods does not move across reasonable values, the choice decides nothing.

How. Only the `val` external matches (never `test`) present in the cache. min_possession_s
touches the ground truth only -- degrade() is steered by the source's raw labels -- so each
(match, seed, config) is degraded and predicted ONCE, exactly as harness.bench_external does
(seed 1000*seed + index, default degrade.PARAMS), and that same prediction is scored against the
ground truth at every value with harness._scores (possession_metrics.evaluate). The comparison
across values is therefore paired. Gap lengths for degrade are coverage_sweep's log-normal
stand-in for the ATD's no-ball runs: this script reads no match data.

A match counts as present if external.available() says so or its converted cache file exists
(the cache is shared through POMPEU_CACHE_DIR; the raw XML may live elsewhere). Nothing is
downloaded.
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import external, paths, pipeline  # noqa: E402
from pompeu.external import degrade, dfl, metrica  # noqa: E402
from pompeu.io import gt as gt_io  # noqa: E402
from pompeu.validate import harness  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
VALUES = (0.0, 2.0, 5.0, 8.0)
CONFIGS = ("rules_v0", "hsmm_v1", "hsmm_v1_rules", "deadball_vote125", "baseline_smooth50_clean")

_spec = importlib.util.spec_from_file_location(
    "coverage_sweep", os.path.join(os.path.dirname(os.path.abspath(__file__)), "coverage_sweep.py"))
_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweep)


def present(name):
    """True if the match can be loaded without downloading: raw files or the converted cache."""
    if external.available(name):
        return True
    module, key = external.parse(name)
    path = dfl._cache_path(key) if module is dfl else metrica._cache_path(key)
    return os.path.isfile(path)


def n_possessions(codes):
    """Number of H/A segments (true possessions) in a code array, X-separated runs respected."""
    return sum(1 for _, _, k in pm.segments(codes, pm.valid_runs(codes)) if k in (pm.H, pm.A))


def ranks(means):
    """{config: rank} with 1 = best (highest score); ties share the better rank."""
    ordered = sorted(means.values(), reverse=True)
    return {c: 1 + ordered.index(v) for c, v in means.items()}


def order(means):
    """Configs best to worst (ties broken by name, so the order is deterministic)."""
    return tuple(sorted(means, key=lambda c: (-means[c], c)))


def ranking_changes(means_by_value):
    """{value: order} and whether any value orders the configs differently from the others."""
    orders = {v: order(m) for v, m in means_by_value.items()}
    return orders, len(set(orders.values())) > 1


def run(matches, seeds, values, configs, run_lengths, verbose=True):
    """rows [{match, seed, config, value, spell, macro, poss, gt_poss}], {match: {value: n}}."""
    rows, counts = [], {}
    for index, name in enumerate(matches):
        tracking, labels = external.load(name)
        fps = tracking.meta.get("fps", 25)
        graded = np.where(tracking.half() > 0, labels, pm.X).astype(np.int8)
        gts = {v: external.align_labels(graded, v, fps) for v in values}
        counts[name] = {v: n_possessions(g) for v, g in gts.items()}
        if verbose:
            print("{}: true possessions {}".format(name, ", ".join(
                "{:g}s={}".format(v, counts[name][v]) for v in values)), flush=True)
        for seed in seeds:
            degraded = degrade.degrade(tracking, labels, 1000 * seed + index, run_lengths, None)
            for cname, cfg in configs.items():
                started = time.time()
                pred = gt_io.labels_to_codes(pipeline.predict(degraded, cfg["pipeline"]))
                for v in values:
                    rows.append(dict(match=name, seed=seed, config=cname, value=v,
                                     **harness._scores(gts[v], pred)))
                if verbose:
                    print("  seed {} {:<24} ".format(seed, cname) + " ".join(
                        "{:g}s {:.4f}".format(r["value"], r["spell"]) for r in rows[-len(values):])
                        + "  ({:.0f} s)".format(time.time() - started), flush=True)
    return rows, counts


def means(rows, metric):
    """{value: {config: mean over (match, seed)}}."""
    acc = {}
    for r in rows:
        acc.setdefault(r["value"], {}).setdefault(r["config"], []).append(r[metric])
    return {v: {c: float(np.mean(x)) for c, x in by.items()} for v, by in acc.items()}


def report(rows, counts, values, configs):
    lines = []
    for metric in ("spell", "macro"):
        table = means(rows, metric)
        lines.append("{}F1: mean over matches x seeds (rank)".format(metric))
        lines.append("  {:<24} ".format("config") + " ".join(
            "{:>14}".format("{:g} s".format(v)) for v in values))
        for c in configs:
            rk = {v: ranks(table[v]) for v in values}
            lines.append("  {:<24} ".format(c) + " ".join(
                "{:>14}".format("{:.4f} ({})".format(table[v][c], rk[v][c])) for v in values))
        orders, changed = ranking_changes(table)
        lines.append("  ranking changes across values: {}".format("YES" if changed else "no"))
        if changed:
            for v in values:
                lines.append("    {:g} s: {}".format(v, " > ".join(orders[v])))
        lines.append("")
    lines.append("true possessions per match")
    for name, by in counts.items():
        lines.append("  {:<14} ".format(name) + "  ".join(
            "{:g}s={}".format(v, by[v]) for v in values))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--values", type=float, nargs="+", default=list(VALUES))
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--run-lengths", help=".npy of ATD no-ball run lengths in frames")
    args = parser.parse_args()

    names = [external.canonical(m) for m in external.SPLITS["val"]]      # never "test"
    matches = [m for m in names if present(m)]
    for m in names:
        if m not in matches:
            print("skipped {}: not present locally (not downloaded)".format(m))
    if not matches:
        sys.exit("no val matches present")
    configs = {}
    for c in args.configs:
        path = os.path.join(paths.CONFIGS_DIR, c + ".yaml")
        if os.path.isfile(path):
            configs[c] = harness.load_config(path)
        else:
            print("skipped config {}: not in configs/".format(c))
    run_lengths = (np.load(args.run_lengths) if args.run_lengths
                   else _sweep.synthetic_run_lengths())
    started = time.time()
    rows, counts = run(matches, args.seeds, args.values, configs, run_lengths)
    print()
    print("val matches {}, seeds {}, degrade.PARAMS default, gaps {}".format(
        ", ".join(matches), args.seeds, args.run_lengths or "log-normal stand-in"))
    print(report(rows, counts, args.values, list(configs)))
    print("\nruntime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
