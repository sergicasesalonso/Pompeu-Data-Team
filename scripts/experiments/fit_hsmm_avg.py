"""Fit the HSMM weights on val against the MEAN spellF1 over min_possession_s 2, 5 and 8 s.

    python scripts/experiments/fit_hsmm_avg.py        # val, seeds 0-1, writes configs/hsmm_v2_fitavg.yaml

scripts/fit_hsmm.py scores against labels aligned with min_possession_s = 5 s only, a convention
first matched to the public ground truth (which the rules no longer allow for tuning); the weights
it finds make possessions long and few and lose to rules_v0 at 2 s. Here the objective does not
lean on one convention: each candidate is decoded once and the same predictions are scored against
the val labels aligned three times (2, 5 and 8 s); the objective is the mean of the three spellF1.

Everything else as fit_hsmm.py: seeds 0-1 degraded with the ATD no-ball run lengths, cleaned and
prepared once (hsmm.prepare); coordinate descent over fit_hsmm.GRID plus gap_team_weight
[0, 0.25, 0.5, 1], two passes, acceptance > best + 1e-4. Start: hsmm_v2 (gap_team_weight 0.25,
configs/gap_team_v1.json, other HSMM params at defaults). Only `val`; test is never touched.
Nothing is printed per frame.
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pompeu import external, paths  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.external import degrade  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.models import hsmm  # noqa: E402
from pompeu.validate.stress import missing_run_lengths  # noqa: E402
import possession_metrics as pm  # noqa: E402
import fit_hsmm  # noqa: E402

VALUES = (2.0, 5.0, 8.0)
GRID = dict(fit_hsmm.GRID)
GRID["gap_team_weight"] = [0.0, 0.25, 0.5, 1.0]
START = os.path.join(paths.CONFIGS_DIR, "hsmm_v2.yaml")
OUT = os.path.join(paths.CONFIGS_DIR, "hsmm_v2_fitavg.yaml")


def prepared_val(seeds, deadball_params):
    """As fit_hsmm.prepared_val, but the ground truth aligned at every value in VALUES."""
    run_lengths = missing_run_lengths(load_tracking("atd"))
    preps, truths = [], {v: [] for v in VALUES}
    for name in external.resolve("val"):
        tracking, labels = external.load(name)
        fps = tracking.meta.get("fps", 25)
        graded = np.where(tracking.half() > 0, labels, pm.X).astype(np.int8)
        aligned = {v: external.align_labels(graded, v, fps) for v in VALUES}
        for seed in seeds:
            degraded = degrade.degrade(tracking, labels, seed, run_lengths)
            clean, _ = clean_tracking(degraded, {})
            preps.append(hsmm.prepare(clean, deadball_params))
            for v in VALUES:
                truths[v].append(aligned[v])
            print("prepared", name, seed, flush=True)
    return preps, truths


def score(preps, truths, params):
    """(mean spellF1, {value: (spellF1, macroF1)}): one decode, three ground truths."""
    gap = np.full(1, pm.X, dtype=np.int8)
    pred = np.concatenate([part for prep in preps for part in (
        pm.label_codes(hsmm.decode_prepared(prep, params)["labels"]), gap)])
    each = {}
    for v in VALUES:
        gt = np.concatenate([part for t in truths[v] for part in (t, gap)])
        result = pm.evaluate(gt, pred)
        each[v] = (result["spell_f1_mean"], result["macro_f1"])
    return float(np.mean([each[v][0] for v in VALUES])), each


def fmt(each):
    return "  ".join("{:g}s {:.4f}".format(v, each[v][0]) for v in VALUES)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    started_all = time.time()

    with open(START, encoding="utf-8") as handle:
        start_cfg = yaml.safe_load(handle)
    start = dict(start_cfg["pipeline"]["params"]["hsmm"])
    deadball_params = start_cfg["pipeline"]["params"].get("deadball") or {}

    preps, truths = prepared_val(args.seeds, deadball_params)
    current = dict(start)
    best, best_each = score(preps, truths, current)
    start_best, start_each = best, best_each
    print("start (hsmm_v2) mean spellF1 {:.4f}  {}".format(best, fmt(best_each)), flush=True)
    for npass in range(args.passes):
        for name, values in GRID.items():
            for value in values:
                trial = dict(current, **{name: value})
                if trial == current:
                    continue
                started = time.time()
                mean, each = score(preps, truths, trial)
                print("  {}={} mean {:.4f}  {} ({:.0f} s)".format(
                    name, value, mean, fmt(each), time.time() - started), flush=True)
                if mean > best + 1e-4:
                    best, best_each, current = mean, each, trial
            print("pass {} {} -> {}  mean {:.4f}".format(npass + 1, name, current.get(name),
                                                        best), flush=True)

    moved = {k: (start.get(k), v) for k, v in current.items() if start.get(k) != v}
    print("\nmoved: {}".format(moved or "nothing"))
    print("val mean spellF1 {:.4f} -> {:.4f}; {}".format(start_best, best, fmt(best_each)))
    print("  (start {})".format(fmt(start_each)))
    edges = {k: v for k, v in current.items() if k in GRID and v in (GRID[k][0], GRID[k][-1])}
    print("on a grid edge: {}".format(edges or "none"))
    cfg = {"name": os.path.splitext(os.path.basename(args.out))[0], "tracking": "atd",
           "pipeline": {"method": "hsmm_v1",
                        "params": {"hsmm": current, "deadball": deadball_params},
                        "clean": {}}}
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "# Fitted by scripts/experiments/fit_hsmm_avg.py from hsmm_v2 on the val external\n"
            "# matches ({}), seeds {}; objective = mean val spellF1 over min_possession_s\n"
            "# 2/5/8 s. val mean {:.4f} ({}). Test never used.\n"
            .format(", ".join(external.resolve("val")), args.seeds, best, fmt(best_each)))
        yaml.safe_dump(cfg, handle, sort_keys=False)
    print("wrote", args.out)
    print("runtime {:.0f} s".format(time.time() - started_all))


if __name__ == "__main__":
    main()
