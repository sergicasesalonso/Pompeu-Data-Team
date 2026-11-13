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

Second fit (writes configs/hsmm_v2_fitavg2.yaml):

    python scripts/experiments/fit_hsmm_avg.py --start configs/hsmm_v2_fitavg.yaml         --gap-team-model configs/gap_team_v2.json --grid wide --out configs/hsmm_v2_fitavg2.yaml

starts from hsmm_v2_fitavg's weights with the augmented gap-team expert (gap_team_v2) and widens
the grids of the parameters that hsmm_v2_fitavg left on an edge (WIDE below).
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
# --grid wide: fit_hsmm.GRID with the edges hsmm_v2_fitavg hit opened up, and gap_team_weight
# over a wider range (its gap_team_v2 expert has never been fitted).
WIDE = dict(fit_hsmm.GRID)
WIDE.update({
    "gap_team_weight": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
    "a3_segment_llr": [3.0, 6.0, 10.0, 15.0],
    "w_dead": [1.0, 2.0, 3.0, 4.0],
    "p_poss_to_dead": [0.3, 0.4, 0.5, 0.6],
    "poss_median_s": [4.0, 6.0, 8.0, 12.0],
    "w_proximity": [0.0, 0.25, 0.5, 1.0],
})
GRIDS = {"v1": GRID, "wide": WIDE}
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
    parser.add_argument("--start", default=START, help="config whose hsmm params start the fit")
    parser.add_argument("--gap-team-model", default=None,
                        help="override the start's hsmm.gap_team_model (path in the config)")
    parser.add_argument("--grid", choices=sorted(GRIDS), default="v1")
    args = parser.parse_args()
    grid = GRIDS[args.grid]
    start_name = os.path.splitext(os.path.basename(args.start))[0]
    started_all = time.time()

    with open(args.start, encoding="utf-8") as handle:
        start_cfg = yaml.safe_load(handle)
    start = dict(start_cfg["pipeline"]["params"]["hsmm"])
    if args.gap_team_model:
        start["gap_team_model"] = args.gap_team_model.replace(os.sep, "/")
    deadball_params = start_cfg["pipeline"]["params"].get("deadball") or {}

    preps, truths = prepared_val(args.seeds, deadball_params)
    current = dict(start)
    best, best_each = score(preps, truths, current)
    start_best, start_each = best, best_each
    print("start (" + start_name + ") mean spellF1 {:.4f}  {}".format(best, fmt(best_each)), flush=True)
    for npass in range(args.passes):
        for name, values in grid.items():
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
    edges = {k: v for k, v in current.items() if k in grid and v in (grid[k][0], grid[k][-1])}
    print("on a grid edge: {}".format(edges or "none"))
    cfg = {"name": os.path.splitext(os.path.basename(args.out))[0], "tracking": "atd",
           "pipeline": {"method": "hsmm_v1",
                        "params": {"hsmm": current, "deadball": deadball_params},
                        "clean": {}}}
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "# Fitted by scripts/experiments/fit_hsmm_avg.py from {} (gap_team_model {},\n"
            "# grid {}) on the val external matches ({}), seeds {}; objective = mean val\n"
            "# spellF1 over min_possession_s 2/5/8 s. val mean {:.4f} ({}). Test never used.\n"
            "# On a grid edge: {}.\n"
            .format(start_name, current.get("gap_team_model"), args.grid,
                    ", ".join(external.resolve("val")), args.seeds, best, fmt(best_each),
                    edges or "none"))
        yaml.safe_dump(cfg, handle, sort_keys=False)
    print("wrote", args.out)
    print("runtime {:.0f} s".format(time.time() - started_all))


if __name__ == "__main__":
    main()
