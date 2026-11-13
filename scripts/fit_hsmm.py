"""Fit the HSMM's few weights on the `val` external matches (plan, stage A5: "the w_k are fitted
only on external").

    python scripts/fit_hsmm.py                         # writes configs/hsmm_v1_fit.yaml
    python scripts/fit_hsmm.py --base '{gbm: 1.0}'     # fit on top of other settings
    python scripts/fit_hsmm.py --learned artifacts/seqmodel_v1.pt --out configs/hsmm_v2_fit.yaml
                                                       # hsmm_v2: A7's sequence model as an expert

Each val match is degraded with seeds 0-1, cleaned and prepared once (hsmm.prepare); every
candidate only rebuilds the emissions and runs Viterbi (hsmm.decode_prepared). The objective is
spellF1 -- the ranking metric -- over all val matches and seeds at once, scored by
possession_metrics.evaluate against the convention-aligned labels.

Coordinate descent: one parameter at a time over its grid, keeping the best, two full passes.
Not fitted: w_tagging (the external matches have no Smart Tagging, so the data says nothing about
it; it keeps its physical value). The test matches are never touched here.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import external, paths  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.external import degrade, examples  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.models import hsmm  # noqa: E402
from pompeu.validate.stress import missing_run_lengths  # noqa: E402
import possession_metrics as pm  # noqa: E402

GRID = {
    "w_proximity": [0.0, 0.5, 1.0, 2.0, 4.0],
    "w_events": [0.0, 0.5, 1.0, 2.0],
    "a3_segment_llr": [0.0, 1.0, 3.0, 6.0],
    "w_dead": [0.5, 1.0, 2.0],
    "rules_llr": [0.0, 0.5, 1.0, 2.0, 4.0],
    "poss_median_s": [4.0, 6.0, 8.0, 12.0],
    "p_poss_to_dead": [0.2, 0.3, 0.4],
}
REPORT = os.path.join(paths.RESULTS_DIR, "hsmm_fit.md")
OUT = os.path.join(paths.CONFIGS_DIR, "hsmm_v1_fit.yaml")


def prepared_val(seeds, deadball_params, learned=None):
    run_lengths = missing_run_lengths(load_tracking("atd"))
    preps, truths = [], []
    for name in external.resolve("val"):
        tracking, labels = external.load(name)
        graded = np.where(tracking.half() > 0, labels, pm.X).astype(np.int8)
        truth = external.align_labels(graded, examples.MIN_POSSESSION_S)
        for seed in seeds:
            degraded = degrade.degrade(tracking, labels, seed, run_lengths)
            clean, _ = clean_tracking(degraded, {})
            prep = hsmm.prepare(clean, deadball_params)
            if learned:
                hsmm.add_learned(prep, learned)
            preps.append(prep)
            truths.append(truth)
            print("prepared", name, seed, flush=True)
    return preps, truths


def score(preps, truths, params):
    gap = np.full(1, pm.X, dtype=np.int8)
    gt = np.concatenate([part for t in truths for part in (t, gap)])
    pred = np.concatenate([part for prep in preps for part in (
        pm.label_codes(hsmm.decode_prepared(prep, params)["labels"]), gap)])
    result = pm.evaluate(gt, pred)
    return result["spell_f1_mean"], result["macro_f1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--base", default="{}", help="hsmm params to start from (YAML)")
    parser.add_argument("--deadball", default="{}", help="deadball params (YAML)")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--learned", nargs="+",
                        help="sequence model(s) to add as the learned expert (A7; several = A8 ensemble)")
    args = parser.parse_args()
    base = yaml.safe_load(args.base) or {}
    deadball_params = yaml.safe_load(args.deadball) or {}

    preps, truths = prepared_val(args.seeds, deadball_params, args.learned)
    grid = dict(GRID)
    current = dict(base)
    if args.learned:
        current["learned_models"] = args.learned
        current.setdefault("w_learned_team", 1.0)
        current.setdefault("w_learned_dead", 1.0)
        grid = dict({"w_learned_team": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
                     "w_learned_dead": [0.0, 0.5, 1.0, 2.0, 4.0]}, **GRID)
    best, best_macro = score(preps, truths, current)
    log = ["start {} -> spellF1 {:.4f} macroF1 {:.4f}".format(current, best, best_macro)]
    print(log[-1], flush=True)
    for _ in range(args.passes):
        for name, values in grid.items():
            for value in values:
                trial = dict(current, **{name: value})
                if trial == current:
                    continue
                started = time.time()
                spell, macro = score(preps, truths, trial)
                print("  {}={} spellF1 {:.4f} ({:.0f} s)".format(name, value, spell,
                                                                time.time() - started), flush=True)
                if spell > best + 1e-4:
                    best, best_macro, current = spell, macro, trial
            log.append("{} -> {}  spellF1 {:.4f}".format(name, current.get(name, "default"), best))
            print(log[-1], flush=True)

    cfg = {"name": os.path.splitext(os.path.basename(args.out))[0], "tracking": "atd",
           "pipeline": {"method": "hsmm_v1",
                        "params": {"hsmm": current, "deadball": deadball_params},
                        "clean": {}}}
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("# Fitted by scripts/fit_hsmm.py on the val external matches ({}), seeds {}.\n"
                     "# val spellF1 {:.4f}, macroF1 {:.4f}. Measured on test by the harness.\n"
                     .format(", ".join(external.resolve("val")), args.seeds, best, best_macro))
        yaml.safe_dump(cfg, handle, sort_keys=False)
    with open(REPORT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Ajuste de pesos del HSMM en `val` (A5)\n\n```\n" + "\n".join(log) +
                     "\n```\n\nResultado: `{}`\n".format(json.dumps(current)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
