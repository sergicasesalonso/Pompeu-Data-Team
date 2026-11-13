"""Robustness of candidate configs on external VAL: seeds and min_possession_s.

    python scripts/experiments/robust_val.py                                  # seeds 0-4
    python scripts/experiments/robust_val.py --configs rules_v0 hsmm_v2_fit hsmm_v2_fit2

Only the `val` external matches (never `test`). Each (match, seed) is degraded exactly like
harness.bench_external (seed 1000*seed + index, default degrade.PARAMS, the ATD's own no-ball run
lengths as run_validation passes them; only their lengths are read). rules_v0-style configs run
through pipeline.predict; hsmm_v1 configs are cleaned (A1 defaults) and prepared once per
(match, seed, deadball/touches params) and only re-decoded per config (hsmm.decode_prepared) --
the same computation as pipeline.predict, shared.

Scoring is the harness's (scripts/experiments/gap_team_weight.score: matches of one seed joined
with an X frame, harness._scores, bootstrap.intervals over seeds and ground-truth possessions) at
min_possession_s 2, 5 and 8 s; the prediction is the same, only the ground truth's merging
changes (paired). Per match as well. Nothing is printed per frame.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from pompeu import external, paths, pipeline  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.external import degrade  # noqa: E402
from pompeu.io import gt as gt_io  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.models import hsmm  # noqa: E402
from pompeu.validate import harness  # noqa: E402
from pompeu.validate.stress import missing_run_lengths  # noqa: E402
import possession_metrics as pm  # noqa: E402
from gap_team_weight import present, score  # noqa: E402

VALUES = (2.0, 5.0, 8.0)
CONFIGS = ("rules_v0", "hsmm_v1", "hsmm_v2", "hsmm_v2_fit")


def collect(matches, seeds, configs, run_lengths):
    """{value: [gt per match]}, {config: {seed: [codes per match]}}."""
    truths, preds = {v: [] for v in VALUES}, {c: {} for c in configs}
    for index, name in enumerate(matches):
        tracking, labels = external.load(name)
        fps = tracking.meta.get("fps", 25)
        graded = np.where(tracking.half() > 0, labels, pm.X).astype(np.int8)
        for v in VALUES:
            truths[v].append(external.align_labels(graded, v, fps))
        for seed in seeds:
            started = time.time()
            degraded = degrade.degrade(tracking, labels, 1000 * seed + index, run_lengths, None)
            clean, preps = None, {}
            for cname, cfg in configs.items():
                pcfg = cfg["pipeline"]
                params = pcfg.get("params") or {}
                if pcfg["method"] == "hsmm_v1" and pcfg.get("clean") == {}:
                    key = json.dumps([params.get("deadball") or {}, params.get("touches") or {}],
                                     sort_keys=True)
                    if key not in preps:
                        if clean is None:
                            clean, _ = clean_tracking(degraded, {})
                        preps[key] = hsmm.prepare(clean, params.get("deadball") or None,
                                                  params.get("touches") or None)
                    codes = pm.label_codes(
                        hsmm.decode_prepared(preps[key], params.get("hsmm") or {})["labels"])
                else:
                    codes = gt_io.labels_to_codes(pipeline.predict(degraded, pcfg))
                preds[cname].setdefault(seed, []).append(codes)
            print("  {} seed {} done ({:.0f} s)".format(name, seed, time.time() - started),
                  flush=True)
    return truths, preds


def fmt(r):
    return "{:.4f} [{:.3f}, {:.3f}]  {:.4f} [{:.3f}, {:.3f}]  {:6.1f}/{}".format(
        r["spell"], r["spell_lo"], r["spell_hi"], r["macro"], r["macro_lo"], r["macro_hi"],
        r["poss"], r["gt_poss"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    started = time.time()

    names = [external.canonical(m) for m in external.SPLITS["val"]]         # never "test"
    matches = [m for m in names if present(m)]
    configs = {c: harness.load_config(os.path.join(paths.CONFIGS_DIR, c + ".yaml"))
               for c in args.configs}
    run_lengths = missing_run_lengths(load_tracking("atd"))
    print("val matches {}, seeds {}, configs {}".format(matches, args.seeds, list(configs)))
    truths, preds = collect(matches, args.seeds, configs, run_lengths)

    subsets = [("all val", None)] + [(m, [i]) for i, m in enumerate(matches)]
    means = {}
    for v in VALUES:
        for sname, keep in subsets:
            print("\nmin_possession_s {:g}, {}:  spellF1 [5%, 95%]   macroF1 [5%, 95%]   "
                  "poss/gt".format(v, sname))
            for c in configs:
                r = score(truths[v], preds[c], args.draws, keep)
                if keep is None:
                    means.setdefault(v, {})[c] = r["spell"]
                print("  {:<16} {}".format(c, fmt(r)), flush=True)
    print("\nranking (all val, spellF1)")
    for v in VALUES:
        print("  {:g} s: {}".format(v, " > ".join(sorted(means[v], key=lambda c: -means[v][c]))))
    print("runtime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
