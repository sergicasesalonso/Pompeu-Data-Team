"""Choose hsmm's gap_team_weight on external VAL; compare hsmm_v1, hsmm_v2 and rules_v0.

    python scripts/experiments/gap_team_weight.py                       # val, seeds 0-4
    python scripts/experiments/gap_team_weight.py --seeds 0 1 --weights 0 1

Only the `val` external matches (never `test`) present locally. Each (match, seed) is degraded
exactly like harness.bench_external (seed 1000*seed + index, default degrade.PARAMS), with
coverage_sweep's log-normal gap lengths (no BAR-ATH data is read), cleaned with A1 defaults and
prepared ONCE (hsmm.prepare); every weight only re-decodes (hsmm.decode_prepared). Weight 0 is
hsmm_v1 exactly. rules_v0 runs through pipeline.predict on the same degraded tracking.

Scoring is the harness's: the matches of one seed concatenated with an X frame between them,
harness._scores (possession_metrics.evaluate) and bootstrap.intervals over seeds and ground-truth
possessions, at min_possession_s 5 (the bench) and also 2 and 8 s (paired: the prediction is the
same, only the ground truth's merging changes). The weight is the argmax of val spellF1 at 5 s
(ties within 1e-4 go to the smaller weight). Nothing is printed per frame.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pompeu import external, paths, pipeline  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.external import degrade, dfl, metrica  # noqa: E402
from pompeu.io import gt as gt_io  # noqa: E402
from pompeu.models import hsmm  # noqa: E402
from pompeu.validate import bootstrap, harness  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
VALUES = (5.0, 2.0, 8.0)
MODEL = "configs/gap_team_v1.json"


def present(name):
    if external.available(name):
        return True
    module, key = external.parse(name)
    path = dfl._cache_path(key) if module is dfl else metrica._cache_path(key)
    return os.path.isfile(path)


def collect(matches, seeds, weights, run_lengths, rules_cfg):
    """{value: gt list per match}, {config: {seed: [codes per match]}}."""
    truths, preds = {v: [] for v in VALUES}, {}
    for index, name in enumerate(matches):
        tracking, labels = external.load(name)
        fps = tracking.meta.get("fps", 25)
        graded = np.where(tracking.half() > 0, labels, pm.X).astype(np.int8)
        for v in VALUES:
            truths[v].append(external.align_labels(graded, v, fps))
        for seed in seeds:
            started = time.time()
            degraded = degrade.degrade(tracking, labels, 1000 * seed + index, run_lengths, None)
            preds.setdefault("rules_v0", {}).setdefault(seed, []).append(
                gt_io.labels_to_codes(pipeline.predict(degraded, rules_cfg["pipeline"])))
            clean, _ = clean_tracking(degraded, {})
            prep = hsmm.prepare(clean)
            for w in weights:
                params = {"gap_team_weight": w, "gap_team_model": MODEL} if w else {}
                preds.setdefault(w, {}).setdefault(seed, []).append(
                    pm.label_codes(hsmm.decode_prepared(prep, params)["labels"]))
            print("  {} seed {} done ({:.0f} s)".format(name, seed, time.time() - started),
                  flush=True)
    return truths, preds


def join(parts):
    gap = np.full(1, pm.X, dtype=np.int8)
    return np.concatenate([p for part in parts for p in (part, gap)])


def score(truth_parts, by_seed, n_draws, keep=None):
    """Harness-style scores for one config: mean over seeds plus the two-level interval."""
    keep = keep if keep is not None else list(range(len(truth_parts)))
    gt = join([truth_parts[i] for i in keep])
    preds = [join([parts[i] for i in keep]) for _, parts in sorted(by_seed.items())]
    s = [harness._scores(gt, p) for p in preds]
    ci = bootstrap.intervals(gt, preds, n_draws)
    return {"spell": float(np.mean([x["spell"] for x in s])),
            "macro": float(np.mean([x["macro"] for x in s])),
            "poss": float(np.mean([x["poss"] for x in s])), "gt_poss": s[0]["gt_poss"],
            "spell_lo": ci["spell"][0], "spell_hi": ci["spell"][1],
            "macro_lo": ci["macro"][0], "macro_hi": ci["macro"][1]}


def fmt(r):
    return "{:.4f} [{:.3f}, {:.3f}]  {:.4f} [{:.3f}, {:.3f}]  {:6.1f}/{}".format(
        r["spell"], r["spell_lo"], r["spell_hi"], r["macro"], r["macro_lo"], r["macro_hi"],
        r["poss"], r["gt_poss"])


def label(c):
    return c if isinstance(c, str) else ("hsmm_v1 (w=0)" if c == 0 else "w={:g}".format(c))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--weights", type=float, nargs="+", default=list(WEIGHTS))
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    started = time.time()

    import coverage_sweep
    run_lengths = coverage_sweep.synthetic_run_lengths()
    names = [external.canonical(m) for m in external.SPLITS["val"]]         # never "test"
    matches = [m for m in names if present(m)]
    print("val matches {}, seeds {}, weights {}".format(matches, args.seeds, args.weights))
    rules_cfg = harness.load_config(os.path.join(paths.CONFIGS_DIR, "rules_v0.yaml"))
    weights = sorted(set([0.0] + list(args.weights)))
    truths, preds = collect(matches, args.seeds, weights, run_lengths, rules_cfg)

    configs = ["rules_v0"] + weights
    subsets = [("all val", None)] + [
        (m + " only", [i]) for i, m in enumerate(matches) if len(matches) > 1]
    results = {}
    for v in VALUES:
        for sname, keep in subsets:
            if v != 5.0 and keep is not None:
                continue
            print("\nmin_possession_s {:g}, {}:  spellF1 [5%, 95%]   macroF1 [5%, 95%]   "
                  "poss/gt".format(v, sname))
            for c in configs:
                r = score(truths[v], preds[c], args.draws, keep)
                results[(v, sname, c)] = r
                print("  {:<16} {}".format(label(c), fmt(r)), flush=True)

    best = max(weights, key=lambda w: (round(results[(5.0, "all val", w)]["spell"], 4), -w))
    print("\nchosen gap_team_weight (argmax val spellF1 at 5 s): {:g}".format(best))
    print("runtime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
