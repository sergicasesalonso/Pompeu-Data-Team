"""The single, pre-registered pass of the final candidates over the external TEST split.

    python scripts/experiments/test_pass.py --workers 6

Fixed before running (nothing on test is tuned or picked): the three configs rules_v0,
hsmm_v1_fit_mbr and hsmm_tcn exactly as committed; the harness's external bench exactly
(split "test", seeds 0..1, degradation seed 1000*seed + index, degrade defaults, the ATD's
no-ball run lengths, pipeline.predict). Each (config, match, seed) is predicted once and scored
at min_possession_s 2, 5 (the bench's own value, the one the decision rule reads) and 8, with
the harness's bootstrap interval. On external matches hsmm_tcn sees NO Smart Tagging (an empty
table, pompeu.experimental.hsmm_tcn), so its number here is the conservative "no tags" case.
full_match_sanity is also run for every config. The public 5 minutes are not touched.

Only aggregated numbers are printed and written to TORRE_METRICS.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pompeu import external, paths  # noqa: E402
from pompeu.validate import harness  # noqa: E402
import possession_metrics as pm  # noqa: E402

CONFIGS = ("rules_v0", "hsmm_v1_fit_mbr", "hsmm_tcn")
VALUES = (2.0, 5.0, 8.0)
SEEDS = 2                      # harness default for the external bench
_STATE = {}


def _cfg(name):
    return harness.load_config(os.path.join(paths.CONFIGS_DIR, name + ".yaml"))


def _init(threads):
    import torch
    torch.set_num_threads(threads)
    from pompeu.io.contract import load_tracking
    from pompeu.validate.stress import missing_run_lengths
    _STATE["run_lengths"] = missing_run_lengths(load_tracking("atd"))


def _job(job):
    kind, config, index, name, seed = job
    from pompeu import pipeline
    from pompeu.io import gt as gt_io
    started = time.time()
    cfg = _cfg(config)
    if kind == "full":
        from pompeu.io.contract import load_tracking
        labels = pipeline.predict(load_tracking(cfg["tracking"]), cfg["pipeline"])
        sanity = harness.bench_full_match_sanity(cfg, gt_io.labels_to_codes(labels))
        return kind, config, index, seed, sanity, time.time() - started
    from pompeu.external import degrade
    tracking, labels = external.load(name)
    degraded = degrade.degrade(tracking, labels, 1000 * seed + index, _STATE["run_lengths"],
                               cfg["validation"]["external"].get("degrade"))
    codes = gt_io.labels_to_codes(pipeline.predict(degraded, cfg["pipeline"]))
    return kind, config, index, seed, codes, time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    from pompeu.validate import bootstrap
    started = time.time()

    matches = external.resolve("test")
    for config in CONFIGS:
        ext = _cfg(config)["validation"]["external"]
        assert ext["datasets"] == "test" and int(ext["seeds"]) == SEEDS, config
    print("test {}, seeds {}, configs {}".format(matches, SEEDS, list(CONFIGS)), flush=True)

    gap = np.full(1, pm.X, dtype=np.int8)
    truths = {v: [] for v in VALUES}
    for name in matches:
        tracking, labels = external.load(name)
        graded = np.where(tracking.half() > 0, labels, pm.X).astype(np.int8)
        for v in VALUES:
            truths[v].append(external.align_labels(graded, v, tracking.meta.get("fps", 25)))
    gts = {v: np.concatenate([p for g in truths[v] for p in (g, gap)]) for v in VALUES}

    jobs = [("full", c, 0, None, 0) for c in CONFIGS]
    jobs += [("ext", c, i, m, s) for c in CONFIGS for i, m in enumerate(matches)
             for s in range(SEEDS)]
    codes, sanity = {}, {}
    with get_context("spawn").Pool(args.workers, initializer=_init,
                                   initargs=(args.threads,)) as pool:
        for kind, config, index, seed, out, secs in pool.imap_unordered(_job, jobs):
            if kind == "full":
                sanity[config] = out
            else:
                codes[(config, seed, index)] = out
            print("  {} {} {} seed {} ({:.0f} s)".format(kind, config, index, seed, secs),
                  flush=True)

    metrics = {"split": "test", "seeds": SEEDS, "n_matches": len(matches)}
    for config in CONFIGS:
        preds = [np.concatenate([p for i in range(len(matches))
                                 for p in (codes[(config, s, i)], gap)]) for s in range(SEEDS)]
        for v in VALUES:
            scores = [harness._scores(gts[v], p) for p in preds]
            ci = bootstrap.intervals(gts[v], preds, args.draws)
            key = "{}.{:g}s".format(config, v)
            metrics[key + ".spell"] = round(float(np.mean([s["spell"] for s in scores])), 4)
            metrics[key + ".spell_lo"] = round(float(ci["spell"][0]), 4)
            metrics[key + ".spell_hi"] = round(float(ci["spell"][1]), 4)
            metrics[key + ".macro"] = round(float(np.mean([s["macro"] for s in scores])), 4)
            metrics[key + ".macro_lo"] = round(float(ci["macro"][0]), 4)
            metrics[key + ".macro_hi"] = round(float(ci["macro"][1]), 4)
            metrics[key + ".poss"] = round(float(np.mean([s["poss"] for s in scores])), 1)
            metrics[key + ".gt_poss"] = scores[0]["gt_poss"]
            print("{:<16} {:g} s  spellF1 {:.4f} [{:.4f}, {:.4f}]  macroF1 {:.4f}".format(
                config, v, metrics[key + ".spell"], metrics[key + ".spell_lo"],
                metrics[key + ".spell_hi"], metrics[key + ".macro"]), flush=True)
        for k, val in sanity[config].items():
            metrics["{}.full_{}".format(config, k)] = (int(val) if isinstance(val, bool)
                                                       else round(float(val), 3))
        print("{:<16} full_match_sanity {}".format(config, sanity[config]), flush=True)
    metrics["runtime_s"] = round(time.time() - started, 1)
    if os.environ.get("TORRE_METRICS"):
        with open(os.environ["TORRE_METRICS"], "w", encoding="utf-8") as handle:
            json.dump(metrics, handle)
    print("runtime {:.0f} s".format(metrics["runtime_s"]))


if __name__ == "__main__":
    main()
