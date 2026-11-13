"""Score the experimental TCN on external VAL against rules_v0 and hsmm_v1_fit_mbr.

    python scripts/experiments/tcn_val.py --model cache/tcn/tcn_v1.pt --seeds 0 1 2 --workers 3

Only the `val` external matches (never `test`). Each (match, seed) is degraded exactly like
robust_val.py (seed 1000*seed + index, degrade defaults, the ATD's no-ball run lengths) and every
candidate sees the SAME degraded match:

  rules_v0            pipeline.predict (configs/rules_v0.yaml)
  hsmm_v1_fit_mbr     cleaned (A1 defaults), hsmm.prepare, decode_prepared, A10 policy -- the
                      config exactly (robust_val's shared-prepare shortcut, plus the policy)
  tcn s=<k>           the TCN alone: argmax of its per-frame posterior after a k-frame moving
                      average, A10 policy
  hsmm+tcn wt/wd      hsmm_v1_fit_mbr with the TCN as one more expert (product of experts): its
                      team log-odds log P(H) - log P(A) with weight wt and its dead log-odds
                      log P(D) - log(P(H) + P(A)) with weight wd, through the HSMM's learned-
                      expert slot (scaled per look like every per-frame cue). Viterbi for the grid
                      (mbr off), MBR for the --mbr pairs.

The TCN's inputs use synthetic Smart Tagging (tag seed TAG_SEED + degradation seed); the two
baselines run as on the bench (no tags on external matches). Scoring: gap_team_weight.score
(seeds joined, harness scores, bootstrap intervals) at min_possession_s 2, 5, 8; decision rule B:
mean spellF1 over the three and the worst loss versus rules_v0. Weights / smoothing are picked
AND reported on val: optimistic. Nothing is printed per frame.
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
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from pompeu import external, paths  # noqa: E402
from pompeu.experimental import tcn  # noqa: E402
from pompeu.validate import harness  # noqa: E402
import possession_metrics as pm  # noqa: E402

VALUES = (2.0, 5.0, 8.0)
_STATE = {}


def _init(model_path, threads):
    import torch
    torch.set_num_threads(threads)
    # deadball_gbm.VERSION = 2 but only artifacts/deadball_gbm_v1.txt is committed (known, Sergi
    # to fix): use v1 when v2 is missing, as the earlier hsmm_v1_fit_mbr val runs did.
    from pompeu.models import deadball_gbm
    if not os.path.isfile(deadball_gbm.ARTIFACT):
        art = os.path.join(paths.ROOT, "artifacts", "deadball_gbm_v1.txt")
        deadball_gbm.load.__defaults__ = (art,)
        deadball_gbm.ARTIFACT = art
    from pompeu.io.contract import load_tracking
    from pompeu.validate.stress import missing_run_lengths
    _STATE["run_lengths"] = missing_run_lengths(load_tracking("atd"))
    _STATE["model"], _STATE["norm"] = tcn.load(model_path, "cpu")
    _STATE["rules"] = harness.load_config(os.path.join(paths.CONFIGS_DIR, "rules_v0.yaml"))
    _STATE["hsmm"] = harness.load_config(os.path.join(paths.CONFIGS_DIR,
                                                      "hsmm_v1_fit_mbr.yaml"))


def _job(job):
    index, name, seed, smooth, grid, mbr_grid = job
    from pompeu import pipeline
    from pompeu.clean import clean_tracking
    from pompeu.decode.policy import finalize
    from pompeu.external import degrade, synthetic_tags as st
    from pompeu.models import hsmm
    started = time.time()
    tracking, labels = external.load(name)
    fps = tracking.meta.get("fps", 25)
    half = tracking.half()
    graded = np.where(half > 0, labels, pm.X).astype(np.int8)
    dseed = 1000 * seed + index
    degraded = degrade.degrade(tracking, labels, dseed, _STATE["run_lengths"], None)
    preds = {"rules_v0": pm.label_codes(pipeline.predict(degraded, _STATE["rules"]["pipeline"]))}

    hcfg = _STATE["hsmm"]["pipeline"]
    params = hcfg["params"]
    base = dict(params.get("hsmm") or {})
    clean, _ = clean_tracking(degraded, hcfg.get("clean") or {})
    prep = hsmm.prepare(clean, params.get("deadball") or None, params.get("touches") or None)

    def run(p):
        out = hsmm.decode_prepared(prep, dict(p))["labels"]
        return pm.label_codes(finalize(out, clean, hcfg.get("policy") or {}))

    preds["hsmm_v1_fit_mbr"] = run(base)
    preds["hsmm_v1_fit (viterbi)"] = run(dict(base, mbr=False))

    tags = st.generate(external.align_labels(graded, 2.0, fps), half, st.load_error_model(),
                       seed=tcn.TAG_SEED + dseed, fps=fps)
    X, names = tcn.inputs_from(clean, tags)
    log_p = tcn.predict_frames(_STATE["model"], _STATE["norm"], X, names, clean.n_frames)
    for k in smooth:
        lab = np.array(["H", "A", "D"])[tcn.smooth_labels(log_p, k)]
        preds["tcn s={}".format(k)] = pm.label_codes(finalize(lab, clean, hcfg.get("policy") or {}))
    prep["learned"] = {"team": log_p[:, 0] - log_p[:, 1],
                       "dead": log_p[:, 2] - np.logaddexp(log_p[:, 0], log_p[:, 1])}
    for wt, wd in grid:
        preds["hsmm+tcn {:g}/{:g} vit".format(wt, wd)] = run(
            dict(base, mbr=False, w_learned_team=wt, w_learned_dead=wd))
    for wt, wd in mbr_grid:
        preds["hsmm+tcn {:g}/{:g} mbr".format(wt, wd)] = run(
            dict(base, w_learned_team=wt, w_learned_dead=wd))
    return index, seed, preds, time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--smooth", type=int, nargs="+", default=[1, 25, 50, 100])
    parser.add_argument("--w-team", type=float, nargs="+", default=[0.5, 1, 2, 4, 8])
    parser.add_argument("--w-dead", type=float, nargs="+", default=[0, 1, 2])
    parser.add_argument("--mbr", nargs="*", default=[],
                        help="wt/wd pairs decoded with MBR, e.g. 2/1 4/0")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    from gap_team_weight import present, score
    from robust_val import fmt
    started = time.time()

    names = [external.canonical(m) for m in external.SPLITS["val"]]          # never "test"
    matches = [m for m in names if present(m)]
    grid = [(wt, wd) for wt in args.w_team for wd in args.w_dead]
    mbr_grid = [tuple(float(x) for x in pair.split("/")) for pair in args.mbr]
    print("val {}, seeds {}, model {}, {} grid + {} mbr".format(
        matches, args.seeds, os.path.basename(args.model), len(grid), len(mbr_grid)), flush=True)

    truths = {v: [] for v in VALUES}
    for name in matches:
        tracking, labels = external.load(name)
        graded = np.where(tracking.half() > 0, labels, pm.X).astype(np.int8)
        for v in VALUES:
            truths[v].append(external.align_labels(graded, v, tracking.meta.get("fps", 25)))

    jobs = [(i, m, s, args.smooth, grid, mbr_grid) for i, m in enumerate(matches)
            for s in args.seeds]
    preds = {}
    with get_context("spawn").Pool(args.workers, initializer=_init,
                                   initargs=(args.model, args.threads)) as pool:
        for index, seed, out, secs in pool.imap_unordered(_job, jobs):
            for c, codes in out.items():
                preds.setdefault(c, {}).setdefault(seed, {})[index] = codes
            print("  {} seed {} ({:.0f} s)".format(matches[index], seed, secs), flush=True)
    preds = {c: {s: [d[i] for i in range(len(matches))] for s, d in by.items()}
             for c, by in preds.items()}

    results = {}
    for v in VALUES:
        for c in preds:
            results.setdefault(c, {})[v] = score(truths[v], preds[c], args.draws)
    order = ["rules_v0", "hsmm_v1_fit_mbr", "hsmm_v1_fit (viterbi)"]
    order += sorted(c for c in preds if c not in order)
    for v in VALUES:
        print("\nmin_possession_s {:g}, all val:  spellF1 [5%, 95%]   macroF1 [5%, 95%]   "
              "poss/gt".format(v))
        for c in order:
            print("  {:<26} {}".format(c, fmt(results[c][v])))
    print("\ndecision rule B (all val): mean spellF1 2/5/8, worst loss vs rules_v0, mean macroF1")
    summary = {}
    for c in order:
        spell = [results[c][v]["spell"] for v in VALUES]
        loss = max(results["rules_v0"][v]["spell"] - results[c][v]["spell"] for v in VALUES)
        summary[c] = {"mean_spell": float(np.mean(spell)), "worst_loss": float(loss),
                      "mean_macro": float(np.mean([results[c][v]["macro"] for v in VALUES])),
                      "by_value": {str(v): results[c][v] for v in VALUES}}
    for c in sorted(order, key=lambda c: -summary[c]["mean_spell"]):
        s = summary[c]
        print("  {:<26} {:.4f}  {:+.4f}  {:.4f}{}".format(
            c, s["mean_spell"], s["worst_loss"], s["mean_macro"],
            "" if s["worst_loss"] <= 0.02 else "  (fails max-loss)"))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1)
    print("runtime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
