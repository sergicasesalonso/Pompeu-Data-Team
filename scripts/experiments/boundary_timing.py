"""Calibrate WHEN a config places possession changes, on external VAL only ("si acabes abans", 2).

    python scripts/experiments/boundary_timing.py                       # hsmm_v1_fit_mbr
    python scripts/experiments/boundary_timing.py --config configs/rules_v0.yaml

spellF1 matches possessions by temporal IoU, so a boundary that is systematically late (or early)
costs IoU on every possession even when the change itself is found. This measures the signed lag
(predicted minus true frame) of the matched changes, separately for changes TO a team (H/A) and TO
dead ball (D), and checks whether moving the predicted boundaries by a constant helps.

Protocol, all on the `val` external matches (never `test`, never the public window), degraded
exactly like harness.bench_external (seed 1000 * seed + index):
  1. FIT seeds (default 0-1): lags measured, and a small grid of shifts scored by the mean spellF1
     over min_possession_s 2/5/8 s (team rule B); the best shift is chosen there.
  2. HELD-OUT seeds (default 2-4): that one shift is scored, against no shift. Only this number
     says whether the calibration generalises; the fit-seed number is in-sample.
The prediction is the full config (pipeline.predict, policy included). Only aggregates are
printed: lag quantiles in frames, spellF1 per convention, counts.
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pompeu import external, paths, pipeline  # noqa: E402
from pompeu.external import degrade  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.validate.stress import missing_run_lengths  # noqa: E402
import possession_metrics as pm  # noqa: E402

VALUES = (2.0, 5.0, 8.0)
TOL_S = 2.0                                   # a change counts as found within +/- 2 s
GRID_S = (-1.0, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 1.0)   # constant shifts tried, seconds


def collect(cfg, seeds, run_lengths):
    """{value: [gt per (match, seed)]}, [pred per (match, seed)], [seed per item]."""
    truths, preds, which = {v: [] for v in VALUES}, [], []
    for index, name in enumerate(external.resolve("val")):
        tracking, labels = external.load(name)
        fps = tracking.meta.get("fps", 25)
        graded = np.where(tracking.half() > 0, labels, pm.X).astype(np.int8)
        aligned = {v: external.align_labels(graded, v, fps) for v in VALUES}
        for seed in seeds:
            started = time.time()
            degraded = degrade.degrade(tracking, labels, 1000 * seed + index, run_lengths, None)
            preds.append(pm.label_codes(pipeline.predict(degraded, cfg["pipeline"])))
            for v in VALUES:
                truths[v].append(aligned[v])
            which.append(seed)
            print("  {} seed {} ({:.0f} s)".format(name, seed, time.time() - started), flush=True)
    return truths, preds, which


def lags(gt, pred, tol):
    """Signed lags (pred - gt, frames) of matched changes, split by destination: team / dead."""
    runs = pm.valid_runs(gt)
    pairs, _, _ = pm.match_transitions(pm.transitions(gt, runs),
                                       pm.transitions(pm.fill_abstentions(pred, runs), runs), tol)
    team = [p[0] - g[0] for g, p in pairs if g[2] in (pm.H, pm.A)]
    dead = [p[0] - g[0] for g, p in pairs if g[2] == pm.D]
    return team, dead


def shift(pred, gt, d_team, d_dead):
    """Copy of pred with every change moved EARLIER by d frames (d < 0: later), per destination.

    A boundary never crosses its neighbours (each segment keeps at least one frame) nor a run
    edge; frames outside the graded runs are left untouched.
    """
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    out = pred.copy()
    for start, stop in runs:
        segs = pm.segments(filled, [(start, stop)])
        bounds = [s[0] for s in segs] + [stop]
        moved = list(bounds)
        for k in range(1, len(segs)):
            d = d_dead if segs[k][2] == pm.D else d_team
            moved[k] = int(np.clip(bounds[k] - d, moved[k - 1] + 1, bounds[k + 1] - 1))
        for k, seg in enumerate(segs):
            out[moved[k]:moved[k + 1]] = seg[2]
    return out


def score(truths, preds, keep):
    """{value: spellF1} over the kept items, matches joined by one X frame (as fit_hsmm)."""
    gap = np.full(1, pm.X, dtype=np.int8)
    out = {}
    for v, gts in truths.items():
        gt = np.concatenate([p for i in keep for p in (gts[i], gap)])
        pr = np.concatenate([p for i in keep for p in (preds[i], gap)])
        out[v] = pm.evaluate(gt, pr)["spell_f1_mean"]
    return out


def fmt(r):
    return "  ".join("{:.4f}".format(r[v]) for v in VALUES) + "  mean {:.4f}".format(
        np.mean(list(r.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(paths.CONFIGS_DIR, "hsmm_v1_fit_mbr.yaml"))
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    started = time.time()
    run_lengths = missing_run_lengths(load_tracking("atd"))
    truths, preds, which = collect(cfg, args.fit_seeds + args.eval_seeds, run_lengths)
    fps = 25
    fit = [i for i, s in enumerate(which) if s in args.fit_seeds]
    held = [i for i, s in enumerate(which) if s in args.eval_seeds]

    print("\nconfig {}, val {}, fit seeds {}, held-out seeds {}".format(
        cfg["name"], external.resolve("val"), args.fit_seeds, args.eval_seeds))
    print("lag of matched changes, pred - gt, frames (25 fps), fit seeds, tol {:g} s".format(TOL_S))
    for v in VALUES:
        team, dead = [], []
        for i in fit:
            t, d = lags(truths[v][i], preds[i], int(TOL_S * fps))
            team += t
            dead += d
        for label, x in (("to H/A", team), ("to D", dead)):
            q = np.percentile(x, [25, 50, 75]) if x else [np.nan] * 3
            print("  {:g} s {:<7} n {:>5}  median {:+6.1f}  IQR [{:+.1f}, {:+.1f}]  mean {:+.1f}".format(
                v, label, len(x), q[1], q[0], q[2], np.mean(x) if x else np.nan))

    print("\nshift grid on FIT seeds (seconds earlier; spellF1 2 / 5 / 8 s, mean)")
    best, best_mean = (0.0, 0.0), -1.0
    for st in GRID_S:
        for sd in GRID_S:
            shifted = [shift(preds[i], truths[5.0][i], int(round(st * fps)), int(round(sd * fps)))
                       if i in fit else preds[i] for i in range(len(preds))]
            r = score(truths, shifted, fit)
            m = float(np.mean(list(r.values())))
            if st == 0.0 or sd == 0.0:
                print("  team {:+.1f}  dead {:+.1f}   {}".format(st, sd, fmt(r)))
            if m > best_mean + 1e-4:
                best, best_mean = (st, sd), m
    print("  best on fit seeds: team {:+.1f} s, dead {:+.1f} s -> mean {:.4f}".format(
        best[0], best[1], best_mean))

    print("\nHELD-OUT seeds (the number that counts)")
    print("  no shift          {}".format(fmt(score(truths, preds, held))))
    shifted = [shift(p, truths[5.0][i], int(round(best[0] * fps)), int(round(best[1] * fps)))
               for i, p in enumerate(preds)]
    print("  team {:+.1f} dead {:+.1f}  {}".format(best[0], best[1], fmt(score(truths, shifted, held))))
    print("runtime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
