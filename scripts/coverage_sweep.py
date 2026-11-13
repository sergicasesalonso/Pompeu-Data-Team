"""How much is a point of ball coverage worth? spellF1 / macroF1 against the ball-missing rate.

    python scripts/coverage_sweep.py                          # val matches, seeds 0-4, nested
    python scripts/coverage_sweep.py --seeds 0 1 2 --mode independent

Line B adds ball observations to the tracking (broadcast fusion: 59.2% -> 62.9% coverage). It
cannot be scored on the external bench directly -- external matches have no video or broadcast --
and the public window may not choose anything. So this measures the curve instead: every `val`
external match (never `test`) is degraded like our ATD (pompeu.external.degrade) with the ball
missing on 50/41/37/30/20% of its in-half frames, rules_v0 and hsmm_v1 run through
pipeline.predict exactly as the harness runs them, and the scores come from
possession_metrics.evaluate through harness._scores.

Two ways to change the rate:

  nested       (default) degrade once at the LOWEST rate, then erase more ball, in ATD-like runs
               (degrade.erase_ball, same run lengths, same dead-ball weighting), to reach each
               higher rate. The levels of one (match, seed) differ ONLY in the ball: same players,
               same swaps, same spikes and jitter on the frames still seen, and the 41% mask
               contains the 37% one. That is what Line B does -- add observations to a fixed
               tracking -- and it makes the per-unit slope a paired difference.
  independent  degrade.degrade with params {"ball_missing": rate}. degrade draws everything from
               one generator, so the number of ball draws shifts every later draw: each rate gets
               a different realisation of the player noise too.

Gap lengths. degrade samples gaps from the ATD's own no-ball runs (stress.missing_run_lengths on
BAR-ATH). This script does not read the match: by default it uses a log-normal stand-in with the
ATD's documented median (10.7 s) and p99 (~105 s; degrade caps at max_gap_s = 105 s anyway).
--run-lengths takes a .npy of run lengths in frames if the real ones have been exported.
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import external, paths, pipeline  # noqa: E402
from pompeu.external import degrade  # noqa: E402
from pompeu.io import gt as gt_io  # noqa: E402
from pompeu.validate import harness  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
RATES = (0.50, 0.41, 0.37, 0.30, 0.20)
REFERENCE = 0.41                     # BAR-ATH today
SLOPES_TO = (0.37, 0.30)             # 41% -> 37% (broadcast fusion, rounded), 41% -> 30%
CONFIGS = ("rules_v0", "hsmm_v1")
GAP_MEDIAN_S = 10.7                  # ATD no-ball runs (stress.py, degrade.py)
GAP_P99_S = 105.0
N_GAP_DRAWS = 20000


def synthetic_run_lengths(fps=25, median_s=GAP_MEDIAN_S, p99_s=GAP_P99_S, n=N_GAP_DRAWS, seed=0):
    """(n,) int64 gap lengths in frames, log-normal with the given median and 99th percentile."""
    sigma = np.log(p99_s / median_s) / 2.3263478740408408       # z of the 99th percentile
    rng = np.random.default_rng(seed)
    out = np.round(rng.lognormal(np.log(median_s * fps), sigma, n)).astype(np.int64)
    return np.maximum(out, 1)


def missing_fraction(tracking):
    """Fraction of in-half frames with no ball -- what degrade's ball_missing targets."""
    rows = tracking.half() > 0
    return float((~tracking.ball_present[rows]).mean())


def nested_ladder(degraded, labels, rates, run_lengths, rng, params=None):
    """{rate: Tracking} with the ball missing on `rate` of the in-half frames, masks nested.

    `degraded` should already sit at (or below) the lowest rate. Rates are visited from low to
    high; each erases more ball on top of the previous one, with degrade.erase_ball -- the same
    run lengths (capped at max_gap_s) and dead-ball weighting degrade uses -- so the only thing
    that changes from one rate to the next is which ball frames are left.
    """
    p = degrade._settings(params)
    fps = degraded.meta.get("fps", 25)
    lengths = run_lengths[run_lengths <= p["max_gap_s"] * fps]
    rows = np.flatnonzero(degraded.half() > 0)
    dead = labels[rows] == 2
    out, current = {}, degraded
    for rate in sorted(rates):
        erase = np.zeros(current.n_frames, dtype=bool)
        erase[rows] = degrade.erase_ball(current.ball_present[rows], dead, lengths, rate,
                                         p["dead_weight"], rng)
        current = current.without_ball(erase) if erase.any() else current
        out[rate] = current
    return out


def summarise(values):
    """(mean, 5th, 95th percentile) of a list of scores."""
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(np.percentile(values, 5)), float(np.percentile(values, 95))


def slope_per_point(scores, reference, target):
    """Change in score per +1 point of coverage going from `reference` to `target` missing rate.

    scores: {rate: score}. Coverage = 1 - missing, so going from 41% to 37% missing is +4 points.
    """
    points = 100.0 * (reference - target)
    return (scores[target] - scores[reference]) / points


def load_configs(names):
    return {name: harness.load_config(os.path.join(paths.CONFIGS_DIR, name + ".yaml"))
            for name in names}


def run(matches, seeds, rates, configs, run_lengths, mode, min_possession_s, params=None,
        verbose=True):
    """[{match, seed, rate, achieved, config, spell, macro, poss, gt_poss}] one per unit."""
    rows = []
    for index, name in enumerate(matches):
        tracking, labels = external.load(name)
        fps = tracking.meta.get("fps", 25)
        graded = np.where(tracking.half() > 0, labels, pm.X).astype(np.int8)
        gt = external.align_labels(graded, min_possession_s, fps)
        if verbose:
            print("{}: {} frames, clean ball missing on {:.1%} of in-half frames".format(
                name, tracking.n_frames, missing_fraction(tracking)), flush=True)
        for seed in seeds:
            unit_seed = 1000 * seed + index                     # as harness.bench_external
            if mode == "nested":
                base = degrade.degrade(tracking, labels, unit_seed, run_lengths,
                                       dict(params or {}, ball_missing=min(rates)))
                ladder = nested_ladder(base, labels, rates, run_lengths,
                                       np.random.default_rng([unit_seed, 7]), params)
            else:
                ladder = {rate: degrade.degrade(tracking, labels, unit_seed, run_lengths,
                                                dict(params or {}, ball_missing=rate))
                          for rate in rates}
            for rate in rates:
                degraded = ladder[rate]
                achieved = missing_fraction(degraded)
                for cname, cfg in configs.items():
                    started = time.time()
                    pred = gt_io.labels_to_codes(pipeline.predict(degraded, cfg["pipeline"]))
                    score = harness._scores(gt, pred)
                    rows.append(dict(match=name, seed=seed, rate=rate, achieved=achieved,
                                     config=cname, **score))
                    if verbose:
                        print("  {} seed {} missing {:.0%} (got {:.1%}) {:<9} spellF1 {:.4f} "
                              "macroF1 {:.4f}  ({:.0f} s)".format(
                                  name, seed, rate, achieved, cname, score["spell"],
                                  score["macro"], time.time() - started), flush=True)
    return rows


def table(rows, rates, configs, reference=REFERENCE, slopes_to=SLOPES_TO):
    """The report as text: per rate and config, mean [5%, 95%] over units; then the slopes."""
    lines = ["{:>9} {:>9}  ".format("missing", "achieved") + "  ".join(
        "{:<26} {:<8}".format(c + " spellF1 [5,95]", "macroF1") for c in configs)]
    by = {}
    for r in rows:
        by.setdefault((r["config"], r["rate"]), []).append(r)
    for rate in sorted(rates, reverse=True):
        achieved = np.mean([r["achieved"] for r in by[(configs[0], rate)]])
        cells = []
        for c in configs:
            mean, lo, hi = summarise([r["spell"] for r in by[(c, rate)]])
            macro = np.mean([r["macro"] for r in by[(c, rate)]])
            cells.append("{:.4f} [{:.4f}, {:.4f}]   {:.4f}  ".format(mean, lo, hi, macro))
        lines.append("{:>8.0f}% {:>8.1f}%  ".format(100 * rate, 100 * achieved) + "  ".join(cells))

    lines.append("")
    lines.append("slope = change per +1 point of coverage (mean of the curve; paired per-unit "
                 "5-95% in brackets)")
    for c in configs:
        for metric in ("spell", "macro"):
            for target in slopes_to:
                curve = {rate: np.mean([r[metric] for r in by[(c, rate)]]) for rate in rates}
                units = {}
                for r in rows:
                    if r["config"] == c:
                        units.setdefault((r["match"], r["seed"]), {})[r["rate"]] = r[metric]
                paired = [slope_per_point(u, reference, target) for u in units.values()]
                _, lo, hi = summarise(paired)
                lines.append("  {:<9} {}F1  {:.0f}%->{:.0f}%: {:+.5f} [{:+.5f}, {:+.5f}]".format(
                    c, metric, 100 * reference, 100 * target, slope_per_point(curve, reference,
                                                                            target), lo, hi))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rates", type=float, nargs="+", default=list(RATES))
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--mode", choices=("nested", "independent"), default="nested")
    parser.add_argument("--run-lengths", help=".npy of ATD no-ball run lengths in frames")
    parser.add_argument("--degrade", default="{}", help="other degrade params (YAML)")
    parser.add_argument("--min-possession-s", type=float,
                        default=harness.DEFAULTS["validation"]["external"]["min_possession_s"])
    args = parser.parse_args()
    if REFERENCE not in args.rates or not set(SLOPES_TO) <= set(args.rates):
        sys.exit("--rates must include {} and {}".format(REFERENCE, SLOPES_TO))

    matches = external.resolve("val")                  # never "test"
    missing = [m for m in external.SPLITS["val"] if m not in matches]
    if missing:
        print("val matches not present (scripts/prepare_external.py): {}".format(missing))
    if not matches:
        sys.exit("no val matches present")
    run_lengths = (np.load(args.run_lengths) if args.run_lengths
                   else synthetic_run_lengths())
    configs = load_configs(args.configs)
    started = time.time()
    rows = run(matches, args.seeds, args.rates, configs, run_lengths, args.mode,
               args.min_possession_s, yaml.safe_load(args.degrade) or {})
    print()
    print("val matches {}, seeds {}, mode {}, gaps {}".format(
        ", ".join(matches), args.seeds, args.mode,
        args.run_lengths or "log-normal median {} s p99 {} s".format(GAP_MEDIAN_S, GAP_P99_S)))
    print(table(rows, args.rates, list(configs)))
    print("\nruntime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
