"""Train A3's live/dead classifier on the external matches and measure it on the test matches.

    python scripts/train_deadball_gbm.py                 # train seeds 0-3, val seeds 0-1
    python scripts/train_deadball_gbm.py --step 4        # thinner (faster) training set

Writes artifacts/deadball_gbm_v<version>.txt and results/deadball_gbm_report.md: frame-level
F1 of live/dead on `test` for the classifier alone, A3's rules alone, and both together, all on
the same degraded test matches (seeds 0-1). The plan's bar for A3: live/dead F1 > 0.90 there.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import external, paths  # noqa: E402
from pompeu.external import examples  # noqa: E402
from pompeu.models import deadball_gbm  # noqa: E402

REPORT = os.path.join(paths.RESULTS_DIR, "deadball_gbm_report.md")


def f1(pred, truth):
    tp = np.sum(pred & truth)
    return 2.0 * tp / max(1, 2 * tp + np.sum(pred & ~truth) + np.sum(~pred & truth))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", type=int, default=2, help="keep every step-th frame")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--val-seeds", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()

    started = time.time()
    X, y, _ = examples.matrix(external.resolve("train"), args.train_seeds, args.step)
    X_val, y_val, _ = examples.matrix(external.resolve("val"), args.val_seeds, args.step)
    print("train {} x {}, val {} ({:.0f} s)".format(X.shape[0], X.shape[1], X_val.shape[0],
                                                    time.time() - started), flush=True)
    booster = deadball_gbm.train(X, y == 2, X_val, y_val == 2)
    os.makedirs(os.path.dirname(deadball_gbm.ARTIFACT), exist_ok=True)
    booster.save_model(deadball_gbm.ARTIFACT, num_iteration=booster.best_iteration)
    print("saved", deadball_gbm.ARTIFACT, flush=True)

    # Test: classifier alone, rules alone, together -- on the same degraded matches.
    from pompeu.events import deadball
    from pompeu.external import degrade
    from pompeu.clean import clean_tracking
    from pompeu.io.contract import load_tracking
    from pompeu.validate.stress import missing_run_lengths
    run_lengths = missing_run_lengths(load_tracking("atd"))
    model = deadball_gbm.load()
    rows = []
    totals = {k: [] for k in ("truth", "gbm", "rules", "both")}
    for name in external.resolve("test"):
        tracking, labels = external.load(name)
        for seed in args.val_seeds:
            F, y_test = examples.load(name, seed, run_lengths)
            graded = y_test != 3
            truth = y_test == 2
            p = deadball_gbm.predict(model, F)
            degraded = degrade.degrade(tracking, labels, seed, run_lengths)
            clean, _ = clean_tracking(degraded, {})
            rules_dead = deadball.detect(clean, tags=None).dead
            both_dead = deadball.detect(clean, tags=None, params={"gbm_weight": 1.0}).dead
            for key, value in (("truth", truth), ("gbm", p > 0.5), ("rules", rules_dead),
                               ("both", both_dead)):
                totals[key].append(value[graded])
            rows.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
                name, seed, f1(p[graded] > 0.5, truth[graded]), f1(rules_dead[graded], truth[graded]),
                f1(both_dead[graded], truth[graded])))
            print(rows[-1], flush=True)
    truth = np.concatenate(totals["truth"])
    summary = {k: f1(np.concatenate(totals[k]), truth) for k in ("gbm", "rules", "both")}

    importance = sorted(zip(booster.feature_name(), booster.feature_importance("gain")),
                        key=lambda kv: -kv[1])[:15]
    lines = ["# Clasificador vivo/muerto (A3, paso 3)", "",
             "`python scripts/train_deadball_gbm.py` — LightGBM entrenado con `train` ({} filas, "
             "{} rasgos), parada temprana con `val`, mejor iteración {}.".format(
                 X.shape[0], X.shape[1], booster.best_iteration), "",
             "F1 de **muerto** por frame en `test` (partidos degradados, semillas {}):".format(
                 args.val_seeds), "",
             "| partido | semilla | LightGBM solo | reglas A3 | A3 + LightGBM |", "|---|---|---|---|---|"]
    lines += rows
    lines += ["| **total** | | **{:.3f}** | **{:.3f}** | **{:.3f}** |".format(
        summary["gbm"], summary["rules"], summary["both"]), "",
              "Rasgos más importantes (ganancia): " + ", ".join(
                  "`{}`".format(name) for name, _ in importance), ""]
    with open(REPORT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print("report ->", REPORT)


if __name__ == "__main__":
    main()
