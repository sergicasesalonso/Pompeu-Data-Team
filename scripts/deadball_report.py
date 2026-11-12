"""Run the A3 dead-ball detector on the full match and write a summary to results/deadball_report.md.

    python scripts/deadball_report.py
    python scripts/deadball_report.py --config configs/deadball_vote50.yaml   # its clean/deadball

The report holds counts and summary statistics only. The per-restart list (frames, types, teams)
is derived match data, so it goes to cache/deadball_restarts.csv, which is gitignored.

The public-window numbers at the end CHECK the detector against the 5 public minutes -- four
dead stretches. They are there to catch a broken pipeline, never to tune a parameter on.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import paths  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.events import deadball  # noqa: E402
from pompeu.features.build import load_tagging_for  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.io.gt import load_gt_codes  # noqa: E402
from pompeu.validate import harness  # noqa: E402
import possession_metrics as pm  # noqa: E402

REPORT = os.path.join(paths.RESULTS_DIR, "deadball_report.md")
RESTARTS = os.path.join(paths.CACHE_DIR, "deadball_restarts.csv")


def f1(predicted, truth):
    tp = int(np.sum(predicted & truth))
    return 2.0 * tp / max(1, 2 * tp + int(np.sum(predicted & ~truth)) + int(np.sum(~predicted & truth)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="take tracking, clean and deadball params from it")
    parser.add_argument("--out", default=REPORT)
    args = parser.parse_args()

    tracking_spec, clean_cfg, params = "atd", {}, None
    if args.config:
        cfg = harness.load_config(args.config)
        tracking_spec = cfg["tracking"]
        clean_cfg = cfg["pipeline"].get("clean")
        params = (cfg["pipeline"].get("params") or {}).get("deadball")

    tracking = load_tracking(tracking_spec)
    if clean_cfg is not None:
        tracking, _ = clean_tracking(tracking, clean_cfg)
    started = time.time()
    result = deadball.detect(tracking, params=params)
    elapsed = time.time() - started
    fps = tracking.meta["fps"]
    graded = tracking.half() > 0

    restarts = result.restarts
    os.makedirs(os.path.dirname(RESTARTS), exist_ok=True)
    restarts.to_csv(RESTARTS, index=False)

    lengths = np.array([(b - a) / fps for a, b in result.segments])
    tagging = load_tagging_for(tracking)
    tagged = {} if tagging is None else \
        tagging[tagging["code"] == "SET PIECES"]["type"].value_counts().to_dict()
    tag_name = {"throw_in": "throw-in", "corner": "corner kick", "goal_kick": "goal kick",
                "free_kick": "free kick"}
    lines = [
        "# Detector de balón muerto (A3)",
        "",
        "`python scripts/deadball_report.py` — {:.1f} s. Parámetros: `pompeu/events/deadball.py` "
        "PARAMS{}.".format(elapsed, " con `{}`".format(params) if params else ""),
        "",
        "## Partido completo (sin GT)",
        "",
        "| | |",
        "|---|---|",
        "| frames puntuados en `D` | {:.1f} % (esperado 35–45 %) |".format(
            100 * result.dead[graded].mean()),
        "| tramos `D` | {} |".format(len(result.segments)),
        "| duración p10 / mediana / p90 | {:.1f} / {:.1f} / {:.1f} s |".format(
            *np.percentile(lengths, [10, 50, 90])),
        "| P(D) media en frames puntuados | {:.3f} |".format(result.p_dead[graded].mean()),
        "| reinicios con equipo | {:.0f} % |".format(100 * (restarts["team"] >= 0).mean()),
        "| confianza media del equipo | {:.2f} |".format(restarts["confidence"].mean()),
        "",
        "| tipo de reinicio | detectados | con salida del balón vista | Smart Tagging |",
        "|---|---|---|---|",
    ]
    for kind in deadball.TYPES:
        mine = restarts[restarts["type"] == kind]
        tag_count = (tagged.get(tag_name[kind], 0) if kind in tag_name else
                     tagged.get("kick off (after goal)", 0) + tagged.get("kick off (start)", 0))
        lines.append("| {} | {} | {} | {} |".format(kind, len(mine), int(mine["exit"].notna().sum()),
                                                   tag_count))
    lines += ["", "Fuentes del equipo: " + ", ".join(
        "`{}` {}".format(k, v) for k, v in restarts["source"].value_counts().items()), ""]

    gt = load_gt_codes()
    window = gt != pm.X
    truth = gt == pm.D
    gt_restarts = [a for a, _, k in pm.segments(gt, pm.valid_runs(gt)) if k != pm.D
                   and a > 0 and gt[a - 1] == pm.D]
    errors = []
    near = restarts["frame"].to_numpy()
    for frame in gt_restarts:
        offset = int(near[np.argmin(np.abs(near - frame))] - frame) if len(near) else None
        errors.append(offset if offset is not None and abs(offset) <= 5 * fps else "perdido")
    lo, hi = np.flatnonzero(window)[[0, -1]]
    dead_w = result.dead[window]
    lines += [
        "## Comprobación en la ventana pública (no sirve para elegir)",
        "",
        "| | |",
        "|---|---|",
        "| F1 de `D` / F1 de vivo | {:.3f} / {:.3f} |".format(
            f1(dead_w, truth[window]), f1(~dead_w, ~truth[window])),
        "| tramos `D` en la GT / detectados | {} / {} |".format(
            sum(1 for _, _, k in pm.segments(gt, pm.valid_runs(gt)) if k == pm.D),
            sum(1 for a, b in result.segments if min(b, hi + 1) - max(a, lo) >= fps)),
        "| error del reinicio (frames, detectado − GT; a más de 5 s = perdido) | {} |".format(
            ", ".join(str(e) for e in errors)),
        "",
    ]
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print("report -> {}\nrestarts -> {}".format(args.out, RESTARTS))


if __name__ == "__main__":
    main()
