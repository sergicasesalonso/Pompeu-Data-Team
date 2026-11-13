"""Does the degradation simulator make external matches look like our ATD? (plan, stage A6 step 3)

    python scripts/degradation_fit.py                    # every external match present, seed 0
    python scripts/degradation_fit.py --matches dfl:J03WPY metrica:2

Ten statistics, computed the same way on BAR-ATH's ATD (graded frames) and on each degraded
external match. Distributions are compared with the two-sample Kolmogorov-Smirnov statistic D
(the largest gap between the two CDFs; p-values are meaningless at these sample sizes, every
difference is "significant"), scalar rates with their ratio. Overlaid histograms go to
results/degradation_fit/, the table to results/degradation_fit.md. Only summary statistics: safe
to commit.
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import external, paths  # noqa: E402
from pompeu.clean import players as clean_players  # noqa: E402
from pompeu.external import degrade  # noqa: E402
from pompeu.features.kinematics import runs  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.validate.stress import missing_run_lengths  # noqa: E402

OUT_MD = os.path.join(paths.RESULTS_DIR, "degradation_fit.md")
OUT_DIR = os.path.join(paths.RESULTS_DIR, "degradation_fit")
# A KS D below this is "the two look alike" for our purposes: no method here is sensitive to a
# few percent of probability mass moving.
KS_OK = 0.15


def statistics(tracking):
    """{name: array (a distribution) or float (a rate)} on the in-half frames of a Tracking."""
    fps = tracking.meta.get("fps", 25)
    graded = tracking.half() > 0
    ball = tracking.ball_present
    out = {}

    gaps = []
    for _, first, last in tracking.half_blocks():
        lo, hi = first - tracking.frame0, last + 1 - tracking.frame0
        starts, stops = runs(~ball[lo:hi])
        inside = (starts > 0) & (stops < hi - lo)
        gaps += ((stops - starts)[inside] / fps).tolist()
    out["ball gap length (s)"] = np.array(gaps)
    out["ball missing (share)"] = float(1 - ball[graded].mean())

    xy = tracking.ball_xy
    step = np.hypot(*(xy[1:] - xy[:-1]).T) * fps
    both = graded[1:] & ball[1:] & ball[:-1]
    out["ball step speed (m/s)"] = np.clip(step[both], 0, 60)
    out["ball jumps > 40 m/s per 1000 frames"] = float(np.sum(step[both] > 40) / both.sum() * 1000)

    team = tracking.players_team[0]
    present = tracking.players_present
    out["players visible per team"] = np.concatenate([present[graded][:, team == s].sum(axis=1)
                                                      for s in (0, 1)]).astype(float)
    lengths = []
    for slot in range(tracking.n_slots):
        starts, stops = runs(present[:, slot] & graded)
        lengths += ((stops - starts) / fps).tolist()
    out["tracklet length (s)"] = np.clip(np.array(lengths), 0, 300)
    out["slots used"] = float(np.sum(present[graded].any(axis=0)))

    frozen = clean_players.frozen_mask(tracking.players_xy, fps)
    out["frozen share of player-frames (A1)"] = float(frozen[graded].sum() / max(1, present[graded].sum()))
    _, stretches = clean_players.duplicate_mask(tracking.players_xy, team, fps)
    out["duplicate stretches per hour (A1)"] = float(stretches / (graded.sum() / fps / 3600))

    speed = clean_players.field_speed(tracking.players_xy, fps)
    out["median player speed (m/s)"] = np.clip(speed[graded & ~np.isnan(speed)], 0, 10)
    from pompeu.events.touches import detect_events
    from pompeu.features.build import build_features
    from pompeu.clean import clean_tracking
    clean, _ = clean_tracking(tracking, {})
    F = build_features(clean, with_windows=False,
                       groups=("ball", "proximity", "pitch_control", "collective", "context"))
    events = detect_events(clean, F)
    minutes = graded.sum() / fps / 60
    out["A4 touches per minute"] = float(np.sum(events["kind"] == "touch") / minutes)
    out["A4 controls per minute"] = float(np.sum(events["kind"] == "control") / minutes)
    d = np.hypot(tracking.players_xy[:, :, 0] - xy[:, None, 0],
                 tracking.players_xy[:, :, 1] - xy[:, None, 1])
    nearest = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    out["nearest player to ball (m)"] = np.clip(nearest[graded & ball & np.isfinite(nearest)], 0, 40)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = external.resolve(args.matches[0] if len(args.matches) == 1 else args.matches)
    atd = load_tracking("atd")
    run_lengths = missing_run_lengths(atd)
    reference = statistics(atd)
    simulated = {}
    for name in names:
        tracking, labels = external.load(name)
        simulated[name] = statistics(degrade.degrade(tracking, labels, args.seed, run_lengths))
        print("done", name)

    os.makedirs(OUT_DIR, exist_ok=True)
    header = "| statistic | ATD | " + " | ".join(names) + " |"
    lines = ["# Simulador de degradación frente al ATD (A6)", "",
             "`python scripts/degradation_fit.py` — semilla {}. Distribuciones: estadístico KS D "
             "(0 = idénticas; ✓ si < {}). Tasas: valor (ATD) y valor simulado.".format(
                 args.seed, KS_OK), "", header, "|---|---|" + "---|" * len(names)]
    for stat, value in reference.items():
        if isinstance(value, float):
            row = "| {} | {:.3f} | ".format(stat, value) + " | ".join(
                "{:.3f}".format(simulated[n][stat]) for n in names) + " |"
        else:
            cells = []
            for n in names:
                d = ks_2samp(value, simulated[n][stat]).statistic
                cells.append("D = {:.3f} {}".format(d, "✓" if d < KS_OK else "✗"))
            row = "| {} | median {:.2f} | ".format(stat, float(np.median(value))) + \
                " | ".join(cells) + " |"
            figure, ax = plt.subplots(figsize=(6, 3.5))
            bins = np.linspace(min(np.min(value), min(np.min(simulated[n][stat]) for n in names)),
                               max(np.percentile(value, 99), max(np.percentile(simulated[n][stat], 99)
                                                                 for n in names)), 60)
            ax.hist(value, bins=bins, density=True, histtype="step", linewidth=2, label="ATD")
            for n in names:
                ax.hist(simulated[n][stat], bins=bins, density=True, histtype="step", label=n)
            ax.set_title(stat)
            ax.legend(fontsize=7)
            figure.tight_layout()
            safe = "".join(c if c.isalnum() else "_" for c in stat).strip("_")
            figure.savefig(os.path.join(OUT_DIR, safe + ".png"), dpi=90)
            plt.close(figure)
        lines.append(row)
    lines += ["", "Gráficas: `results/degradation_fit/*.png`.", ""]
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print("written to", OUT_MD)


if __name__ == "__main__":
    main()
