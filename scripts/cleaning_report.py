"""Clean the full-match tracking and write what was removed, joined and filled to a Markdown report.

    python scripts/cleaning_report.py                                   # defaults
    python scripts/cleaning_report.py --config configs/x.yaml           # that config's clean block
    python scripts/cleaning_report.py --out results/cleaning_report.md

Only counts and summary statistics go in the report -- nothing that reconstructs the tracking --
so it is safe to commit.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import paths  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.features.kinematics import ball_kinematics, player_kinematics  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.models import baseline  # noqa: E402
from pompeu.validate import harness  # noqa: E402

OUT = os.path.join(paths.RESULTS_DIR, "cleaning_report.md")


def jumps(xy, fps, vmax=40.0):
    """Consecutive-frame ball steps faster than vmax."""
    step = np.hypot(*(xy[1:] - xy[:-1]).T) * fps
    with np.errstate(invalid="ignore"):
        return int(np.sum(step > vmax))


def players_per_team(tracking, graded):
    team = tracking.players_team[0]
    present = tracking.players_present[graded]
    return [float(present[:, team == side].sum(axis=1).mean()) for side in (0, 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="use this config's pipeline.clean block")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    clean_cfg = {}
    if args.config:
        clean_cfg = harness.load_config(args.config)["pipeline"].get("clean") or {}

    raw = load_tracking("atd")
    started = time.time()
    clean, report = clean_tracking(raw, clean_cfg)
    elapsed = time.time() - started
    fps = raw.meta["fps"]
    graded = raw.half() > 0

    ball_k = ball_kinematics(clean)
    player_k = player_kinematics(clean)
    changed = int(np.sum(baseline.predict(raw, 50)[graded] != baseline.predict(clean, 50)[graded]))

    b, fr, du, st = report["ball"], report["frozen"], report["duplicates"], report["stitching"]
    lines = [
        "# Informe de limpieza (A1)",
        "",
        "Generado con `python scripts/cleaning_report.py` en {:.1f} s. Configuración de limpieza: "
        "`{}` (vacío = valores por defecto de `pompeu/clean/__init__.py`).".format(
            elapsed, clean_cfg or {}),
        "",
        "## Balón",
        "",
        "| | |",
        "|---|---|",
        "| frames con balón antes | {:,} ({:.1f} %) |".format(
            b["present_before"], 100.0 * b["present_before"] / raw.n_frames),
        "| saltos > 40 m/s entre frames consecutivos, antes | {:,} |".format(
            jumps(raw.ball_xy, fps)),
        "| spikes eliminados | {:,} |".format(b["spikes_removed"]),
        "| puntos en fragmentos de < 3 frames eliminados | {:,} |".format(
            b["fragment_points_removed"]),
        "| frames interpolados (huecos < 0,5 s, spline cúbico) | {:,} |".format(
            b["interpolated_frames"]),
        "| frames con balón después | {:,} ({:.1f} %) |".format(
            b["present_after"], 100.0 * b["present_after"] / raw.n_frames),
        "| saltos > 40 m/s después | {:,} |".format(jumps(clean.ball_xy, fps)),
        "| velocidad del balón (Savitzky–Golay) p50 / p95 / p99.9 | {:.1f} / {:.1f} / {:.1f} m/s |"
        .format(*np.nanpercentile(ball_k["speed"], [50, 95, 99.9])),
        "",
        "## Jugadores",
        "",
        "| | |",
        "|---|---|",
        "| observaciones antes / después | {:,} / {:,} |".format(
            report["players"]["observations_before"], report["players"]["observations_after"]),
        "| frames congelados (< 0,2 m en 10 s con el resto moviéndose) | {:,}, acción `{}` |"
        .format(fr["frames"], fr["action"]),
        "| slots con más frames congelados | {} |".format(
            ", ".join("{} ({:,})".format(k, v) for k, v in fr["top_slots"].items())),
        "| tramos duplicados (< 0,5 m durante > 1 s) | {:,}, {:,} observaciones eliminadas |"
        .format(du["stretches"], du["frames_dropped"]),
        "| tracklets / uniones (stitching) / cadenas | {:,} / {:,} / {:,} |".format(
            st["tracklets"], st["links"], st["chains"]),
        "| slots usados antes / después | {} / {} |".format(st["slots_before"], st["slots_after"]),
        "| mediana tracklet / cadena | {:.1f} s / {:.1f} s |".format(
            st["median_tracklet_s"], st["median_chain_s"]),
        "| frames rellenados dentro de una cadena (huecos <= 1 s) | {:,} |".format(
            st["filled_frames"]),
        "| jugadores visibles por equipo en frames puntuados, antes | {:.2f} / {:.2f} |".format(
            *players_per_team(raw, graded)),
        "| jugadores visibles por equipo en frames puntuados, después | {:.2f} / {:.2f} |".format(
            *players_per_team(clean, graded)),
        "| velocidad de jugadores p50 / p99 | {:.1f} / {:.1f} m/s |".format(
            *np.nanpercentile(player_k["speed"], [50, 99])),
        "",
        "## Notas",
        "",
        "- Los slots 24 y 46 son los **porteros** (x ≈ 11 m / 94 m según la parte), no objetos "
        "fijos. Se congelan cuando salen de plano; por defecto esos frames se conservan con "
        "confianza 0,3 en vez de borrarse.",
        "- La columna de velocidad del ATD no se usa: viene vacía en el {:.0f} % de los frames."
        .format(100.0 * float(np.mean(np.isnan(load_speed_column(raw))))),
        "- Efecto sobre el baseline (ventana ±50) en frames puntuados: {:,} etiquetas cambian "
        "de {:,}.".format(changed, int(graded.sum())),
        "",
    ]
    folder = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(folder, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print("written to {}".format(args.out))


def load_speed_column(tracking):
    """The ATD's own ball speed column (only for the note on how empty it is)."""
    from pompeu.io.atd import load_atd
    return load_atd(game=tracking.meta["game"]).B[:, 2]


if __name__ == "__main__":
    main()
