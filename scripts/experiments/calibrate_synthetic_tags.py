"""Calibrate the synthetic-tag generator's free rates on external TRAIN labels.

    python scripts/experiments/calibrate_synthetic_tags.py   # adds "generator" to the JSON

The measured block of configs/synthetic_tags_v1.json (scripts/experiments/tag_error_model.py)
holds the moments of BAR-ATH's real Smart Tagging against our confident reference. Here the
generator (pompeu.external.synthetic_tags) runs on the DFL `train` matches (val and test never
touched) and synthetic_tags.tag_statistics measures the SAME moments of the synthetic tags
against those matches' labels (aligned at 2 s; live = H/A with the ball, dead = D; spells >= 2 s
with >= 50% ball). A small grid over the rates that cannot be read off directly (random miss,
wrong team, false positives) and the shape knobs whose BAR-ATH measurement is inflated by the
reference's own errors (offset spread, minimum spell, cutting the last phase) picks the closest match; the shapes (offsets, durations, code mix,
pairing, set pieces) are copied from the measurement. Only aggregates are printed.
"""

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pompeu import external, paths  # noqa: E402
from pompeu.external import synthetic_tags as st  # noqa: E402

# Params
PATH = os.path.join(paths.CONFIGS_DIR, "synthetic_tags_v1.json")
SEEDS = (0, 1, 2)
GRID = {"offset_scale": (0.25, 0.5, 1.0),     # x the measured IQRs of the start/end offsets
        "truncate_last": (False, True),
        "min_spell_s": (0.0, 3.0),
        "missing_rate": (0.0, 0.1, 0.2, 0.3),
        "wrong_team_rate": (0.0, 0.05, 0.1, 0.15),
        "false_rate": (0.0, 0.03)}
# moment: scale (a difference of one scale costs 1)
MOMENTS = {"missing_rate": 0.05, "frame_agreement": 0.03, "coverage_live": 0.05,
           "coverage_dead": 0.05, "wrong_team_rate": 0.05, "false_dead_rate": 0.02,
           "tag_spells_per_spell": 0.1}


def shapes(measured):
    """Generator parameters copied from the BAR-ATH measurement."""
    per = measured["per_code"]
    sp = measured["spells"]
    iqr = lambda d: round(d["q75"] - d["q25"], 3)  # noqa: E731
    attack = [c for c in st.DEFAULTS["attack_mix"] if c in per]
    sps = measured["set_pieces"]
    types = {k: v for k, v in measured["set_piece_types"].items()
             if k in ("throw-in", "corner kick", "goal kick", "free kick")}
    return {
        "start_offset_s": {"median": sp["start_offset_s"]["median"], "iqr": iqr(sp["start_offset_s"])},
        "end_offset_s": {"median": sp["end_offset_s"]["median"], "iqr": iqr(sp["end_offset_s"])},
        "attack_mix": {c: per[c]["n"] for c in attack},
        "phase_duration_s": {c: {"median": per[c]["duration_s"]["median"],
                                 "iqr": iqr(per[c]["duration_s"])} for c in attack},
        "pairing_rate": measured["pairing"]["with_opposite_defend"],
        "set_piece_rate": sps["set_piece_rate"],
        "set_piece_offset_s": {"median": sps["start_minus_dead_end_s"]["median"],
                               "iqr": iqr(sps["start_minus_dead_end_s"])},
        "set_piece_team_na": per["SET PIECES"]["team_na"],
        "set_piece_duration_s": {"median": per["SET PIECES"]["duration_s"]["median"],
                                 "iqr": iqr(per["SET PIECES"]["duration_s"])},
        "set_piece_types": types,
    }


def reference(name):
    tracking, labels = external.load(name)
    fps = tracking.meta.get("fps", 25)
    half = tracking.half()
    graded = np.where(half > 0, labels, 3).astype(np.int8)
    lab = external.align_labels(graded, 2.0, fps)
    ball = np.asarray(tracking.ball_present, bool)
    team = np.where(lab == 0, 0, np.where(lab == 1, 1, -1))
    live = (team >= 0) & ball
    dead = lab == 2
    usable = half > 0
    spells = [(a, b, t) for a, b, t in st.spells(lab, half, 2.0, fps) if ball[a:b].mean() >= 0.5]
    return dict(lab=lab, half=half, team=team, live=live, dead=dead, usable=usable,
                spells=spells, fps=fps)


def resolve(base, knobs):
    """Generator parameters: base with the grid knobs applied (offset_scale scales the IQRs)."""
    model = json.loads(json.dumps(base))
    for k, v in knobs.items():
        if k == "offset_scale":
            for key in ("start_offset_s", "end_offset_s"):
                model[key]["iqr"] = round(base[key]["iqr"] * v, 3)
        else:
            model[k] = v
    return model


def moments(refs, model, seeds):
    acc = {k: [] for k in MOMENTS}
    for i, r in enumerate(refs):
        for seed in seeds:
            tags = st.generate(r["lab"], r["half"], model, seed=1000 * seed + i, fps=r["fps"])
            s = st.tag_statistics(tags, r["team"], r["live"], r["dead"], r["usable"], r["spells"],
                                  r["fps"])
            for k in MOMENTS:
                acc[k].append(s[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    with open(PATH, encoding="utf-8") as handle:
        doc = json.load(handle)
    measured = doc["measured"]
    target = dict(measured["spells"])
    names = [external.canonical(m) for m in external.SPLITS["train"] if m.startswith("dfl:")]
    refs = [reference(m) for m in names]
    base = st.load_error_model(None)
    base.update(shapes(measured))
    print("train matches {}; target {}".format(
        names, {k: round(target[k], 4) for k in MOMENTS}))
    best = None
    for values in itertools.product(*GRID.values()):
        model = resolve(base, dict(zip(GRID, values)))
        got = moments(refs, model, SEEDS[:1])
        loss = sum(((got[k] - target[k]) / s) ** 2 for k, s in MOMENTS.items())
        if best is None or loss < best[0]:
            best = (loss, dict(zip(GRID, values)), got)
    model = resolve(base, best[1])
    got = moments(refs, model, SEEDS)
    loss = sum(((got[k] - target[k]) / s) ** 2 for k, s in MOMENTS.items())
    print("chosen {}  loss (seeds {}) {:.2f}".format(best[1], list(SEEDS), loss))
    for k in MOMENTS:
        print("  {:<22} real {:.4f}  synthetic {:.4f}".format(k, target[k], got[k]))
    doc["generator"] = {k: model[k] for k in sorted(model)}
    doc["calibration"] = {"matches": names, "seeds": list(SEEDS), "grid": GRID,
                          "moment_scales": MOMENTS, "loss": round(loss, 4),
                          "synthetic_moments": {k: round(v, 4) for k, v in got.items()},
                          "real_moments": {k: round(float(target[k]), 4) for k in MOMENTS}}
    with open(PATH, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(doc, handle, indent=1)
        handle.write("\n")
    print("wrote", os.path.relpath(PATH, ROOT))


if __name__ == "__main__":
    main()
