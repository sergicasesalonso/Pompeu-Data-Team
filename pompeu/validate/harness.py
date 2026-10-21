"""The validation harness: one config in, four benches out, one row in results/leaderboard.csv.

    public_clean       the public window as provided. Checks the pipeline is right; never chooses.
    public_stress      the public window with the ball erased to 30/50/70% (stress.py), 20 seeds
                       per level. The bench closest to what the full match looks like.
    external           degraded external matches (A6). THE bench that decides. Pending until A6.
    full_match_sanity  no ground truth: possession count, mean duration, %D, home share, and
                       whether they sit in the expected ranges.

Every score comes from possession_metrics.evaluate -- the official code -- and every interval from
bootstrap.py, which reuses it. The acceptance rule (stage 0.4): a change goes in only if it
improves external AND public_stress with non-overlapping intervals, or clearly improves both
while worsening neither. Improving public_clean alone is not evidence of anything.
"""

import copy
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import yaml

from pompeu import paths, pipeline
from pompeu.io import gt as gt_io
from pompeu.io.contract import load_tracking
from pompeu.validate import bootstrap, stress
import games
import possession_metrics as pm

# Params
DEFAULTS = {
    "tracking": "atd",
    "validation": {
        "bootstrap_n": 1000,
        "stress": {"levels": list(stress.DEFAULT_LEVELS), "seeds": stress.DEFAULT_SEEDS,
                   "max_gap_s": stress.DEFAULT_MAX_GAP_S,
                   # Seconds of untouched tracking either side of the window, so methods with a
                   # temporal context see the same neighbourhood as on the full match.
                   "context_s": 60},
        "external": {"datasets": []},
        # Expected on a real full match (guía, 0.4). Outside these is a red flag, not a score.
        "sanity": {"possessions": [300, 400], "dead_pct": [35.0, 45.0]},
    },
}
LEADERBOARD = os.path.join(paths.RESULTS_DIR, "leaderboard.csv")
RUNS_DIR = os.path.join(paths.RESULTS_DIR, "runs")
SUBMISSIONS_DIR = os.path.join(paths.RESULTS_DIR, "submissions")


# -- config ------------------------------------------------------------------------------------

def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(
            out.get(key), dict) else value
    return out


def load_config(path):
    """A YAML config merged over DEFAULTS. `name` defaults to the file name."""
    with open(path, encoding="utf-8") as handle:
        cfg = _merge(DEFAULTS, yaml.safe_load(handle) or {})
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    if "pipeline" not in cfg or "method" not in cfg["pipeline"]:
        raise ValueError("{}: config needs pipeline.method".format(path))
    return cfg


def git_version():
    """'<short hash>' or '<short hash>+dirty' (tracked files modified), 'nogit' outside a repo."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=paths.ROOT, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"
    return head + ("+dirty" if dirty else "")


# -- benches -----------------------------------------------------------------------------------

def _scores(gt_codes, pred_codes):
    result = pm.evaluate(gt_codes, pred_codes)
    return {"spell": result["spell_f1_mean"], "macro": result["macro_f1"],
            "poss": result["n_pred_spells"], "gt_poss": result["n_gt_spells"]}


def bench_public_clean(gt_codes, full_codes, n_draws):
    out = _scores(gt_codes, full_codes)
    ci = bootstrap.intervals(gt_codes, full_codes, n_draws)
    out.update(spell_lo=ci["spell"][0], spell_hi=ci["spell"][1],
               macro_lo=ci["macro"][0], macro_hi=ci["macro"][1])
    return out


def bench_public_stress(cfg, tracking, gt_codes, n_draws, verbose=False):
    """{level: scores averaged over seeds, with the two-level bootstrap interval}."""
    settings = cfg["validation"]["stress"]
    runs = pm.valid_runs(gt_codes)
    if len(runs) != 1:
        raise ValueError("public_stress expects one contiguous public window, got {}".format(runs))
    window = runs[0]
    context = int(settings["context_s"] * tracking.meta.get("fps", 25))
    crop = tracking.crop(window[0] - context, window[1] + context)

    by_level = {}
    for level, seed, achieved, stressed in stress.stressed_trackings(
            crop, window, settings["levels"], settings["seeds"], settings["max_gap_s"],
            run_lengths=stress.missing_run_lengths(tracking)):
        codes = np.full(len(gt_codes), pm.X, dtype=np.int8)
        codes[crop.frame0:crop.frame0 + crop.n_frames] = gt_io.labels_to_codes(
            pipeline.predict(stressed, cfg["pipeline"]))
        entry = by_level.setdefault(level, {"preds": [], "scores": [], "achieved": []})
        entry["preds"].append(codes)
        entry["scores"].append(_scores(gt_codes, codes))
        entry["achieved"].append(achieved)

    out = {}
    for level, entry in by_level.items():
        ci = bootstrap.intervals(gt_codes, entry["preds"], n_draws)
        out[level] = {
            "spell": float(np.mean([s["spell"] for s in entry["scores"]])),
            "macro": float(np.mean([s["macro"] for s in entry["scores"]])),
            "poss": float(np.mean([s["poss"] for s in entry["scores"]])),
            "spell_lo": ci["spell"][0], "spell_hi": ci["spell"][1],
            "macro_lo": ci["macro"][0], "macro_hi": ci["macro"][1],
            "missing": float(np.mean(entry["achieved"])),
            "seeds": len(entry["preds"]),
        }
        if verbose:
            print("  stress {:.0%}: spellF1 {:.4f}".format(level, out[level]["spell"]))
    return out


def bench_external(cfg):
    """Placeholder until A6 registers degraded external matches in validation.external."""
    if not cfg["validation"]["external"]["datasets"]:
        return {"status": "pending: no external datasets yet (stage A6)"}
    raise NotImplementedError("external bench arrives with A6 (pompeu/external)")


def bench_full_match_sanity(cfg, full_codes, game=games.GAME):
    runs = [(first, last + 1) for _, first, last in games.half_video_blocks(game)]
    graded = np.concatenate([full_codes[a:b] for a, b in runs])
    filled = pm.fill_abstentions(full_codes, runs)
    stats = pm.possession_stats(filled, runs)
    expected = cfg["validation"]["sanity"]
    dead_pct = 100.0 * float(np.mean(graded == pm.D))
    out = {
        "poss": stats["n_spells"],
        "dur_s": stats["mean_duration_s"],
        "dead_pct": dead_pct,
        "home_share": stats["home_share"],
        "x_on_graded": int(np.sum(graded == pm.X)),
    }
    out["ok"] = bool(expected["possessions"][0] <= out["poss"] <= expected["possessions"][1]
                     and expected["dead_pct"][0] <= dead_pct <= expected["dead_pct"][1]
                     and out["x_on_graded"] == 0)
    return out


# -- run, record -------------------------------------------------------------------------------

def run(cfg, tracking=None, gt_path=paths.PUBLIC_GT, save=True, verbose=True):
    """Run every bench for one config. Returns (result dict, flat leaderboard row)."""
    started = time.time()
    if tracking is None:
        tracking = load_tracking(cfg["tracking"])
    if tracking.frame0 != 0 or tracking.n_frames != games.GAMES[games.GAME]["n_video_frames"]:
        raise ValueError("the harness needs a full-match Tracking")
    gt_codes = gt_io.load_gt_codes(gt_path)
    n_draws = int(cfg["validation"]["bootstrap_n"])

    full_labels = pipeline.predict(tracking, cfg["pipeline"])
    submission = pd.DataFrame({"frame": np.arange(len(full_labels)), "label": full_labels})
    reason = pm.validate_submission(submission, len(gt_codes))
    if reason:
        raise ValueError("{}: full-match prediction is not a valid submission: {}".format(
            cfg["name"], reason))
    full_codes = gt_io.labels_to_codes(full_labels)

    result = {
        "config": cfg["name"],
        "config_hash": pipeline.config_hash(cfg),
        "git": git_version(),
        "tracking": cfg["tracking"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "public_clean": bench_public_clean(gt_codes, full_codes, n_draws),
        "public_stress": bench_public_stress(cfg, tracking, gt_codes, n_draws, verbose),
        "external": bench_external(cfg),
        "full_match_sanity": bench_full_match_sanity(cfg, full_codes),
    }
    result["runtime_s"] = round(time.time() - started, 1)
    row = flatten(result)

    if save:
        os.makedirs(RUNS_DIR, exist_ok=True)
        stamp = result["timestamp"].replace(":", "")
        with open(os.path.join(RUNS_DIR, "{}_{}.json".format(stamp, cfg["name"])), "w") as handle:
            json.dump({"result": result, "config": cfg}, handle, indent=2, default=str)
        os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
        submission.to_csv(os.path.join(SUBMISSIONS_DIR, "{}.csv".format(cfg["name"])),
                          index=False, lineterminator="\n")
        append_leaderboard(row)
    return result, row


def flatten(result):
    row = {key: result[key] for key in ("timestamp", "config", "config_hash", "git", "tracking")}
    for key in ("spell", "spell_lo", "spell_hi", "macro", "macro_lo", "macro_hi", "poss"):
        row["clean_" + key] = result["public_clean"][key]
    for level, scores in sorted(result["public_stress"].items()):
        prefix = "stress{:d}_".format(int(round(level * 100)))
        for key in ("spell", "spell_lo", "spell_hi", "macro", "macro_lo", "macro_hi", "poss"):
            row[prefix + key] = scores[key]
    external = result["external"]
    row["external_spell"] = external.get("spell")
    row["external_status"] = external.get("status", "ok")
    for key, value in result["full_match_sanity"].items():
        row["full_" + key] = value
    row["runtime_s"] = result["runtime_s"]
    return row


def append_leaderboard(row, path=LEADERBOARD):
    """Add one row, keeping every earlier one. New columns are added, never dropped."""
    new = pd.DataFrame([row])
    if os.path.isfile(path):
        new = pd.concat([pd.read_csv(path), new], ignore_index=True, sort=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")
    return path


def format_result(result):
    """The run as a short text table, for the terminal."""
    def interval(scores, key):
        low, high = scores.get(key + "_lo"), scores.get(key + "_hi")
        if low is None:
            return "{:.4f}".format(scores[key])
        return "{:.4f} [{:.3f}, {:.3f}]".format(scores[key], low, high)

    lines = ["{}  (config {}, git {}, tracking {}, {} s)".format(
        result["config"], result["config_hash"], result["git"], result["tracking"],
        result["runtime_s"]),
        "  {:<22} {:<24} {:<24} {:>6}".format("bench", "spellF1 [5%, 95%]",
                                               "macroF1 [5%, 95%]", "poss")]
    clean = result["public_clean"]
    lines.append("  {:<22} {:<24} {:<24} {:>6}".format(
        "public_clean", interval(clean, "spell"), interval(clean, "macro"),
        "{}/{}".format(clean["poss"], clean["gt_poss"])))
    for level, scores in sorted(result["public_stress"].items()):
        lines.append("  {:<22} {:<24} {:<24} {:>6.1f}".format(
            "public_stress {:.0%}".format(level), interval(scores, "spell"),
            interval(scores, "macro"), scores["poss"]))
    lines.append("  {:<22} {}".format("external", result["external"].get("status", "")))
    sanity = result["full_match_sanity"]
    lines.append("  {:<22} {} possessions, mean {:.1f} s, D {:.1f}%, home {:.1%}, X on graded {}"
                 "  -> {}".format("full_match_sanity", sanity["poss"], sanity["dur_s"],
                                  sanity["dead_pct"], sanity["home_share"],
                                  sanity["x_on_graded"], "in range" if sanity["ok"]
                                  else "OUT OF RANGE"))
    return "\n".join(lines)
