"""Re-run scripts/fit_hsmm.py's coordinate descent from hsmm_v2_fit with wider grids at its edges.

    python scripts/experiments/fit_hsmm_wide.py        # val, seeds 0-1, writes configs/hsmm_v2_fit2.yaml if it moves

hsmm_v2_fit landed on the edge of fit_hsmm.GRID for w_dead (2.0, max), poss_median_s (12.0, max)
and p_poss_to_dead (0.4, max). This script extends those three grids and keeps the others as in
fit_hsmm.GRID. Same objective (spellF1 over all val matches x seeds), same preparation
(fit_hsmm.prepared_val: seeds 0-1, ATD no-ball run lengths), same acceptance (> best + 1e-4),
two passes. Only `val`; the test matches are never touched. Nothing is printed per frame.
"""

import argparse
import os
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pompeu import external, paths  # noqa: E402
import fit_hsmm  # noqa: E402

GRID = dict(fit_hsmm.GRID)
GRID.update({
    "w_dead": [1.0, 2.0, 3.0, 4.0, 6.0],
    "poss_median_s": [8.0, 12.0, 16.0, 20.0, 25.0],
    "p_poss_to_dead": [0.3, 0.4, 0.5, 0.6],
})
START = os.path.join(paths.CONFIGS_DIR, "hsmm_v2_fit.yaml")
OUT = os.path.join(paths.CONFIGS_DIR, "hsmm_v2_fit2.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    started_all = time.time()

    with open(START, encoding="utf-8") as handle:
        start_cfg = yaml.safe_load(handle)
    start = dict(start_cfg["pipeline"]["params"]["hsmm"])
    deadball_params = start_cfg["pipeline"]["params"].get("deadball") or {}

    preps, truths = fit_hsmm.prepared_val(args.seeds, deadball_params)
    current = dict(start)
    best, best_macro = fit_hsmm.score(preps, truths, current)
    start_spell = best
    print("start (hsmm_v2_fit) spellF1 {:.4f} macroF1 {:.4f}".format(best, best_macro), flush=True)
    for npass in range(args.passes):
        for name, values in GRID.items():
            for value in values:
                trial = dict(current, **{name: value})
                if trial == current:
                    continue
                started = time.time()
                spell, macro = fit_hsmm.score(preps, truths, trial)
                print("  {}={} spellF1 {:.4f} ({:.0f} s)".format(name, value, spell,
                                                                time.time() - started), flush=True)
                if spell > best + 1e-4:
                    best, best_macro, current = spell, macro, trial
            print("pass {} {} -> {}  spellF1 {:.4f}".format(npass + 1, name, current.get(name),
                                                           best), flush=True)

    moved = {k: (start.get(k), v) for k, v in current.items() if start.get(k) != v}
    print("\nmoved: {}".format(moved or "nothing"))
    print("val spellF1 {:.4f} -> {:.4f} (gain {:+.4f}), macroF1 {:.4f}".format(
        start_spell, best, best - start_spell, best_macro))
    edges = {k: v for k, v in current.items() if k in GRID and v in (GRID[k][0], GRID[k][-1])}
    print("still on a grid edge: {}".format(edges or "none"))
    if moved:
        cfg = {"name": os.path.splitext(os.path.basename(args.out))[0], "tracking": "atd",
               "pipeline": {"method": "hsmm_v1",
                            "params": {"hsmm": current, "deadball": deadball_params},
                            "clean": {}}}
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                "# Refitted by scripts/experiments/fit_hsmm_wide.py from hsmm_v2_fit with wider grids\n"
                "# for w_dead, poss_median_s, p_poss_to_dead, on the val external matches ({}),\n"
                "# seeds {}. val spellF1 {:.4f} (hsmm_v2_fit {:.4f}), macroF1 {:.4f}. Test never used.\n"
                .format(", ".join(external.resolve("val")), args.seeds, best, start_spell,
                        best_macro))
            yaml.safe_dump(cfg, handle, sort_keys=False)
        print("wrote", args.out)
    print("runtime {:.0f} s".format(time.time() - started_all))


if __name__ == "__main__":
    main()
