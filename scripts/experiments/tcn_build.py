"""Build the TCN's 5 Hz inputs (A2 features + synthetic-tag channels) and labels at 2/5/8 s.

    python scripts/experiments/tcn_build.py --split train --seeds 0 1 2 3 --workers 4
    python scripts/experiments/tcn_build.py --split val --seeds 10 11          # early stopping

train: every (match, seed) in four augmentations (plain, team swap, y-mirror, both).
val: plain only, degradation seed 1000*seed + index (as robust_val.py), tag seed TAG_SEED + that.
extra_train: SkillCorner (opt-in). Never `test`. Output: cache/tcn/*.npz (gitignored). Only
counts and timings are printed.
"""

import argparse
import os
import sys
import time
from multiprocessing import get_context

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pompeu import external, paths  # noqa: E402
from pompeu.experimental import tcn  # noqa: E402

_RUN_LENGTHS = None


def _init():
    global _RUN_LENGTHS
    from pompeu.io.contract import load_tracking
    from pompeu.validate.stress import missing_run_lengths
    _RUN_LENGTHS = missing_run_lengths(load_tracking("atd"))


def _one(job):
    name, seed, swap, mirror, dseed, tag = job
    path = tcn.item_path(name, seed, swap, mirror, tag)
    if os.path.isfile(path):
        return name, "cached", 0.0
    started = time.time()
    item = tcn.build_item(name, seed, swap, mirror, _RUN_LENGTHS, degrade_seed=dseed)
    tcn.save_item(item, path)
    return name, "built", time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("train", "val", "extra_train"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()
    raw = external.EXTRA_TRAIN if args.split == "extra_train" else external.SPLITS[args.split]
    names = [external.canonical(m) for m in raw]
    # Present = raw files found, or the parsed match already in cache/external/.
    names = [m for m in names if external.available(m) or os.path.isfile(os.path.join(
        paths.CACHE_DIR, "external", m.replace(":", "_") + "_v1.npz"))]
    jobs = []
    for index, name in enumerate(names):
        for seed in args.seeds:
            if args.split == "val":
                jobs.append((name, seed, False, False, 1000 * seed + index, args.tag))
                continue
            augs = [(False, False)] if args.no_aug else [(False, False), (True, False),
                                                         (False, True), (True, True)]
            for swap, mirror in augs:
                jobs.append((name, seed, swap, mirror, seed, args.tag))
    print("{} jobs over {} matches, {} workers".format(len(jobs), len(names), args.workers),
          flush=True)
    started = time.time()
    with get_context("spawn").Pool(args.workers, initializer=_init, maxtasksperchild=4) as pool:
        for i, (name, status, secs) in enumerate(pool.imap_unordered(_one, jobs), 1):
            print("  {}/{} {} {} {:.0f} s".format(i, len(jobs), name, status, secs), flush=True)
    print("done in {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
