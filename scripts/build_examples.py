"""Build (and cache) the training examples of every external match for some seeds.

    python scripts/build_examples.py                          # train x 4 seeds, val x 2, test x 2
    python scripts/build_examples.py --split train --seeds 0 1 2 3 4 5 6 7

See pompeu/external/examples.py. About a minute per (match, seed).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import external  # noqa: E402
from pompeu.external import examples  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.validate.stress import missing_run_lengths  # noqa: E402

DEFAULT_SEEDS = {"train": range(4), "val": range(2), "test": range(2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    run_lengths = missing_run_lengths(load_tracking("atd"))
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        seeds = args.seeds if args.seeds is not None else DEFAULT_SEEDS[split]
        for name in external.resolve(split):
            for seed in seeds:
                started = time.time()
                F, y = examples.load(name, seed, run_lengths)
                print("{:<6} {:<12} seed {}  {} x {}  ({:.0f} s)".format(
                    split, name, seed, F.shape[0], F.shape[1], time.time() - started), flush=True)


if __name__ == "__main__":
    main()
