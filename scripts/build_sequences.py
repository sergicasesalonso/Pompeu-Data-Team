"""Build the A7 training set: features + player tokens for every (match, seed, swap).

    python scripts/build_sequences.py                         # train 8 seeds x swap, val/test 2
    python scripts/build_sequences.py --split train --seeds 8 9 10 11

One cleaning per (match, seed, swap) feeds both caches (pompeu.external.examples): the A2
feature parquet and the 5 Hz token npz. Already cached pieces are skipped.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import external  # noqa: E402
from pompeu.external import examples  # noqa: E402
from pompeu.features.build import build_features  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.validate.stress import missing_run_lengths  # noqa: E402

PLAN = {"train": (range(8), (False, True)), "val": (range(2), (False,)),
        "test": (range(2), (False,))}


def build_one(name, seed, swap, run_lengths):
    feature_path = examples._path(name, seed, swap)
    token_path = examples._path(name, seed, swap, kind="tokens")
    if os.path.isfile(feature_path) and os.path.isfile(token_path):
        return "cached"
    clean, y = examples.cleaned(name, seed, swap, run_lengths)
    os.makedirs(examples.CACHE, exist_ok=True)
    if not os.path.isfile(feature_path):
        frame = build_features(clean, tagging=None)
        frame["__label__"] = y
        frame.to_parquet(feature_path + ".part", engine="pyarrow")
        os.replace(feature_path + ".part", feature_path)
    if not os.path.isfile(token_path):
        import numpy as np
        out = examples.player_tokens(clean)
        out["y"] = y[::examples.TOKEN_STEP]
        with open(token_path + ".part", "wb") as handle:
            np.savez_compressed(handle, **out)
        os.replace(token_path + ".part", token_path)
    return "built"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    run_lengths = missing_run_lengths(load_tracking("atd"))
    for split in (("train", "val", "test") if args.split == "all" else (args.split,)):
        seeds, swaps = PLAN[split]
        seeds = args.seeds if args.seeds is not None else seeds
        for name in external.resolve(split):
            for seed in seeds:
                for swap in swaps:
                    started = time.time()
                    status = build_one(name, seed, swap, run_lengths)
                    print("{:<6} {:<12} seed {} swap {}: {} ({:.0f} s)".format(
                        split, name, seed, swap, status, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
