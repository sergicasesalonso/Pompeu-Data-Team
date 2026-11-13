"""How much more the ATD loses the ball when play is dead -- measured without any ground truth.

    python scripts/ball_missing_by_state.py

external/degrade.py makes a ball gap `dead_weight` times likelier to start on a dead-ball frame,
and the value (5) was read off the public 5-minute window (34 % missing on dead frames, ~6 % on
live ones). The public window may not be used to tune anything, so this measures the same thing
on the whole match with our own dead-ball detector (A3) instead of the labels: no GT is read.

Prints aggregates only: missing rate on dead and live graded frames, the rate at which no-ball runs
START on dead and live frames (what dead_weight actually scales), and their ratio.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu.events import deadball  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402


def run_starts(missing):
    """Boolean (N,): True where a no-ball run starts."""
    starts = missing.copy()
    starts[1:] &= ~missing[:-1]
    return starts


def main():
    tracking = load_tracking("atd")              # raw ATD: degrade.py imitates the raw gaps
    graded = tracking.half() > 0
    missing = ~np.isfinite(tracking.ball_xy).all(axis=1)
    dead = deadball.detect(tracking).dead.astype(bool)

    starts = run_starts(missing)
    live = ~dead
    rows = []
    for name, mask in (("dead (A3)", dead & graded), ("live (A3)", live & graded)):
        n = int(mask.sum())
        rows.append((name, n, missing[mask].mean(), starts[mask].sum() / n))
    for name, n, miss, start_rate in rows:
        print("{:<10} frames {:>7}  ball missing {:6.1%}  run starts per 1000 frames {:6.2f}".format(
            name, n, miss, 1000 * start_rate))
    print("ratio dead/live: missing {:.2f}  run starts {:.2f}".format(
        rows[0][2] / rows[1][2], rows[0][3] / rows[1][3]))
    print("(public-window values used by degrade.py: 34 % vs ~6 % missing -> dead_weight 5)")


if __name__ == "__main__":
    main()
