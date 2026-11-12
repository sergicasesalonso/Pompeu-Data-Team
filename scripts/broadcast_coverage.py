"""B8: how much ball does the broadcast add where the tactical ATD has none?

    python scripts/broadcast_coverage.py

Aligns the broadcast ATD to the tactical one (pompeu.io.broadcast) and prints counts and
fractions only. The ball is not used for the alignment, so ball agreement where both recordings
see it is an independent check that the alignment is right.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu.io import broadcast  # noqa: E402
from pompeu.io.atd import load_atd  # noqa: E402
from pompeu.io.contract import Tracking, load_tracking  # noqa: E402
from starter import load_homography  # noqa: E402

# Params
GOOD_FIT_M = 2.0       # per-frame fit for an aligned frame to count
NEAR_M = 2.0           # ball agreement threshold for the sanity check


def main():
    started = time.time()
    tac = load_tracking("atd")
    bc = Tracking.from_atd(load_atd(source="broadcast"))
    _, _, valid = load_homography.load(source="broadcast")
    al = broadcast.align(tac, bc, valid)

    st = al.stats
    good = (al.b_of_t >= 0) & (al.fit_of_t <= GOOD_FIT_M)
    graded = tac.half() > 0
    pitch = tac.meta["pitch"]

    bball = bc.ball_xy.copy()
    if al.orientation[0]:
        bball[:, 0] = pitch[0] - bball[:, 0]
        bball[:, 1] = pitch[1] - bball[:, 1]
    b_ball_at_t = np.full((tac.n_frames, 2), np.nan)
    b_ball_at_t[good] = bball[al.b_of_t[good]]
    t_has = np.isfinite(tac.ball_xy).all(axis=1)
    b_has = np.isfinite(b_ball_at_t).all(axis=1)

    both = graded & good & t_has & b_has
    dist = np.linalg.norm(tac.ball_xy[both] - b_ball_at_t[both], axis=1)
    gap = graded & ~t_has
    gained = gap & b_has

    print("broadcast  {} frames, {} with valid homography".format(bc.n_frames, int(valid.sum())))
    print("segments   {} shots ({} frames), {} live, {} replays discarded, {} unmatched".format(
        st["segments"], st["segment_frames"], st["live_segments"], st["replays_discarded"],
        st["unmatched_segments"]))
    print("offsets    {} live offset(s); orientation flip={} swap_teams={}".format(
        st["offsets"], al.orientation[0], al.orientation[1]))
    print("aligned    {} tactical frames aligned, {} with fit <= {} m; median segment fit {:.2f} m,"
          " median frame fit {:.2f} m".format(
              st["aligned_tactical_frames"], int(good.sum()), GOOD_FIT_M,
              st["median_segment_fit_m"], float(np.nanmedian(al.fit_of_t[good]))))
    print("aligned    {:.1%} of graded frames have an aligned broadcast frame".format(
        (graded & good).sum() / graded.sum()))
    print("sanity     {} graded frames where both see the ball: median distance {:.2f} m, "
          "{:.1%} within {} m".format(int(both.sum()), float(np.median(dist)) if len(dist) else float("nan"),
                                     float((dist < NEAR_M).mean()) if len(dist) else float("nan"), NEAR_M))
    print("KEY        graded frames without tactical ball: {}; with ball in the aligned broadcast: "
          "{} ({:.1%})".format(int(gap.sum()), int(gained.sum()), gained.sum() / max(gap.sum(), 1)))
    print("KEY        graded frames with a ball: tactical {:.1%} -> tactical + broadcast {:.1%}".format(
        (graded & t_has).sum() / graded.sum(), (graded & (t_has | b_has)).sum() / graded.sum()))
    print("time       {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
