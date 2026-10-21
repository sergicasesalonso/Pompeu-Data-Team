"""A1 normalisation: coordinates where a chosen team always attacks towards +x.

The ATD keeps the camera's frame, so the teams swap ends at halftime. Many features read more
simply as "towards the goal I attack": a defensive line height, a press, a pass forward. This turns
the pitch 180 degrees (x -> L - x, y -> W - y) on the frames where `team` attacks towards -x. A
rotation and not a mirror, so left and right wings stay where they are for the attacking team.

The flag is per half, from the ATD metadata (Barcelona attacks left to right in the 1st half).
Ungraded frames need a side too: everything before the 2nd half's first frame (pre-match, 1st half,
halftime) uses the 1st-half direction, everything from it on the 2nd-half direction.
"""

from dataclasses import replace

import numpy as np

from pompeu import paths  # noqa: F401
import games


def attack_flip(tracking, team):
    """(N,) bool: True where `team` (0 home, 1 away) attacks towards -x and must be rotated."""
    second_start = dict((half, first) for half, first, _ in games.half_video_blocks(
        tracking.meta.get("game", games.GAME)))[2]
    half = np.where(tracking.frames >= second_start, 2, 1)
    return np.where(half == 1, tracking.attack_sign(team, 1), tracking.attack_sign(team, 2)) < 0


def rotate(xy, flip, pitch):
    """Copy of xy (N, ..., 2) turned 180 degrees about the centre spot on the rows in `flip`."""
    out = np.array(xy, dtype=np.float64, copy=True)
    length, width = pitch
    out[flip, ..., 0] = length - out[flip, ..., 0]
    out[flip, ..., 1] = width - out[flip, ..., 1]
    return out


def normalized(tracking, team):
    """Tracking in the frame of `team` attacking +x. meta['attacking_team'] records which."""
    flip = attack_flip(tracking, team)
    pitch = tracking.meta["pitch"]
    meta = dict(tracking.meta, attacking_team=int(team))
    return replace(tracking, players_xy=rotate(tracking.players_xy, flip, pitch),
                   ball_xy=rotate(tracking.ball_xy, flip, pitch), meta=meta)
