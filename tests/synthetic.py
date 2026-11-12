"""Small synthetic Trackings for unit tests: known geometry, known answer."""

import numpy as np

from pompeu.io.contract import SRC_ATD, SRC_NONE, Tracking

FPS = 25
PITCH = (105.3, 68.0)


def make_tracking(players_xy, team, ball_xy, frame0=0):
    """Tracking from positions; team is (K,). Observed wherever the position is not NaN."""
    n, k = players_xy.shape[:2]
    present = ~np.isnan(players_xy[:, :, 0])
    ball_present = ~np.isnan(ball_xy[:, 0])
    return Tracking(players_xy=players_xy, players_team=np.broadcast_to(team, (n, k)),
                    players_conf=present.astype(float), ball_xy=ball_xy,
                    ball_z=np.full(n, np.nan), ball_conf=ball_present.astype(float),
                    ball_source=np.where(ball_present, SRC_ATD, SRC_NONE), frame0=frame0,
                    meta={"game": "BAR-ATH", "fps": FPS, "pitch": PITCH, "home_attack_sign": 1,
                          "team_names": ("Barcelona", "Atletic Club")})


def static_players(n, positions):
    """(n, K, 2) players standing at `positions` (K, 2) for n frames."""
    return np.repeat(np.asarray(positions, dtype=float)[None], n, axis=0)
