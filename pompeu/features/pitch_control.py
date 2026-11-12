"""A2 space control: Spearman (2018) pitch control at the ball and on a 5 m ring around it.

Distance to the nearest player ignores that a player running the right way gets there first.
Spearman's model does not: each player's arrival time at a target is

    tau_i = reaction + |(p_i + v_i * reaction) - target| / v_max

his chance of having arrived by time T is a logistic in (T - tau_i), and control accrues as

    dPPCF_team/dT = (1 - PPCF_home - PPCF_away) * sum_{i in team} P_arrived_i(T) * lambda

from the moment the ball itself could get there (ball distance / average ball speed) until one of
the teams holds it. Parameters are the paper's (via Shaw's public implementation): v_max 5 m/s,
reaction 0.7 s, sigma 0.45 s, lambda 4.3/s, ball 15 m/s.

Both teams get the same lambda: the model usually favours the attacking side, but who attacks is
exactly what we are trying to infer, so it must not be an input. Goalkeepers are not given their
own lambda for the same kind of reason -- slots carry no role.

Speed: integrated with dt 0.1 s (the paper uses 0.04; the difference is below 0.01 in control),
vectorised over (targets x players x steps), with only the MAX_PLAYERS earliest arrivals per team --
later players cannot contribute before the earlier ones have taken control. NaN without a ball.
"""

import numpy as np

# Params
V_MAX = 5.0
REACTION = 0.7
SIGMA = 0.45
LAMBDA = 4.3
BALL_SPEED = 15.0
DT = 0.1
MAX_TIME = 10.0
TOLERANCE = 0.01
MAX_PLAYERS = 4
RING_RADIUS = 5.0
RING_POINTS = 8
CHUNK = 20000  # targets per vectorised block


def arrival_times(players_xy, players_vel, targets):
    """(B, K) Spearman time to intercept of every player for each of B targets. inf if absent."""
    start = players_xy + np.nan_to_num(players_vel) * REACTION
    distance = np.hypot(start[..., 0] - targets[:, None, 0], start[..., 1] - targets[:, None, 1])
    tau = REACTION + distance / V_MAX
    return np.where(np.isnan(tau), np.inf, tau)


def control(tau_home, tau_away, ball_time):
    """(B,) home pitch control from arrival times (B, k) of each team and ball arrival (B,)."""
    taus = np.concatenate((tau_home, tau_away), axis=1)
    is_home = np.concatenate((np.ones(tau_home.shape[1]), np.zeros(tau_away.shape[1])))
    home = np.zeros(len(taus))
    away = np.zeros(len(taus))
    scale = np.pi / np.sqrt(3.0) / SIGMA
    for step in range(int(MAX_TIME / DT)):
        remaining = 1.0 - home - away
        if remaining.max() < TOLERANCE:
            break
        T = ball_time + step * DT
        with np.errstate(over="ignore"):
            arrived = 1.0 / (1.0 + np.exp(-scale * (T[:, None] - taus)))
        rate = arrived * LAMBDA * DT * remaining[:, None]
        gain_home = rate @ is_home
        gain_away = rate @ (1.0 - is_home)
        # Never hand out more than what remains (only matters with a coarse dt).
        total = gain_home + gain_away
        excess = np.where(total > remaining, remaining / np.maximum(total, 1e-12), 1.0)
        home += gain_home * excess
        away += gain_away * excess
    with np.errstate(invalid="ignore"):
        return home / (home + away)


def _earliest(tau, k):
    return np.sort(tau, axis=1)[:, :k]


def pitch_control_features(tracking, player_vel):
    """{'pc_ball_H', 'pc_ring_mean_H', 'pc_ring_min_H', 'pc_ring_max_H'}: home control in [0, 1]."""
    n = tracking.n_frames
    out = {name: np.full(n, np.nan) for name in
           ("pc_ball_H", "pc_ring_mean_H", "pc_ring_min_H", "pc_ring_max_H")}
    frames = np.flatnonzero(tracking.ball_present)
    if not len(frames):
        return out

    angles = np.linspace(0, 2 * np.pi, RING_POINTS, endpoint=False)
    offsets = np.vstack(([0.0, 0.0], RING_RADIUS * np.column_stack((np.cos(angles),
                                                                     np.sin(angles)))))
    n_targets = len(offsets)
    team = tracking.players_team
    observed = tracking.players_conf > 0
    per_chunk = max(1, CHUNK // n_targets)
    values = np.empty((len(frames), n_targets))

    for lo in range(0, len(frames), per_chunk):
        chunk = frames[lo:lo + per_chunk]
        ball = tracking.ball_xy[chunk]
        targets = (ball[:, None, :] + offsets[None]).reshape(-1, 2)
        repeat = np.repeat(np.arange(len(chunk)), n_targets)
        xy = np.where(observed[chunk][..., None], tracking.players_xy[chunk], np.nan)[repeat]
        tau = arrival_times(xy, player_vel[chunk][repeat], targets)
        side = team[chunk][repeat]
        tau_home = _earliest(np.where(side == 0, tau, np.inf), MAX_PLAYERS)
        tau_away = _earliest(np.where(side == 1, tau, np.inf), MAX_PLAYERS)
        ball_time = np.hypot(*(targets - ball[repeat]).T) / BALL_SPEED
        values[lo:lo + len(chunk)] = control(tau_home, tau_away, ball_time).reshape(-1, n_targets)

    out["pc_ball_H"][frames] = values[:, 0]
    ring = values[:, 1:]
    out["pc_ring_mean_H"][frames] = np.nanmean(ring, axis=1)
    out["pc_ring_min_H"][frames] = np.nanmin(ring, axis=1)
    out["pc_ring_max_H"][frames] = np.nanmax(ring, axis=1)
    return out
