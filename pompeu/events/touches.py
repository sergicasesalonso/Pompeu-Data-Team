"""A4 ball events: touches, and stretches where one player keeps the ball (control).

    from pompeu.events.touches import detect_events
    events = detect_events(clean_tracking, F)     # DataFrame, one row per event, time-ordered

Two kinds of event, both with a team and a probability that it is the home team:

TOUCH. The ball's motion changes -- its speed by more than 3 m/s or its direction by more than 30
degrees between TOUCH_LAG frames before and after -- with a player within 2 m. Consecutive
candidate frames are one touch, dated at the frame of largest change. The team is the nearest
player's, softened when both teams are close: p_home mixes a distance logistic (TOUCH_DIST_SCALE
m per unit of log-odds) with Spearman pitch control at the ball. `speed_in` / `speed_out` are the
ball speed just before and after: fast on both sides means the ball flew past (or over) the player
-- a header, an aerial duel or a false candidate -- not a ball anybody controlled.

CONTROL. The ball stays within 1.5 m of the same tracked player for at least 0.6 s -- a dribble,
or a ball held or placed next to someone. That is control by the rules of the challenge, unlike a
single touch.

Both need the ball; frames without it produce no events, which is why the possession rules that
consume these (pompeu.models.rules) keep the last team in control through every gap.
"""

import numpy as np
import pandas as pd

from pompeu.features.kinematics import ball_kinematics, runs
from pompeu.features.proximity import ball_distances

# Params
PARAMS = {
    "touch_speed_change": 3.0,   # m/s
    "touch_turn_deg": 30.0,
    "touch_min_speed": 1.0,      # m/s on both sides for the turn to mean anything
    "touch_lag": 2,              # frames either side
    "touch_player_m": 2.0,
    "touch_merge_frames": 3,     # candidate frames closer than this are one touch
    "touch_dist_scale": 0.5,     # metres of distance difference per unit of log-odds
    "pitch_control_weight": 0.2,
    "control_player_m": 1.5,
    "control_min_s": 0.6,
}


def _settings(params):
    out = dict(PARAMS)
    unknown = set(params or {}) - set(out)
    if unknown:
        raise KeyError("unknown touches params {}".format(sorted(unknown)))
    out.update(params or {})
    return out


def touch_frames(tracking, kin, p):
    """(frames, change) of touch candidates after merging, change = speed change in m/s."""
    vel, speed = kin["vel"], kin["speed"]
    lag = p["touch_lag"]
    n = len(speed)
    before_v = np.full(vel.shape, np.nan)
    after_v = np.full(vel.shape, np.nan)
    before_v[lag:] = vel[:-lag]
    after_v[:-lag] = vel[lag:]
    before_s = np.hypot(*before_v.T)
    after_s = np.hypot(*after_v.T)
    with np.errstate(invalid="ignore"):
        cosine = np.sum(before_v * after_v, axis=1) / (before_s * after_s)
        turn = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        speed_change = np.abs(after_s - before_s)
        turned = (turn > p["touch_turn_deg"]) & (np.fmin(before_s, after_s) > p["touch_min_speed"])
        candidate = (speed_change > p["touch_speed_change"]) | turned
    near = np.nanmin(np.where(np.isnan(ball_distances(tracking)), np.inf,
                              ball_distances(tracking)), axis=1)
    candidate &= near <= p["touch_player_m"]

    # Merge candidates closer than touch_merge_frames into one event at its largest change.
    strength = np.nan_to_num(np.hypot(*(after_v - before_v).T))
    index = np.flatnonzero(candidate)
    frames = []
    if len(index):
        groups = np.split(index, np.flatnonzero(np.diff(index) > p["touch_merge_frames"]) + 1)
        frames = [int(g[np.argmax(strength[g])]) for g in groups]
    frames = np.array(frames, dtype=np.int64)
    return frames, strength[frames] if len(frames) else np.array([])


def detect_events(tracking, F=None, params=None):
    """DataFrame: frame (absolute), end, kind ('touch' | 'control'), team, p_home, distance,
    speed_in, speed_out, slot. Sorted by frame. F: A2 features (proximity + pitch control used if present)."""
    p = _settings(params)
    fps = tracking.meta.get("fps", 25)
    kin = ball_kinematics(tracking)
    distance = ball_distances(tracking)
    filled = np.where(np.isnan(distance), np.inf, distance)
    nearest_slot = np.argmin(filled, axis=1)
    nearest_dist = filled[np.arange(len(filled)), nearest_slot]
    team = tracking.players_team

    def team_distance(side):
        mine = np.where(team == side, filled, np.inf)
        return mine.min(axis=1)

    d_home, d_away = team_distance(0), team_distance(1)
    with np.errstate(invalid="ignore", over="ignore"):
        logit = (d_away - d_home) / p["touch_dist_scale"]
        p_distance = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
    p_distance[np.isinf(d_home) & np.isinf(d_away)] = np.nan
    if F is not None and "pc_ball_H" in F:
        pc = F["pc_ball_H"].to_numpy(dtype=np.float64)
        w = p["pitch_control_weight"]
        p_home = np.where(np.isnan(pc), p_distance, (1 - w) * p_distance + w * pc)
    else:
        p_home = p_distance

    rows = []
    frames, _ = touch_frames(tracking, kin, p)
    lag = p["touch_lag"]
    out_speed = np.full(len(kin["speed"]), np.nan)
    out_speed[:-lag] = kin["speed"][lag:]
    in_speed = np.full(len(kin["speed"]), np.nan)
    in_speed[lag:] = kin["speed"][:-lag]
    for frame in frames:
        rows.append({"frame": frame, "end": frame + 1, "kind": "touch",
                     "slot": int(nearest_slot[frame]), "distance": float(nearest_dist[frame]),
                     "p_home": float(p_home[frame]), "speed_in": float(in_speed[frame]),
                     "speed_out": float(out_speed[frame])})

    # Control: the same slot is the nearest player, within control_player_m, for long enough.
    close = nearest_dist <= p["control_player_m"]
    need = int(round(p["control_min_s"] * fps))
    slot_series = np.where(close, nearest_slot, -1)
    change = np.flatnonzero(np.diff(slot_series) != 0) + 1
    bounds = np.concatenate(([0], change, [len(slot_series)]))
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if slot_series[start] < 0 or stop - start < need:
            continue
        rows.append({"frame": int(start), "end": int(stop), "kind": "control",
                     "slot": int(slot_series[start]),
                     "distance": float(np.mean(nearest_dist[start:stop])),
                     "p_home": float(np.nanmean(p_home[start:stop])),
                     "speed_in": float(np.nanmean(kin["speed"][start:stop])),
                     "speed_out": float(np.nanmean(kin["speed"][start:stop]))})

    events = pd.DataFrame(rows, columns=["frame", "end", "kind", "slot", "distance", "p_home",
                                         "speed_in", "speed_out"])
    events["team"] = np.where(events["p_home"] >= 0.5, 0, 1)
    events["frame"] += tracking.frame0
    events["end"] += tracking.frame0
    return events.sort_values(["frame", "kind"], kind="stable").reset_index(drop=True)
