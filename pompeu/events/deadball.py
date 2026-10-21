"""A3 dead-ball detector: P(D) per frame, the dead-ball stretches, and how and by whom play restarts.

    from pompeu.events.deadball import detect
    result = detect(clean_tracking)
    result.p_dead        (N,) posterior probability that the ball is dead
    result.dead          (N,) bool, the most likely dead/live segmentation
    result.segments      [(start, stop)] dead stretches, stop exclusive
    result.restarts      DataFrame: frame, type, team, confidence, source

WHY A TWO-STATE HMM. A dead ball is a stretch, not a frame, and every cue for it is weak on its own
-- players slowing down also happens in live play, a ball leaving the tracker happens everywhere.
So each cue adds log-likelihood-ratio evidence for "dead" or "live" per frame, the two states are
sticky (switching costs log(1/p_switch) nats, so it takes sustained evidence), Viterbi gives the
segmentation and forward-backward gives P(D). No frame-level threshold anywhere.

The cues, all from A2 features (log-odds per frame, before the autocorrelation scale below):

    slow field        speed_slope * (speed_ref - mean player speed over 2 s), clipped. The plan's
                      1.4 m/s reference: on the public window 74% of dead frames sit below it.
    ball out          the ball is beyond a line by > 0.5 m: dead by the Laws, so its weight (8)
                      beats the strongest live cue from moving players (-4).
    ball absent       weakly dead: the tracker loses the ball more during stoppages.
    still at a spot   the ball stands still (> 1 s) at a corner, penalty spot, centre spot, on the
                      touchline or in the six-yard box: placed for a restart.
    set piece soon    a Smart Tagging set piece starts within the next few seconds.
    goal tag          inside a GOALS phase.
    before kick-off   from a half's first graded frame to the kick inside its "Kick Off (Start)"
                      tag: dead by definition (the 2nd half is graded 2934 frames before its
                      kick-off). Only applied when the half's first frame is in this Tracking.
    kick              strong LIVE for kick_hold_s after a restart-like kick: the ball goes from
                      slow or absent (>= 1 s) to > 4 m/s within 0.5 s, with a player within 2 m
                      (or reappears already fast after a long absence). Without it the slow players
                      right after a restart would keep the state dead for seconds.

Consecutive frames are far from independent (the speeds are 2-s averages), so every frame's
evidence is multiplied by 1 / (EVIDENCE_TAU_S * fps): roughly one independent look per 0.5 s.

After Viterbi, live gaps shorter than 5 s between two dead stretches are closed (a ball fetched
or thrown back during a stoppage looks like a restart; a restart followed by another stoppage
within 5 s is rarer -- an assumption to check on A6). Then each stretch's START is pulled back to the ball crossing a line if that happened
just before, or else to where the field started slowing down, and its END is pulled to the restart
kick if one is near. Then each restart gets a type and a team:

    ball left over a touchline           throw-in, to the opponents of the last player near it
    over a goal line, inside the mouth   goal, kickoff to the team that conceded
    over a goal line, outside            corner if the defenders touched it last, else goal kick
    no exit seen                         by where the ball is put down: corner arc, six-yard box,
                                         touchline, centre spot, else free kick
    kicker                               the team of the player within 2 m at the kick
    Smart Tagging                        a set piece tag around the restart gives the type, and the
                                         team when it is not N/A (it is N/A on every throw-in)

Team priority: tag > geometry > kicker; confidence grows with how many agree.

Every number here is a physical or literature value, not fitted -- the public window has four dead
stretches, too few to fit anything. A6 (external matches with real BallStatus) is where they get
calibrated, and the plan's done-criterion (live/dead F1 > 0.90 and median restart error < 10
frames on `external`) can only be checked there.
"""

import copy
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pompeu.features.kinematics import runs

# Params
PARAMS = {
    "speed_ref": 1.4,          # m/s, the plan's threshold
    "speed_slope": 3.0,        # log-odds per m/s below / above it
    "speed_clip": (-4.0, 3.0),
    "speed_window_s": 2.0,
    # Out of play by the Laws whatever the players do: must outweigh the strongest live
    # evidence the slow-field cue can give (its clip, -4), or players chasing the ball keep it live.
    "out_llr": 8.0,
    "absent_llr": 0.5,
    "still_spot_llr": 2.0,
    "still_llr": 0.5,
    "setpiece_soon_s": 3.0,
    "setpiece_soon_llr": 1.0,
    "goal_llr": 2.0,
    "pre_kickoff_llr": 8.0,    # before a half's kick-off the ball is dead by definition
    "kick_llr": -4.0,
    "kick_hold_s": 1.5,
    "kick_speed": 4.0,         # m/s after the kick
    "kick_quiet_speed": 2.5,   # m/s: the ball was slower than this, or absent, ...
    "kick_quiet_s": 1.0,       # ... for at least this long
    "kick_ramp_s": 0.5,        # and got from quiet to fast within this
    "kick_player_m": 2.0,
    "kick_reappear_s": 2.0,    # or it reappears already fast after an absence this long
    "evidence_tau_s": 0.5,
    "mean_live_s": 40.0,       # expected stretch lengths -> switching probabilities
    "mean_dead_s": 20.0,
    # A live stretch shorter than this between two dead ones is a ball handled during the
    # stoppage (fetched, thrown back), not a restart: the two dead stretches are merged.
    "min_live_gap_s": 5.0,
    "start_lookback_s": 4.0,
    "start_speed": 2.0,        # pull the start back to the last frame the field was this fast
    "end_lookback_s": 6.0,
    "end_lookahead_s": 1.0,
    "restart_tag_margin_s": 3.0,
}
SPOT_CORNER_M = 3.0
SPOT_PENALTY_M = 1.5
SPOT_CENTRE_M = 2.0
SPOT_TOUCHLINE_M = 1.0
SIX_YARD_DEPTH, SIX_YARD_HALF_WIDTH = 5.5, 9.16
GOAL_HALF_WIDTH = 3.66
TYPES = ("throw_in", "goal_kick", "corner", "free_kick", "kickoff")  # a goal restarts as kickoff
FEATURE_GROUPS = ("ball", "proximity", "collective", "context")


@dataclass
class DeadBall:
    p_dead: np.ndarray
    dead: np.ndarray
    segments: list
    restarts: pd.DataFrame
    evidence: dict
    frame0: int = 0


def _settings(params):
    out = copy.deepcopy(PARAMS)
    unknown = set(params or {}) - set(out)
    if unknown:
        raise KeyError("unknown deadball params {}".format(sorted(unknown)))
    out.update(params or {})
    return out


def _col(F, name):
    return F[name].to_numpy(dtype=np.float64)


# -- evidence ----------------------------------------------------------------------------------

def kick_frames(F, fps, p):
    """Frames (row indices) where a restart-like kick starts.

    A kick is the ball going from quiet (slower than kick_quiet_speed, or absent) for at least
    kick_quiet_s to faster than kick_speed within kick_ramp_s -- a short goal kick accelerates over
    most of a second, it does not jump -- with a player within kick_player_m when it starts to
    move. The kick is dated at that start of motion. A ball that reappears already fast after a
    long absence also counts (dated when it reappears): the restart happened unseen.
    """
    speed = _col(F, "ball_speed")
    present = _col(F, "ball_present") > 0
    near = np.fmin(_col(F, "d1_H"), _col(F, "d1_A"))
    n = len(speed)
    with np.errstate(invalid="ignore"):
        quiet = ~present | (speed < p["kick_quiet_speed"])
        fast = present & (speed > p["kick_speed"])
    # Length of the quiet run ending at each frame, and the last quiet frame at or before it.
    quiet_len = np.zeros(n, dtype=np.int64)
    starts, stops = runs(quiet)
    for start, stop in zip(starts, stops):
        quiet_len[start:stop] = np.arange(1, stop - start + 1)
    last_quiet = np.maximum.accumulate(np.where(quiet, np.arange(n), -1))

    first_fast = np.flatnonzero(fast & ~np.concatenate(([False], fast[:-1])))
    ramp = int(round(p["kick_ramp_s"] * fps))
    need = int(round(p["kick_quiet_s"] * fps))
    kicks = set()
    for t in first_fast:
        q = last_quiet[t - 1] if t > 0 else -1
        if q < 0 or t - q > ramp + 1 or quiet_len[q] < need:
            continue
        moving = q + 1
        if not present[q] and present[moving] and t == moving:
            continue  # reappeared already fast: handled below with the absence rule
        if near[moving] <= p["kick_player_m"]:
            kicks.add(int(moving))
    age = _col(F, "ball_gap_age_s")
    reappear = np.flatnonzero(fast[1:] & ~present[:-1] & (age[:-1] >= p["kick_reappear_s"])) + 1
    kicks.update(int(t) for t in reappear)
    return np.array(sorted(kicks), dtype=np.int64)


def at_restart_spot(F):
    """Boolean: the ball is somewhere a restart is taken from."""
    with np.errstate(invalid="ignore"):
        spot = ((_col(F, "ball_dist_corner") < SPOT_CORNER_M)
                | (_col(F, "ball_dist_penalty") < SPOT_PENALTY_M)
                | (_col(F, "ball_dist_centre") < SPOT_CENTRE_M)
                | (np.abs(_col(F, "ball_to_touchline_min")) < SPOT_TOUCHLINE_M))
    return spot | in_six_yard_box(F)


def in_six_yard_box(F):
    x_line = _col(F, "ball_to_goal_line_min")
    y = _col(F, "ball_y_home")
    with np.errstate(invalid="ignore"):
        return (x_line >= 0) & (x_line < SIX_YARD_DEPTH) & (np.abs(y - 34.0) < SIX_YARD_HALF_WIDTH)


def evidence(F, fps, params=None):
    """{cue: (N,) log-odds for DEAD}. Sum them for the HMM's per-frame evidence."""
    p = _settings(params)
    n = len(F)
    width = max(1, int(round(p["speed_window_s"] * fps)))
    speed = pd.Series(_col(F, "field_speed_mean")).rolling(width, center=True,
                                                           min_periods=1).mean().to_numpy()
    slow = np.clip(p["speed_slope"] * (p["speed_ref"] - speed), *p["speed_clip"])
    out = {"slow_field": np.nan_to_num(slow)}

    out["ball_out"] = np.where(_col(F, "ball_out") > 0, p["out_llr"], 0.0)
    out["ball_absent"] = np.where(_col(F, "ball_present") > 0, 0.0, p["absent_llr"])
    still = _col(F, "ball_still") > 0
    out["still_spot"] = np.where(still & at_restart_spot(F), p["still_spot_llr"],
                                 np.where(still, p["still_llr"], 0.0))
    soon = _col(F, "sp_next_s")
    with np.errstate(invalid="ignore"):
        out["setpiece_soon"] = np.where((soon > 0) & (soon <= p["setpiece_soon_s"]),
                                        p["setpiece_soon_llr"], 0.0)
    out["goal_tag"] = np.where(_col(F, "goal_active") > 0, p["goal_llr"], 0.0)

    kicks = kick_frames(F, fps, p)
    kick = np.zeros(n)
    hold = int(round(p["kick_hold_s"] * fps))
    for frame in kicks:
        kick[frame:frame + hold] = p["kick_llr"]
    out["kick"] = kick
    out["pre_kickoff"] = pre_kickoff(F, kicks, p)
    return out


def pre_kickoff(F, kicks, p):
    """Dead evidence from each half's first graded frame to its kick-off kick.

    The kick-off is the first kick inside the half's "Kick Off (Start)" Smart Tagging phase (or
    the phase's start when no kick is seen there). Needs the half's true first frame to be in F,
    so a crop that starts mid-half gets nothing.
    """
    import games
    out = np.zeros(len(F))
    frames = F.index.to_numpy()
    tag = _col(F, "sp_kickoff_start") > 0
    for half, first, _ in games.half_video_blocks(games.GAME):
        rows = np.flatnonzero(frames == first)
        if not len(rows):
            continue
        start = int(rows[0])
        tagged = np.flatnonzero(tag[start:]) + start
        if not len(tagged):
            continue
        tag_start = int(tagged[0])
        tag_stop = tag_start + int(np.argmin(tag[tag_start:])) if not tag[tag_start:].all()             else len(F)
        inside = kicks[(kicks >= max(start, tag_start - 1)) & (kicks < tag_stop)]
        stop = int(inside[0]) if len(inside) else tag_start
        out[start:stop] = p["pre_kickoff_llr"]
    return out


# -- two-state HMM -----------------------------------------------------------------------------

def hmm(llr_dead, p_live_to_dead, p_dead_to_live):
    """(posterior P(dead) (N,), Viterbi path (N,) bool) for per-frame log-odds of dead vs live.

    Emissions enter as log-likelihoods (live 0, dead llr); the prior is 50/50. Plain Python loops
    over floats: two states, so three passes over 199k frames take a second or two.
    """
    n = len(llr_dead)
    stay_l, go_ld = math.log1p(-p_live_to_dead), math.log(p_live_to_dead)
    stay_d, go_dl = math.log1p(-p_dead_to_live), math.log(p_dead_to_live)
    e = [float(v) for v in llr_dead]

    def lse(a, b):
        m = a if a > b else b
        return m + math.log1p(math.exp(-abs(a - b)))

    fl = [0.0] * n
    fd = [0.0] * n
    fl[0], fd[0] = math.log(0.5), math.log(0.5) + e[0]
    for t in range(1, n):
        fl[t] = lse(fl[t - 1] + stay_l, fd[t - 1] + go_dl)
        fd[t] = lse(fl[t - 1] + go_ld, fd[t - 1] + stay_d) + e[t]
    bl = [0.0] * n
    bd = [0.0] * n
    for t in range(n - 2, -1, -1):
        bl[t] = lse(stay_l + bl[t + 1], go_ld + e[t + 1] + bd[t + 1])
        bd[t] = lse(go_dl + bl[t + 1], stay_d + e[t + 1] + bd[t + 1])
    post = np.array([1.0 / (1.0 + math.exp(max(-700.0, min(700.0, (fl[t] + bl[t])
                                                               - (fd[t] + bd[t])))))
                     for t in range(n)])

    # Viterbi, with back-pointers as two bit arrays.
    vl, vd = math.log(0.5), math.log(0.5) + e[0]
    from_l = np.zeros(n, dtype=bool)  # True: best predecessor of live at t was dead
    from_d = np.zeros(n, dtype=bool)  # True: best predecessor of dead at t was dead
    for t in range(1, n):
        a, b = vl + stay_l, vd + go_dl
        c, d = vl + go_ld, vd + stay_d
        from_l[t], from_d[t] = b > a, d > c
        vl, vd = (b if b > a else a), (d if d > c else c) + e[t]
    path = np.zeros(n, dtype=bool)
    path[-1] = vd > vl
    for t in range(n - 1, 0, -1):
        path[t - 1] = from_d[t] if path[t] else from_l[t]
    return post, path


# -- segments and restarts ---------------------------------------------------------------------

def refine(segments, F, fps, kicks, p):
    """Pull each dead stretch's start back to the exit or the slowdown, and its end to the kick."""
    n = len(F)
    out_now = _col(F, "ball_edge_dist")
    speed = pd.Series(_col(F, "field_speed_mean")).rolling(
        max(1, int(round(p["speed_window_s"] * fps))), center=True, min_periods=1).mean().to_numpy()
    look_start = int(round(p["start_lookback_s"] * fps))
    look_end = int(round(p["end_lookback_s"] * fps))
    ahead = int(round(p["end_lookahead_s"] * fps))
    refined = []
    for start, stop in segments:
        lo = max(0, start - look_start)
        with np.errstate(invalid="ignore"):
            crossed = np.flatnonzero(out_now[lo:start + ahead] < 0)
        if len(crossed):
            new_start = lo + int(crossed[0])
        else:
            with np.errstate(invalid="ignore"):
                fast = np.flatnonzero(speed[lo:start] >= p["start_speed"])
            new_start = lo + int(fast[-1]) + 1 if len(fast) else start
        candidates = kicks[(kicks >= max(new_start + 1, stop - look_end)) & (kicks <= stop + ahead)]
        new_stop = int(candidates[-1]) if len(candidates) else stop
        refined.append((int(new_start), int(min(new_stop, n))))
    # Refining can make neighbours touch or overlap, and short live gaps are closed: merge.
    gap = int(round(p["min_live_gap_s"] * fps))
    merged = []
    for start, stop in sorted(refined):
        if stop <= start:
            continue
        if merged and start - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _signed_to_team(sign):
    return 0 if sign > 0 else (1 if sign < 0 else -1)


def classify_restart(F, tracking, start, stop, tags, p):
    """{'type', 'team', 'confidence', 'source'} for the restart that ends dead stretch [start, stop)."""
    fps = tracking.meta.get("fps", 25)
    n = len(F)
    kick = min(stop, n - 1)
    x_home = _col(F, "ball_x_home")
    y_home = _col(F, "ball_y_home")
    edge = _col(F, "ball_edge_dist")
    nearest = _col(F, "nearest_team")
    near = np.fmin(_col(F, "d1_H"), _col(F, "d1_A"))
    half = int(_col(F, "half")[kick]) or 1
    length = tracking.meta["pitch"][0]

    def defender_of(goal_x_home):
        """Team defending the goal at x = 0 or x = length of the HOME frame: home defends 0."""
        return 0 if goal_x_home < length / 2 else 1

    geometry_type, geometry_team = None, -1
    # 1) Did the ball leave the pitch around the start? Last frame inside, first frame outside.
    window = np.arange(max(0, start - int(2 * fps)), min(n, start + int(3 * fps)))
    with np.errstate(invalid="ignore"):
        outside = window[edge[window] < 0]
    if len(outside):
        first_out = int(outside[0])
        inside = window[(window < first_out) & ~np.isnan(edge[window]) & (edge[window] >= 0)]
        toucher = int(nearest[inside[-1]]) if len(inside) and not np.isnan(nearest[inside[-1]]) \
            and near[inside[-1]] < 3.0 else -1
        x, y = x_home[first_out], y_home[first_out]
        over_goal_line = min(x, length - x) < min(y, 68.0 - y)
        if not over_goal_line:
            geometry_type = "throw_in"
            geometry_team = 1 - toucher if toucher >= 0 else -1
        else:
            defender = defender_of(x)
            if abs(y - 34.0) < GOAL_HALF_WIDTH:
                geometry_type, geometry_team = "goal", defender
            elif toucher == defender:
                geometry_type, geometry_team = "corner", 1 - defender
            else:
                geometry_type, geometry_team = "goal_kick", defender
    else:
        # 2) No exit seen: where is the ball put down for the restart?
        row = F.iloc[[kick]]
        if (_col(row, "ball_dist_corner") < SPOT_CORNER_M).any():
            geometry_type = "corner"
            geometry_team = 1 - defender_of(x_home[kick])
        elif in_six_yard_box(row).any():
            geometry_type, geometry_team = "goal_kick", defender_of(x_home[kick])
        elif (np.abs(_col(row, "ball_to_touchline_min")) < 1.5).any():
            geometry_type = "throw_in"
        elif (_col(row, "ball_dist_centre") < SPOT_CENTRE_M).any():
            geometry_type = "kickoff"
        else:
            geometry_type = "free_kick"
    if geometry_type == "goal":
        restart_type = "kickoff"
    else:
        restart_type = geometry_type

    kicker = int(nearest[kick]) if not np.isnan(nearest[kick]) and near[kick] <= 2.0 else -1

    tag_type, tag_team = None, -1
    margin = int(round(p["restart_tag_margin_s"] * fps))
    if tags is not None and len(tags):
        hit = tags[(tags["start_frame"] - margin <= kick) & (tags["end_frame"] + margin >= kick)]
        if len(hit):
            best = hit.iloc[np.argmin(np.abs(hit["start_frame"].to_numpy() - kick))]
            tag_type = {"throw-in": "throw_in", "corner kick": "corner", "goal kick": "goal_kick",
                        "free kick": "free_kick", "kick off (after goal)": "kickoff",
                        "kick off (start)": "kickoff"}.get(best["type"])
            tag_team = int(best["team"])
    if tag_type:
        restart_type = tag_type

    sources = (("tag", tag_team), ("geometry", geometry_team), ("kicker", kicker))
    team = next((value for _, value in sources if value >= 0), -1)
    agree = sum(1 for _, value in sources if value >= 0 and value == team)
    disagree = sum(1 for _, value in sources if value >= 0 and value != team)
    confidence = 0.0 if team < 0 else max(0.0, {1: 0.6, 2: 0.8, 3: 0.95}[agree] - 0.3 * disagree)
    source = "+".join(name for name, value in sources if value == team and team >= 0)
    return {"type": restart_type, "team": team, "confidence": confidence,
            "source": source or "none", "exit": geometry_type if len(outside) else None}


def detect(tracking, F=None, params=None, tags="auto", extra_llr=None):
    """DeadBall for a (cleaned) Tracking. F: A2 features for it (built if None).

    extra_llr: optional (N,) log-odds of dead from another source -- the LightGBM live/dead
    classifier once A6 provides labelled external matches, video or audio cues from Line B --
    added to the rule cues as one more expert.
    """
    from pompeu.features.build import build_features, load_tagging_for
    p = _settings(params)
    fps = tracking.meta.get("fps", 25)
    if F is None:
        F = build_features(tracking, with_windows=False, groups=FEATURE_GROUPS)
    if isinstance(tags, str) and tags == "auto":
        tagging = load_tagging_for(tracking)
        tags = None if tagging is None else tagging[tagging["code"] == "SET PIECES"]

    cues = evidence(F, fps, p)
    if extra_llr is not None:
        cues["extra"] = np.nan_to_num(np.asarray(extra_llr, dtype=np.float64))
    scale = 1.0 / (p["evidence_tau_s"] * fps)
    llr = scale * np.sum(list(cues.values()), axis=0)
    p_dead, path = hmm(llr, 1.0 / (p["mean_live_s"] * fps), 1.0 / (p["mean_dead_s"] * fps))

    starts, stops = runs(path)
    kicks = kick_frames(F, fps, p)
    segments = refine(list(zip(starts.tolist(), stops.tolist())), F, fps, kicks, p)
    dead = np.zeros(len(F), dtype=bool)
    rows = []
    for start, stop in segments:
        dead[start:stop] = True
        if stop < len(F):
            info = classify_restart(F, tracking, start, stop, tags, p)
            rows.append(dict(frame=tracking.frame0 + stop, start=tracking.frame0 + start, **info))
    restarts = pd.DataFrame(rows, columns=["frame", "start", "type", "team", "confidence",
                                           "source", "exit"])
    return DeadBall(p_dead=p_dead, dead=dead,
                    segments=[(tracking.frame0 + a, tracking.frame0 + b) for a, b in segments],
                    restarts=restarts, evidence=cues, frame0=tracking.frame0)
