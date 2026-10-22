"""A4: rules_v0 -- the first real possession model. A3 decides dead/live, A4 decides who has it.

The challenge's rule is that possession changes only when the other team gains CONTROL, and the
public ground truth shows where it then dates the change: at that team's FIRST touch of the
sequence that ends in control (an interception is H -> A at the intercepting touch, not when the
interceptor has settled the ball). So the state machine below waits for control and then
back-dates the change.

Inside each live stretch (pompeu.events.deadball), walking the ball events of
pompeu.events.touches in time order, with `poss` the team in possession:

  start        poss = the team the restart was given to (A3); if unknown, the first event's team.
  own event    a touch or control by `poss` confirms it and cancels any pending rival touch
               (a rival touch followed by ours was a deflection).
  rival control (the ball stays within 1.5 m of one rival player >= 0.6 s)
               -> possession changes, dated at the pending rival touch if there is one, else at
               the start of the control.
  rival touch  -> becomes pending. A second rival touch within chain_gap_s (chained passes, or
               touch-and-carry) confirms it: the change is dated at the first one.
  fly-by       a touch with the ball faster than fast_ball_speed both before and after it is a
               ball flying past or over a player (a header, an aerial duel, a false candidate):
               ignored -- "an aerial duel won without subsequent control" changes nothing.
  uncertain    events whose team call is closer to 50/50 than min_team_confidence are skipped:
               two players of different teams at the same distance say nothing.
  no ball      no events, so possession simply holds: the challenge's "in phases where the ball is
               contested or loose, possession stays with the last team that had it".

The ball itself is sometimes a false detection -- a "still" ball with nobody within 7 m in open
play -- so events need a player within reach (touch 2 m, control 1.5 m), which is what keeps
those phantom balls from ever producing a change.
"""

import copy

import numpy as np

from pompeu.events.deadball import detect as detect_dead
from pompeu.events.touches import detect_events
from pompeu.features.kinematics import runs

# Params
PARAMS = {
    "chain_gap_s": 3.0,
    "fast_ball_speed": 15.0,       # m/s before AND after a touch: the ball flew past
    "min_team_confidence": 0.3,    # |2 * p_home - 1|
    # Where a confirmed change is dated. "touch": the pending rival touch, else the start of the
    # rival control. "streak": further back, to where the rival became the team nearest the ball
    # (never before possession's own last event) -- touches are sparse in this tracking, so the
    # first rival touch the ground truth dates changes at is often not detected at all.
    "backdate": "streak",
}
HOME, AWAY = 0, 1


def _settings(params):
    out = copy.deepcopy(PARAMS)
    unknown = set(params or {}) - set(out)
    if unknown:
        raise KeyError("unknown rules params {}".format(sorted(unknown)))
    out.update(params or {})
    return out


def streak_start(nearest, frame, team, floor):
    """Earliest frame >= floor such that `team` is the nearest team on every frame up to `frame`."""
    at = frame
    while at - 1 >= floor and nearest[at - 1] == team:
        at -= 1
    return at


def possession_changes(events, start, stop, opening, fps, p, nearest=None):
    """[(frame, team)] change points for one live stretch [start, stop), rows relative.

    `nearest` (N,) is the nearest-player team per frame (-1 without ball), used by backdate="streak".
    """
    chain = int(round(p["chain_gap_s"] * fps))
    poss = opening if opening in (HOME, AWAY) else None
    changes = [(start, poss)]
    pending = None  # (team, frame) of an unconfirmed rival touch
    last_own = start  # frame of the possession team's latest event
    for ev in events:
        if ev.frame < start or ev.frame >= stop:
            continue
        if abs(2 * ev.p_home - 1) < p["min_team_confidence"]:
            continue
        if ev.kind == "touch" and min(ev.speed_in, ev.speed_out) > p["fast_ball_speed"]:
            continue
        team = int(ev.team)
        if poss is None:
            poss = team
            changes[0] = (start, team)
            continue
        if team == poss:
            pending = None
            last_own = ev.end
            continue
        if ev.kind == "control":
            at = pending[1] if pending is not None and pending[0] == team else ev.frame
        elif pending is not None and pending[0] == team and ev.frame - pending[1] <= chain:
            at = pending[1]
        else:
            pending = (team, ev.frame)
            continue
        if p["backdate"] == "streak" and nearest is not None:
            at = streak_start(nearest, at, team, max(last_own, changes[-1][0] + 1))
        changes.append((at, team))
        poss, pending, last_own = team, None, ev.end
    return changes


def predict(tracking, deadball=None, touches=None, rules=None):
    """(N,) H/A/D labels. Each argument overrides that module's PARAMS."""
    from pompeu.features.build import build_features
    p = _settings(rules)
    fps = tracking.meta.get("fps", 25)
    F = build_features(tracking, with_windows=False,
                       groups=("ball", "proximity", "pitch_control", "collective", "context"))
    dead = detect_dead(tracking, F=F, params=deadball)
    events = detect_events(tracking, F, params=touches)
    events["frame"] -= tracking.frame0
    events["end"] -= tracking.frame0
    rows = list(events.itertuples(index=False))
    restart_team = dict(zip((dead.restarts["frame"] - tracking.frame0).tolist(),
                            dead.restarts["team"].tolist()))
    from pompeu.models.baseline import nearest_team
    nearest = nearest_team(tracking)

    teams = np.full(tracking.n_frames, -1, dtype=np.int64)
    starts, stops = runs(~dead.dead)
    for start, stop in zip(starts, stops):
        changes = possession_changes(rows, int(start), int(stop),
                                     restart_team.get(int(start), -1), fps, p, nearest)
        bounds = [frame for frame, _ in changes[1:]] + [int(stop)]
        for (frame, team), end in zip(changes, bounds):
            teams[frame:end] = -1 if team is None else team
    labels = np.where(teams == HOME, "H", np.where(teams == AWAY, "A", "D"))
    labels[dead.dead] = "D"
    # A live stretch where nothing ever named a team: fall back to the nearest-player vote.
    unknown = (teams < 0) & ~dead.dead
    if unknown.any():
        from pompeu.models.deadball_vote import vote_in_stretch
        for start, stop in zip(*runs(unknown)):
            voted = vote_in_stretch(nearest[start:stop], stop - start)
            labels[start:stop] = np.where(voted == AWAY, "A", "H")
    return labels
