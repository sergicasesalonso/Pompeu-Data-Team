"""A2 context: match clock and Smart Tagging, as per-frame evidence.

What the Smart Tagging `Team` means depends on the code, checked against the public window:
  - attacking codes (BUILD UP, PROGRESSION, CREATING CHANCES, LONG BALL, ATTACKING TRANSITION,
    SET PIECES) name the team WITH the ball: from 67% (ATTACKING TRANSITION) to 100% (BUILD UP,
    CREATING CHANCES) of their live public frames agree;
  - defending codes (DEFENDING IN * THIRD, DEFENSIVE TRANSITION) name the team WITHOUT it, with
    the mirror-image agreement (they are tagged in pairs with the attacking ones);
  - zone codes (BALL IN FINAL THIRD, BALL IN THE BOX, PLAYERS IN THE BOX) are close to 50/50 and
    get no vote, only their own signed column.
So each active phase votes +1 (home has the ball) or -1 (away), and `st_vote` sums the votes.
Signed team columns use the same convention: +1 home, -1 away, 0 unknown (N/A).

    half, minute                 1/2 and minutes since that half's first frame (0 / NaN outside)
    st_active                    phases covering the frame
    st_vote, st_vote_n           sum of possession votes, and how many phases voted
    st_zone_team                 signed team of active zone codes
    st_prev_start_s, st_prev_end_s, st_next_start_s, st_prev_vote
                                 time since the last phase started / ended, until the next starts,
                                 and the vote of the last phase that started
    sp_<type>                    a set piece of that type is active (throw-in, corner, ...)
    sp_team                      signed team of the active set piece (0 for N/A: all throw-ins)
    sp_prev_s, sp_next_s         time since the last / until the next set piece starts
    goal_active                  inside a GOALS phase
"""

import numpy as np

from pompeu import paths  # noqa: F401
import games

# Params
ATTACK_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "LONG BALL",
                "ATTACKING TRANSITION", "SET PIECES")
DEFEND_CODES = ("DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
                "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION")
ZONE_CODES = ("BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX")
SET_PIECE_TYPES = {"throw-in": "throw_in", "corner kick": "corner", "goal kick": "goal_kick",
                   "free kick": "free_kick", "kick off (after goal)": "kickoff_goal",
                   "kick off (start)": "kickoff_start"}


def signed_team(team):
    """0 home -> +1, 1 away -> -1, unknown -> 0."""
    return np.where(team == 0, 1.0, np.where(team == 1, -1.0, 0.0))


def possession_vote(code, team):
    """+1 home has the ball, -1 away, 0 no vote, for one tagging row."""
    sign = float(signed_team(np.array(team)))
    if code in ATTACK_CODES:
        return sign
    if code in DEFEND_CODES:
        return -sign
    return 0.0


def _coverage(starts, stops, values, n):
    """Sum of `values` over the intervals covering each of n frames (difference array)."""
    delta = np.zeros(n + 1)
    np.add.at(delta, np.clip(starts, 0, n), values)
    np.add.at(delta, np.clip(stops, 0, n), -np.asarray(values, dtype=float))
    return np.cumsum(delta)[:n]


def _since(events, n, fps):
    """Seconds since the latest event frame <= each frame (NaN before the first)."""
    marks = np.full(n, -1)
    events = events[(events >= 0) & (events < n)]
    marks[events] = events
    last = np.maximum.accumulate(marks)
    index = np.arange(n)
    return np.where(last >= 0, (index - last) / fps, np.nan), last


def _until(events, n, fps):
    """Seconds until the next event frame >= each frame (NaN after the last)."""
    marks = np.full(n, n)
    events = events[(events >= 0) & (events < n)]
    marks[events] = events
    following = np.minimum.accumulate(marks[::-1])[::-1]
    index = np.arange(n)
    return np.where(following < n, (following - index) / fps, np.nan)


TAGGING_COLUMNS = ("st_active", "st_vote", "st_vote_n", "st_zone_team", "sp_team", "goal_active",
                   "st_prev_start_s", "st_prev_vote", "st_prev_end_s", "st_next_start_s",
                   "sp_prev_s", "sp_next_s") + tuple("sp_" + v for v in SET_PIECE_TYPES.values())


def context_features(tracking, tagging):
    """{column: (N,)} for the frames of `tracking` (crops included). tagging from io.tagging.

    With tagging=None (a match without Smart Tagging, e.g. an external dataset) the tagging
    columns exist and are NaN, so every match has the same schema.
    """
    game = tracking.meta.get("game", games.GAME)
    fps = tracking.meta.get("fps", 25)
    n_video = games.GAMES[game]["n_video_frames"]
    rows = tracking.frames

    half = tracking.half().astype(float)
    minute = np.full(len(rows), np.nan)
    for number, first, last in games.half_video_blocks(game):
        inside = (rows >= first) & (rows <= last)
        minute[inside] = (rows[inside] - first) / fps / 60.0
    out = {"half": half, "minute": minute}
    if tagging is None:
        out.update({name: np.full(len(rows), np.nan) for name in TAGGING_COLUMNS})
        return out

    starts =tagging["start_frame"].to_numpy()
    stops = tagging["end_frame"].to_numpy()
    codes = tagging["code"].to_numpy()
    teams = tagging["team"].to_numpy()
    votes = np.array([possession_vote(c, t) for c, t in zip(codes, teams)])
    voted = np.array([c in ATTACK_CODES + DEFEND_CODES and t >= 0 for c, t in zip(codes, teams)])
    zone = np.isin(codes, ZONE_CODES)
    set_piece = codes == "SET PIECES"
    full = {
        "st_active": _coverage(starts, stops, np.ones(len(starts)), n_video),
        "st_vote": _coverage(starts, stops, votes, n_video),
        "st_vote_n": _coverage(starts, stops, voted.astype(float), n_video),
        "st_zone_team": _coverage(starts[zone], stops[zone], signed_team(teams[zone]), n_video),
        "sp_team": _coverage(starts[set_piece], stops[set_piece],
                             signed_team(teams[set_piece]), n_video),
        "goal_active": np.minimum(_coverage(starts[codes == "GOALS"], stops[codes == "GOALS"],
                                            np.ones(int(np.sum(codes == "GOALS"))), n_video), 1),
    }
    types = tagging["type"].to_numpy()
    for value, name in SET_PIECE_TYPES.items():
        chosen = set_piece & (types == value)
        full["sp_" + name] = np.minimum(_coverage(starts[chosen], stops[chosen],
                                                  np.ones(int(chosen.sum())), n_video), 1)

    order = np.argsort(starts, kind="stable")
    full["st_prev_start_s"], last = _since(starts, n_video, fps)
    vote_at_start = np.zeros(n_video)
    vote_at_start[np.clip(starts[order], 0, n_video - 1)] = votes[order]  # later start wins
    full["st_prev_vote"] = np.where(last >= 0, vote_at_start[np.maximum(last, 0)], np.nan)
    full["st_prev_end_s"], _ = _since(stops, n_video, fps)
    full["st_next_start_s"] = _until(starts, n_video, fps)
    full["sp_prev_s"], _ = _since(starts[set_piece], n_video, fps)
    full["sp_next_s"] = _until(starts[set_piece], n_video, fps)

    if set(full) != set(TAGGING_COLUMNS):
        raise AssertionError("TAGGING_COLUMNS is out of date: {}".format(
            sorted(set(full) ^ set(TAGGING_COLUMNS))))
    for name in TAGGING_COLUMNS:  # one fixed order, with or without tagging
        out[name] = full[name][rows]
    return out
