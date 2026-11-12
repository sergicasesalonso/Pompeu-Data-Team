"""A6: make clean external tracking look like our ATD before any method sees it.

A method validated on clean DFL tracking would be validated on the wrong problem: our ATD loses
the ball on 41% of graded frames, jumps it to false detections, and sees ~10.2 of 11 players per
team in fragments. Each degradation below reproduces a statistic measured on our ATD:

  ball gaps      the ball is erased in runs whose lengths are drawn from the ATD's own no-ball
                 runs (median ~10.7 s, capped at 105 s, as in validate/stress.py) until
                 ball_missing (0.41, DATA.md) of the in-half frames have no ball. A run is
                 dead_weight times likelier to start on a dead-ball frame: on the public window the
                 ATD misses the ball on 34% of dead frames and ~6% of live ones.
  false ball     spike_rate (~1%: 2043 jumps > 40 m/s in 199319 ATD frames) of the remaining ball
                 frames are moved 10-40 m in a random direction -- what A1 cleans.
  players        each player disappears in gaps (exponential, mean player_gap_s) between visible
                 stretches (exponential, mean player_visible_s), which leaves ~10.2 of 11 visible;
                 a reappearing player comes back under a NEW slot, as the ATD's tracker does,
                 so identities fragment like the ATD's 46 slots for 22 players.
  jitter         gaussian noise of jitter_m on every observed position.

Deliberately not simulated yet (the plan's full A6 list): identity swaps between two crossing
players, duplicated players, static fake objects, Smart Tagging (external matches have none, so
every tagging-based cue is silent on them). The KS comparison of simulator vs ATD statistics is
also still to do. Everything is seeded: same seed, same degraded match.
"""

import copy
import heapq

import numpy as np

from pompeu.io.contract import SRC_ATD, SRC_NONE, Tracking

# Params
PARAMS = {
    "ball_missing": 0.41,
    "max_gap_s": 105.0,
    "dead_weight": 5.0,
    "spike_rate": 0.01,
    "spike_m": (10.0, 40.0),
    # Exponential stretches: mean 19.6 s gives the ATD's median tracklet (13.6 s = 19.6 * ln 2);
    # mean gap 1.5 s then leaves ~7% of player-frames unseen, ~10.2 of 11 visible per team.
    "player_visible_s": 19.6,
    "player_gap_s": 1.5,
    "jitter_m": 0.05,
}


def _settings(params):
    out = copy.deepcopy(PARAMS)
    unknown = set(params or {}) - set(out)
    if unknown:
        raise KeyError("unknown degrade params {}".format(sorted(unknown)))
    out.update(params or {})
    return out


def erase_ball(present, dead, run_lengths, target, dead_weight, rng):
    """Boolean mask of ball frames to erase so that `target` of the frames have no ball."""
    n = len(present)
    missing = ~present.copy()
    erase = np.zeros(n, dtype=bool)
    goal = int(round(target * n))
    weights = np.where(dead, dead_weight, 1.0)
    weights = weights / weights.sum()
    while missing.sum() < goal:
        length = int(rng.choice(run_lengths))
        start = int(rng.choice(n, p=weights)) - length // 2
        span = np.arange(max(start, 0), min(start + length, n))
        new = span[~missing[span]][:goal - int(missing.sum())]
        missing[new] = True
        erase[new] = True
    return erase


def fragment_players(observed, fps, p, rng):
    """[(source slot, (n,) bool mask)]: each player's frames cut into tracklets with gaps.

    Each tracklet becomes its own slot in the degraded match, as the ATD's tracker gives a
    re-acquired player a new slot.
    """
    n, k = observed.shape
    out = []
    for slot in range(k):
        t = 0
        while t < n:
            stay = max(1, int(rng.exponential(p["player_visible_s"] * fps)))
            gap = max(1, int(rng.exponential(p["player_gap_s"] * fps)))
            mask = np.zeros(n, dtype=bool)
            mask[t:min(t + stay, n)] = observed[t:min(t + stay, n), slot]
            if mask.any():
                out.append((slot, mask))
            t += stay + gap
    return out


def degrade(tracking, labels, seed, run_lengths, params=None):
    """A degraded copy of a clean external Tracking. labels (N,) only steer where the ball is lost."""
    p = _settings(params)
    rng = np.random.default_rng(seed)
    fps = tracking.meta.get("fps", 25)
    n = tracking.n_frames
    pitch = np.array(tracking.meta["pitch"], dtype=np.float64)

    # Ball.
    ball_xy = tracking.ball_xy.copy()
    present = tracking.ball_present.copy()
    lengths = run_lengths[run_lengths <= p["max_gap_s"] * fps]
    in_half = tracking.half() > 0
    erase = np.zeros(n, dtype=bool)
    rows = np.flatnonzero(in_half)
    erase[rows] = erase_ball(present[rows], labels[rows] == 2, lengths, p["ball_missing"],
                             p["dead_weight"], rng)
    ball_xy[erase] = np.nan
    left = np.flatnonzero(~np.isnan(ball_xy[:, 0]))
    spikes = rng.choice(left, size=int(round(p["spike_rate"] * len(left))), replace=False)
    angle = rng.uniform(0, 2 * np.pi, len(spikes))
    distance = rng.uniform(*p["spike_m"], len(spikes))
    ball_xy[spikes] += np.column_stack((np.cos(angle), np.sin(angle))) * distance[:, None]
    ball_xy[spikes] = np.clip(ball_xy[spikes], -5.0, pitch + 5.0)
    ball = ~np.isnan(ball_xy[:, 0])
    ball_xy[ball] += rng.normal(0, p["jitter_m"], (int(ball.sum()), 2))

    # Players: fragment into tracklets, then pack the tracklets of each team into reusable slots
    # (the earliest-free slot, interval partitioning) -- as the ATD's tracker reuses its 46
    # slots, so a slot carries different players over time and is never a person.
    team = tracking.players_team[0]
    fragments = fragment_players(tracking.players_present, fps, p, rng)
    placed, slot_team = [], []
    for side in (0, 1):
        mine = sorted(((np.flatnonzero(mask)[0], np.flatnonzero(mask)[-1] + 1, slot, mask)
                       for slot, mask in fragments if team[slot] == side), key=lambda f: f[0])
        free = []  # heap of (frame the slot is free again, slot index)
        for start, stop, slot, mask in mine:
            if free and free[0][0] <= start:
                _, column = heapq.heappop(free)
            else:
                column = len(slot_team)
                slot_team.append(side)
            heapq.heappush(free, (stop, column))
            placed.append((column, slot, mask))
    players_xy = np.full((n, len(slot_team), 2), np.nan)
    for column, slot, mask in placed:
        players_xy[mask, column] = tracking.players_xy[mask, slot]
    new_team = np.array(slot_team, dtype=np.int8)
    seen = ~np.isnan(players_xy[:, :, 0])
    players_xy[seen] += rng.normal(0, p["jitter_m"], (int(seen.sum()), 2))

    return Tracking(
        players_xy=players_xy,
        players_team=np.broadcast_to(new_team, seen.shape),
        players_conf=seen.astype(np.float32),
        ball_xy=ball_xy,
        ball_z=np.where(ball, tracking.ball_z, np.nan),
        ball_conf=ball.astype(np.float32),
        ball_source=np.where(ball, SRC_ATD, SRC_NONE).astype(np.int8),
        frame0=tracking.frame0,
        meta=dict(tracking.meta, source=tracking.meta["source"] + "+degraded:{}".format(seed)),
    )

