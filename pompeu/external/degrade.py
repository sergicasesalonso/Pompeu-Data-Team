"""A6: make clean external tracking look like our ATD before any method sees it.

A method validated on clean DFL tracking would be validated on the wrong problem: our ATD loses
the ball on 41% of graded frames, jumps it to false detections, and sees ~10.2 of 11 players per
team in fragments that keep changing slot. Each degradation reproduces something measured on our
ATD (scripts/degradation_fit.py compares the two, statistic by statistic):

  ball gaps      the ball is erased in runs whose lengths are drawn from the ATD's own no-ball
                 runs (median ~10.7 s, capped at 105 s, as in validate/stress.py) until
                 ball_missing (0.41, DATA.md) of the in-half frames have no ball -- frames already
                 missing (Metrica's ball is) count towards it. A run is dead_weight times likelier
                 to start on a dead-ball frame: on the public window the ATD misses the ball on 34%
                 of dead frames and ~6% of live ones.
  false ball     spike_rate (~1%: 2043 jumps > 40 m/s in 199319 ATD frames) of the remaining ball
                 frames are moved 10-40 m in a random direction -- what A1 cleans.
  identity swaps when two teammates come within swap_m of each other, with probability p_swap
                 their trajectories swap slots from then on -- the tracker follows the wrong one.
  frozen players a player more than freeze_far_m from the ball for over freeze_min_s is, with
                 probability p_freeze, held at the position where the stretch began: the ATD holds
                 a player it cannot see (A1 found it, mostly goalkeepers, ~1% of player-frames).
  tracklets      each trajectory disappears in gaps (exponential, mean player_gap_s) between
                 heavy-tailed visible stretches (log-normal, median player_visible_s).
  group losses   the players of one team furthest from the ball vanish together for a few
                 seconds -- the ATD loses players in clusters, not one by one.
  duplicates     ghost copies of a visible player, ~2 m off, 2-8 s long.
  slots          the pieces of each team are packed into reusable slots (earliest-free slot,
                 interval partitioning): a re-acquired player usually comes back under another
                 slot and a slot carries different players over time, like the ATD's 46.
  noise          ball: gaussian per frame (ball_jitter_m); players: a constant offset per tracklet
                 (tracklet_bias_m) -- the ATD's player positions are smooth.

Not simulated: Smart Tagging (external matches have none, so every tagging cue is silent on them)
and team errors (a player given the wrong team). Everything is seeded: same seed, same match.
"""

import copy
import heapq

import numpy as np

from pompeu.features.kinematics import runs
from pompeu.io.contract import SRC_ATD, SRC_NONE, Tracking

# Params
PARAMS = {
    "ball_missing": 0.41,
    "max_gap_s": 105.0,
    "dead_weight": 5.0,
    "spike_rate": 0.0012,      # 2 jumps > 40 m/s per spike: the ATD's 2.3 per 1000 graded frames
    "spike_m": (10.0, 40.0),
    "swap_m": 1.0,
    "p_swap": 0.3,
    "freeze_far_m": 45.0,
    "freeze_min_s": 3.0,
    "p_freeze": 0.12,          # gives A1's frozen detector the ATD's 1.1% of player-frames
    # Independent tracklet gaps: visible stretches log-normal (median player_visible_s, spread
    # player_visible_sigma) -- the ATD's tracklets are heavy-tailed, p25/50/75 = 8/35/145 s in
    # play, some players followed for minutes -- and gaps exponential (mean player_gap_s).
    "player_visible_s": 90.0,
    "player_visible_sigma": 1.4,
    "player_gap_s": 6.0,         # ~2.4% of each player's frames: 10 visible on ~22% of frames
    # Group losses: the 1-3 players of one team furthest from the ball vanish together for 1-10 s
    # -- with the gaps above, the ATD's 6-9-visible tail (~10% of frames per team).
    "group_dropouts_per_hour": 72.0,
    "group_size": (1, 3),
    "group_s": (1.0, 10.0),
    # Ghost copies ~2 m off: 12 visible on ~6% of frames, yet only ~30 an hour close enough
    # (< 0.5 m for > 1 s) for A1's duplicate detector -- both as on the ATD.
    "duplicates_per_hour": 200.0,
    "duplicate_s": (2.0, 8.0),
    "duplicate_offset_m": 2.0,
    # The ATD's ball does carry per-frame noise: its step-speed distribution needs ~5 cm.
    "ball_jitter_m": 0.05,
    # A constant offset per tracklet, not per-frame noise: the ATD's positions are smooth (per-
    # frame noise of 5 cm would add ~1.8 m/s of fake speed to every player).
    "tracklet_bias_m": 0.1,
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


def swap_identities(xy, team, p, rng):
    """Copy of xy (n, k, 2) where crossing teammates sometimes exchange slots for good.

    Crossings are found on the true trajectories; each accepted one swaps which person the two
    slots follow from that frame on. Returns (xy, number of swaps).
    """
    n, k = xy.shape[:2]
    events = []
    for a in range(k):
        for b in range(a + 1, k):
            if team[a] != team[b]:
                continue
            with np.errstate(invalid="ignore"):
                close = np.hypot(*(xy[:, a] - xy[:, b]).T) < p["swap_m"]
            starts, _ = runs(close)
            for t in starts:
                if rng.random() < p["p_swap"]:
                    events.append((int(t), a, b))
    events.sort()
    person = np.arange(k)              # person[slot] = whose trajectory the slot shows
    out = np.empty_like(xy)
    t0 = 0
    for t, a, b in events + [(n, -1, -1)]:
        out[t0:t] = xy[t0:t][:, person]
        if a >= 0:
            # The persons a and b are now shown by each other's slot.
            sa, sb = int(np.flatnonzero(person == a)[0]), int(np.flatnonzero(person == b)[0])
            person[sa], person[sb] = person[sb], person[sa]
        t0 = t
    return out, len(events)


def freeze_far_players(xy, ball_xy, fps, p, rng):
    """Copy of xy where players far from the ball are sometimes held still.

    Returns (xy, held (n, k) bool). Held positions are EXACTLY constant -- no jitter is added to
    them later -- because that is what the ATD does, and what A1's frozen detector looks for.
    """
    out = xy.copy()
    held = np.zeros(xy.shape[:2], dtype=bool)
    need = int(round(p["freeze_min_s"] * fps))
    distance = np.hypot(xy[:, :, 0] - ball_xy[:, None, 0], xy[:, :, 1] - ball_xy[:, None, 1])
    for slot in range(xy.shape[1]):
        with np.errstate(invalid="ignore"):
            far = distance[:, slot] > p["freeze_far_m"]
        for start, stop in zip(*runs(far)):
            if stop - start >= need and rng.random() < p["p_freeze"]:
                seen = ~np.isnan(out[start:stop, slot, 0])
                out[start:stop, slot][seen] = xy[start, slot]
                held[start:stop, slot] = seen
    return out, held


def group_dropouts(observed, xy, team, ball_xy, fps, p, rng):
    """Copy of observed (n, k) with clusters of far-from-the-ball players of one team hidden."""
    n = observed.shape[0]
    out = observed.copy()
    distance = np.hypot(xy[:, :, 0] - ball_xy[:, None, 0], xy[:, :, 1] - ball_xy[:, None, 1])
    for side in (0, 1):
        mine = np.flatnonzero(team == side)
        for _ in range(rng.poisson(p["group_dropouts_per_hour"] * n / (fps * 3600.0))):
            start = int(rng.integers(0, n))
            stop = min(n, start + int(rng.uniform(*p["group_s"]) * fps))
            size = int(rng.integers(p["group_size"][0], p["group_size"][1] + 1))
            far = np.nan_to_num(distance[start, mine], nan=-1.0)
            hidden = mine[np.argsort(-far)[:size]]
            out[start:stop, hidden] = False
    return out


def fragment_players(observed, fps, p, rng):
    """[(source slot, (n,) bool mask, offset (2,))]: trajectories cut into tracklets with gaps,
    plus ghost duplicates (offset != 0)."""
    n, k = observed.shape
    out = []
    zero = np.zeros(2)
    for slot in range(k):
        t = 0
        while t < n:
            stay = max(1, int(rng.lognormal(np.log(p["player_visible_s"] * fps),
                                            p["player_visible_sigma"])))
            gap = max(1, int(rng.exponential(p["player_gap_s"] * fps)))
            mask = np.zeros(n, dtype=bool)
            mask[t:min(t + stay, n)] = observed[t:min(t + stay, n), slot]
            if mask.any():
                out.append((slot, mask, zero))
            t += stay + gap
    count = rng.poisson(p["duplicates_per_hour"] * n / (fps * 3600.0))
    for _ in range(count):
        slot = int(rng.integers(k))
        length = int(rng.uniform(*p["duplicate_s"]) * fps)
        start = int(rng.integers(0, max(1, n - length)))
        mask = np.zeros(n, dtype=bool)
        mask[start:start + length] = observed[start:start + length, slot]
        if mask.any():
            out.append((slot, mask, rng.normal(0, p["duplicate_offset_m"], 2)))
    return out


def pack(fragments, team):
    """Place fragments into reusable slots per team. Returns (column of each fragment, teams)."""
    placed, slot_team = [], []
    for side in (0, 1):
        mine = sorted(((np.flatnonzero(mask)[0], np.flatnonzero(mask)[-1] + 1, index)
                       for index, (slot, mask, _) in enumerate(fragments) if team[slot] == side))
        free = []  # heap of (frame the slot is free again, column)
        for start, stop, index in mine:
            if free and free[0][0] <= start:
                _, column = heapq.heappop(free)
            else:
                column = len(slot_team)
                slot_team.append(side)
            heapq.heappush(free, (stop, column))
            placed.append((index, column))
    return placed, np.array(slot_team, dtype=np.int8)


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
    rows = np.flatnonzero(tracking.half() > 0)
    erase = np.zeros(n, dtype=bool)
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
    ball_xy[ball] += rng.normal(0, p["ball_jitter_m"], (int(ball.sum()), 2))

    # Players: swaps and holds on the full trajectories, then tracklets, ghosts and slots.
    team = tracking.players_team[0]
    xy, swaps = swap_identities(tracking.players_xy, team, p, rng)
    xy, held = freeze_far_players(xy, tracking.ball_xy, fps, p, rng)
    visible = group_dropouts(~np.isnan(xy[:, :, 0]), xy, team, tracking.ball_xy, fps, p, rng)
    fragments = fragment_players(visible, fps, p, rng)
    placed, new_team = pack(fragments, team)
    players_xy = np.full((n, len(new_team), 2), np.nan)
    frozen = np.zeros((n, len(new_team)), dtype=bool)
    for index, column in placed:
        slot, mask, offset = fragments[index]
        bias = rng.normal(0, p["tracklet_bias_m"], 2)
        players_xy[mask, column] = xy[mask, slot] + offset + np.where(held[mask, slot][:, None],
                                                                       0.0, bias)
        frozen[mask, column] = held[mask, slot] & (offset == 0).all()
    seen = ~np.isnan(players_xy[:, :, 0])

    meta = dict(tracking.meta, source=tracking.meta["source"] + "+degraded:{}".format(seed),
                degraded={"swaps": swaps, "frozen_frames": int(frozen.sum()),
                          "spikes": int(len(spikes))})
    return Tracking(
        players_xy=players_xy,
        players_team=np.broadcast_to(new_team, seen.shape),
        players_conf=seen.astype(np.float32),
        ball_xy=ball_xy,
        ball_z=np.where(ball, tracking.ball_z, np.nan),
        ball_conf=ball.astype(np.float32),
        ball_source=np.where(ball, SRC_ATD, SRC_NONE).astype(np.int8),
        frame0=tracking.frame0,
        meta=meta,
    )
