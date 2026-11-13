"""B8: broadcast-to-tactical alignment on a synthetic match with a cut and a replay."""

import numpy as np

from pompeu.io import broadcast
from synthetic import make_tracking

OFFSET = 500          # tactical frame = broadcast frame + OFFSET while the broadcast is live


def _tactical(n=2600, seed=0):
    """22 players wandering smoothly over the pitch."""
    rng = np.random.default_rng(seed)
    k = 22
    start = rng.uniform([10, 8], [95, 60], size=(k, 2))
    vel = np.zeros((k, 2))
    xy = np.empty((n, k, 2))
    pos = start.copy()
    for i in range(n):
        vel = 0.95 * vel + rng.normal(0, 0.03, size=(k, 2))
        pos = np.clip(pos + vel, [1, 1], [104, 67])
        xy[i] = pos
    team = np.array([0] * 11 + [1] * 11)
    ball = np.full((n, 2), np.nan)
    return make_tracking(xy, team, ball)


def _shot(tac_xy, team, t0, length, rng):
    """A broadcast shot of tactical frames t0..t0+length: 10 visible players, new slot order."""
    k = tac_xy.shape[1]
    keep = rng.choice(k, size=10, replace=False)
    out = np.full((length, 14, 2), np.nan)
    teams = np.full(14, 0, dtype=np.int8)
    order = rng.permutation(14)[:10]
    out[:, order] = tac_xy[t0:t0 + length][:, keep] + rng.normal(0, 0.3, size=(length, 10, 2))
    teams[order] = team[keep]
    return out, teams


def _broadcast(tac):
    """Live 500..1199, cut to a replay of 800..999, cut back to live 1400..1999 (offset 500)."""
    rng = np.random.default_rng(1)
    team = np.asarray(tac.players_team[0])
    xy = tac.players_xy
    pieces = [_shot(xy, team, 500, 700, rng), _shot(xy, team, 800, 200, rng),
              _shot(xy, team, 1400, 600, rng)]
    # One slot layout for the whole recording, as in a real ATD: give each shot its own block of
    # slots so that every cut is a slot reset.
    n = sum(p[0].shape[0] for p in pieces)
    bxy = np.full((n, 14 * len(pieces), 2), np.nan)
    bteam = np.zeros(14 * len(pieces), dtype=np.int8)
    row = 0
    for j, (sxy, steam) in enumerate(pieces):
        bxy[row:row + len(sxy), 14 * j:14 * (j + 1)] = sxy
        bteam[14 * j:14 * (j + 1)] = steam
        row += len(sxy)
    return make_tracking(bxy, bteam, np.full((n, 2), np.nan))


def test_alignment_recovers_offset_and_drops_the_replay():
    tac = _tactical()
    bc = _broadcast(tac)
    valid = np.ones(bc.n_frames, dtype=bool)
    al = broadcast.align(tac, bc, valid, {"vote_every": 50})

    assert al.stats["replays_discarded"] == 1
    live = np.r_[np.arange(510, 1190), np.arange(1410, 1990)]
    # Live frames map back with the live offset (+/- a frame for the fine search).
    assert np.all(np.abs((live - al.b_of_t[live]) - OFFSET) <= 1)
    # Tactical frames shown only in the replay window, or never shown, stay unaligned.
    assert np.all(al.b_of_t[1250:1350] == -1)
    assert np.all(al.b_of_t[:400] == -1)
    assert np.nanmedian(al.fit_of_t[live]) < 1.0


def test_alignment_handles_flipped_coordinates():
    tac = _tactical(seed=2)
    bc = _broadcast(tac)
    bc.players_xy[..., 0] = 105.3 - bc.players_xy[..., 0]
    bc.players_xy[..., 1] = 68.0 - bc.players_xy[..., 1]
    al = broadcast.align(tac, bc, np.ones(bc.n_frames, dtype=bool), {"vote_every": 50})
    assert al.orientation[0] is True
    live = np.arange(1410, 1990)
    assert np.all(np.abs((live - al.b_of_t[live]) - OFFSET) <= 1)


def test_invalid_homography_frames_are_never_used():
    tac = _tactical(seed=3)
    bc = _broadcast(tac)
    valid = np.ones(bc.n_frames, dtype=bool)
    valid[100:300] = False
    al = broadcast.align(tac, bc, valid, {"vote_every": 50})
    used = al.b_of_t[al.b_of_t >= 0]
    assert not np.any((used >= 100) & (used < 300))
