"""A3: the two-state HMM, the kick detector, a synthetic throw-in end to end, and the real match."""

import time

import numpy as np
import pandas as pd
import pytest

from pompeu.events import deadball
from pompeu.features.build import build_features
from pompeu.models import deadball_vote
from synthetic import FPS, make_tracking

FIRST_HALF = 20000


# -- HMM ---------------------------------------------------------------------------------------

def test_hmm_switches_on_sustained_evidence_only():
    llr = np.full(3000, -0.5)
    llr[1000:2000] = 0.5      # 40 s of mild "dead" evidence
    llr[2500] = 5.0           # a single-frame blip
    post, path = deadball.hmm(llr, 1 / 1000, 1 / 500)
    assert not path[:990].any() and path[1020:1990].all() and not path[2010:].any()
    assert post[1500] > 0.99 and post[500] < 0.01
    assert ((post >= 0) & (post <= 1)).all()


def test_hmm_matches_brute_force_on_a_short_chain():
    """Viterbi and the posterior against exhaustive enumeration of all 2^8 paths."""
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, 8)
    p_ld, p_dl = 0.2, 0.3
    best, best_score = None, -np.inf
    weight_dead = np.zeros(8)
    total = 0.0
    for code in range(2 ** 8):
        path = [(code >> i) & 1 for i in range(8)]
        score = np.log(0.5) + (llr[0] if path[0] else 0.0)
        for t in range(1, 8):
            a, b = path[t - 1], path[t]
            score += np.log({(0, 0): 1 - p_ld, (0, 1): p_ld, (1, 0): p_dl, (1, 1): 1 - p_dl}[a, b])
            score += llr[t] if b else 0.0
        if score > best_score:
            best, best_score = path, score
        total += np.exp(score)
        weight_dead += np.exp(score) * np.array(path)
    post, path = deadball.hmm(llr, p_ld, p_dl)
    assert path.astype(int).tolist() == best
    np.testing.assert_allclose(post, weight_dead / total, atol=1e-9)


# -- synthetic stoppages ----------------------------------------------------------------------

def oscillating(n, active, start_xy):
    """(n, K, 2) players running back and forth along x (~4 m/s) on `active` frames, nearly
    still (0.3 m/s) otherwise. Positions integrate the velocity, so nobody ever teleports."""
    t = np.arange(n) / FPS
    amplitude = np.where(active, 6.0, 0.4)
    phase = np.linspace(0, 2 * np.pi, len(start_xy), endpoint=False)
    velocity = amplitude[:, None] * np.cos(2 * np.pi * t[:, None] / 8.0 + phase[None])
    xy = np.repeat(np.asarray(start_xy, dtype=float)[None], n, axis=0)
    xy[..., 0] += np.cumsum(velocity, axis=0) / FPS
    return xy


def throw_in_scene(n=1500):
    """Home dribbles the ball over the touchline (crossing y = 68 at frame 63), the players slow
    down, the ball is gone until it is placed on the line (950-999) next to away player 4, who
    has been standing there all along, and thrown in at frame 1000; play speeds up again."""
    t = np.arange(n) / FPS
    amplitude = np.where((np.arange(n) >= 175) & (np.arange(n) < 1000), 0.4, 6.0)
    phase = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    velocity = amplitude[:, None] * np.cos(2 * np.pi * t[:, None] / 8.0 + phase[None])
    x = np.array([30, 40, 50, 60, 35, 45, 55, 65], dtype=float)[None] + np.cumsum(velocity, 0) / FPS
    y = np.tile(np.array([20, 30, 40, 50, 25, 35, 45, 55], dtype=float), (n, 1))
    players = np.stack((x, y), axis=-1)

    ball = np.full((n, 2), np.nan)
    ball[:175, 0] = 40.0
    ball[:175, 1] = 63.0 + 2.0 * t[:175]                 # crosses 68 at t = 2.5 s (frame 63)
    players[:63, 0] = ball[:63] - (0.0, 0.5)            # home player 0 dribbles it out ...
    walk = np.clip(np.arange(n - 63) / FPS * 0.3, 0, 10)
    players[63:, 0, 0] = 40.0
    players[63:, 0, 1] = 67.5 - walk                    # ... and walks back slowly
    players[:, 4] = (41.0, 68.3)                        # away player 4 waits by the line
    ball[950:1000] = (40.0, 68.0)                       # where the ball is placed
    ball[1000:, 0] = 40.0
    ball[1000:, 1] = 68.0 - 12.0 * (t[1000:] - t[1000])  # thrown in at 12 m/s
    ball[1000:, 1] = np.maximum(ball[1000:, 1], 20.0)
    team = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return make_tracking(players, team, ball, FIRST_HALF)


def test_kick_detected_at_start_of_motion():
    tr = throw_in_scene()
    F = build_features(tr, tagging=None, with_windows=False, groups=deadball.FEATURE_GROUPS)
    kicks = deadball.kick_frames(F, FPS, deadball.PARAMS)
    assert any(abs(k - 1000) <= 2 for k in kicks)


def test_kick_needs_a_player_near():
    tr = throw_in_scene()
    xy = tr.players_xy.copy()
    xy[:, 4] = (10.0, 10.0)  # nobody next to the ball when it moves
    F = build_features(make_tracking(xy, tr.players_team[0], tr.ball_xy, FIRST_HALF),
                       tagging=None, with_windows=False, groups=deadball.FEATURE_GROUPS)
    kicks = deadball.kick_frames(F, FPS, deadball.PARAMS)
    assert not any(abs(k - 1000) <= 2 for k in kicks)


def test_throw_in_end_to_end():
    result = deadball.detect(throw_in_scene(), tags=None)
    assert len(result.segments) == 1
    start, stop = result.segments[0]
    assert abs(start - (FIRST_HALF + 63)) <= 2       # the ball crossing the line
    assert abs(stop - (FIRST_HALF + 1000)) <= 2      # the throw
    restart = result.restarts.iloc[0]
    assert restart["type"] == "throw_in" and restart["exit"] == "throw_in"
    assert restart["team"] == 1                      # home touched it last: away throws
    assert restart["confidence"] >= 0.8              # geometry and kicker agree
    assert result.p_dead[500] > 0.99 and result.p_dead[1300] < 0.01


def test_deadball_vote_labels():
    tr = throw_in_scene()
    ball = tr.ball_xy.copy()
    ball[1010:] = np.nan  # after the throw nobody sees the ball again
    tr = make_tracking(tr.players_xy, tr.players_team[0], ball, FIRST_HALF)
    labels = deadball_vote.predict(tr, window=50)
    assert set(labels[:60]) == {"H"}                 # home dribbling
    assert (labels[70:995] == "D").all()             # the stoppage, ball visible or not
    assert labels[1000] == "A"                       # away throws in ...
    assert (labels[1010:] == "A").all()              # ... and keeps it while the ball is unseen


def test_vote_in_stretch():
    codes = np.array([-1, -1, 0, 0, 1, 1, 1, -1, -1, -1, -1])
    np.testing.assert_array_equal(deadball_vote.vote_in_stretch(codes, 1),
                                  [-1, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1])


def test_goal_kick_vs_corner_by_last_touch():
    """Ball over the goal line home defends in the 1st half (x = 0), outside the posts.

    Slot 0 stands next to where it goes out; slot 7 takes the restart from the six-yard box.
    Home touched it last -> corner for away; away touched it last -> goal kick for home.
    """
    n = 1500
    t = np.arange(n) / FPS
    frames = np.arange(n)
    players = oscillating(n, (frames < 75) | (frames >= 950),
                          [(1.5, 50.5), (30, 20), (40, 30), (50, 40), (35, 25), (45, 35),
                           (55, 45), (3.5, 30)])
    players[:, 0] = (1.5, 50.5)
    players[:, 7] = (3.5, 30.0)
    ball = np.full((n, 2), np.nan)
    ball[:150] = np.column_stack((6.0 - 3.0 * t[:150], np.full(150, 50.0)))  # out at frame 50
    ball[900:950] = (4.0, 30.0)
    ball[950:, 0] = np.minimum(4.0 + 15.0 * (t[950:] - t[950]), 60.0)
    ball[950:, 1] = 30.0
    for toucher_team, (kind, restart_team) in ((0, ("corner", 1)), (1, ("goal_kick", 0))):
        team = np.array([toucher_team, 0, 0, 0, 1, 1, 1, restart_team])
        result = deadball.detect(make_tracking(players, team, ball, FIRST_HALF), tags=None)
        assert len(result.restarts) == 1
        restart = result.restarts.iloc[0]
        assert abs(restart["start"] - (FIRST_HALF + 51)) <= 2
        assert abs(restart["frame"] - (FIRST_HALF + 950)) <= 2
        assert restart["exit"] == kind and restart["type"] == kind
        assert restart["team"] == restart_team


def test_unknown_params_rejected():
    with pytest.raises(KeyError, match="unknown deadball params"):
        deadball.detect(throw_in_scene(), params={"out_lr": 3}, tags=None)


# -- the real match ----------------------------------------------------------------------------

def test_real_match(tracking):
    from pompeu.clean import clean_tracking
    clean, _ = clean_tracking(tracking)
    started = time.time()
    result = deadball.detect(clean)
    assert time.time() - started < 120
    graded = clean.half() > 0
    share = result.dead[graded].mean()
    assert 0.35 <= share <= 0.50                     # the plan expects 35-45% dead
    assert 60 <= len(result.segments) <= 160
    assert set(result.restarts["type"]) <= set(deadball.TYPES)
    assert set(result.restarts["team"]) <= {-1, 0, 1}
    assert (result.restarts["team"] >= 0).mean() > 0.5
    # The 2nd half is graded from 110808 but kicks off at 113742: all of that is dead.
    assert result.dead[111000:113700].mean() > 0.95
    assert isinstance(result.restarts, pd.DataFrame)
