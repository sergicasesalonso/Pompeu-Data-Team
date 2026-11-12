"""A1: each cleaning step on synthetic cases where the right answer is known, then the real match."""

import numpy as np
import pytest

from pompeu.clean import ball, clean_tracking, normalize, players, stitching
from pompeu.features import kinematics
from pompeu.io.contract import SRC_INTERP
from synthetic import FPS, make_tracking


def rolling_ball(n=100, speed=10.0):
    """Ball rolling along x at `speed` m/s from (20, 30)."""
    t = np.arange(n) / FPS
    return np.column_stack((20 + speed * t, np.full(n, 30.0)))


# -- kinematics --------------------------------------------------------------------------------

def test_savgol_exact_on_constant_velocity_and_acceleration():
    t = np.arange(50) / FPS
    xy = np.column_stack((3 + 4 * t + 0.5 * 2 * t ** 2, 10 - 1.5 * t))
    velocity, acceleration = kinematics.savgol_kinematics(xy, FPS)
    np.testing.assert_allclose(velocity[:, 0], 4 + 2 * t, atol=1e-9)
    np.testing.assert_allclose(velocity[:, 1], -1.5, atol=1e-9)
    np.testing.assert_allclose(acceleration[:, 0], 2.0, atol=1e-6)


def test_savgol_never_crosses_a_gap():
    xy = rolling_ball(40)
    xy[20:25] = np.nan
    xy[25:] += 50.0  # a jump across the gap must not show up as a velocity
    velocity, _ = kinematics.savgol_kinematics(xy, FPS)
    assert np.isnan(velocity[20:25]).all()
    np.testing.assert_allclose(velocity[~np.isnan(velocity[:, 0]), 0], 10.0, atol=1e-9)


def test_short_runs():
    xy = np.full((10, 2), np.nan)
    xy[2] = (1, 1)
    xy[5:7] = [(0, 0), (0.4, 0)]
    velocity, _ = kinematics.savgol_kinematics(xy, FPS)
    assert np.isnan(velocity[2]).all()
    np.testing.assert_allclose(velocity[5:7, 0], 10.0)


# -- ball --------------------------------------------------------------------------------------

def test_spike_removed_and_neighbours_kept():
    xy = rolling_ball()
    xy[50] = (90, 5)
    removed = ball.remove_spikes(xy, FPS)
    assert removed.tolist().count(True) == 1 and removed[50]


def test_two_point_burst_removed_as_fragment():
    xy = rolling_ball()
    xy[50:52] = [(90, 5), (90.2, 5)]  # two consistent points far away: invisible to the spike rule
    assert not ball.remove_spikes(xy, FPS)[50:52].any()
    removed = ball.remove_short_fragments(xy, FPS)
    assert removed[50:52].all() and removed.sum() == 2


def test_isolated_point_surrounded_by_gaps():
    xy = rolling_ball()
    xy[40:60] = np.nan
    xy[50] = (40, 30)
    removed = ball.remove_short_fragments(xy, FPS)
    assert removed[50] and removed.sum() == 1


def test_short_gap_interpolated_long_gap_not():
    xy = rolling_ball(200)
    truth = xy.copy()
    xy[50:60] = np.nan     # 0.4 s: filled
    xy[100:120] = np.nan   # 0.8 s: left alone
    out, filled = ball.interpolate_gaps(xy, FPS)
    assert filled[50:60].all() and not filled[100:120].any()
    np.testing.assert_allclose(out[50:60], truth[50:60], atol=1e-9)
    assert np.isnan(out[100:120]).all()


def test_gap_not_bridged_when_impossibly_fast():
    xy = rolling_ball(100)
    xy[50:55] = np.nan
    xy[55:] += 60.0
    _, filled = ball.interpolate_gaps(xy, FPS)
    assert not filled.any()


# -- players -----------------------------------------------------------------------------------

def moving_players(n, k, seed=0):
    rng = np.random.default_rng(seed)
    start = rng.uniform(10, 90, (k, 2))
    velocity = rng.uniform(-3, 3, (k, 2))
    velocity[np.hypot(*velocity.T) < 1.5] = (2.0, 0.0)
    return start + velocity * (np.arange(n)[:, None, None] / FPS)


def test_frozen_flagged_only_while_others_move():
    n = 1000  # 40 s
    xy = moving_players(n, 6)
    xy[200:600, 0] = (5.0, 34.0)        # held for 16 s while everyone runs
    frozen = players.frozen_mask(xy, FPS)
    assert frozen[250:550, 0].all()
    assert not frozen[:, 1:].any()
    assert not frozen[700:, 0].any()

    still = np.repeat(xy[:1], n, axis=0)  # nobody moves at all: a stoppage, not a frozen slot
    assert not players.frozen_mask(still, FPS).any()


def test_duplicate_shorter_track_dropped():
    xy = moving_players(300, 3)
    xy[:, 1] = xy[:, 0] + 0.2       # slot 1 shadows slot 0 ...
    xy[:100, 1] = np.nan            # ... but its track is shorter
    xy[:, 2] = xy[:, 0] + 0.1       # slot 2 is as close, but on the other team
    drop, stretches = players.duplicate_mask(xy, np.array([0, 0, 1]), FPS)
    assert stretches == 1
    assert drop[100:, 1].all() and not drop[:, 0].any() and not drop[:, 2].any()


def test_stitching_links_and_repacks():
    n = 300
    xy = np.full((n, 3, 2), np.nan)
    path = moving_players(n, 1)[:, 0]
    xy[:100, 0] = path[:100]             # player seen as slot 0 ...
    xy[110:, 1] = path[110:]             # ... lost 10 frames, then reacquired as slot 1
    xy[110:, 2] = path[110:] + (20, 0)   # someone else, too far away to be the same player
    conf = (~np.isnan(xy[:, :, 0])).astype(np.float32)
    new_xy, new_conf, team, report = stitching.stitch(xy, conf, np.array([0, 0, 0]), FPS)
    assert report["links"] == 1 and len(team) == 2
    joined = [s for s in range(2) if not np.isnan(new_xy[0, s, 0])][0]
    np.testing.assert_allclose(new_xy[:, joined], path, atol=1e-6)  # gap filled on the line
    assert (new_conf[100:110, joined] == stitching.FILL_CONF).all()


def test_stitching_respects_team_and_picks_closest():
    n = 200
    xy = np.full((n, 3, 2), np.nan)
    xy[:100, 0] = (50.0, 30.0)
    xy[105:, 1] = (51.0, 30.0)   # same team, 1 m away
    xy[105:, 2] = (50.2, 30.0)   # closer, but the other team
    pieces = stitching.tracklets(~np.isnan(xy[:, :, 0]))
    links = stitching.link(xy, np.array([0, 0, 1]), pieces, FPS)
    assert {pieces[i][0]: pieces[j][0] for i, j in links.items()} == {0: 1}


# -- normalize ---------------------------------------------------------------------------------

def test_normalize_flips_per_half():
    frames = np.array([11185, 50000, 110808, 150000])
    n = 200000
    xy = np.full((n, 1, 2), np.nan)
    xy[frames, 0] = (10.0, 20.0)
    tr = make_tracking(xy, np.array([0]), np.full((n, 2), np.nan))
    home = normalize.attack_flip(tr, 0)[frames]
    away = normalize.attack_flip(tr, 1)[frames]
    assert home.tolist() == [False, False, True, True]  # Barcelona attacks +x in the 1st half
    assert (away == ~home).all()
    out = normalize.normalized(tr, 0)
    np.testing.assert_allclose(out.players_xy[frames[2], 0], (95.3, 48.0))
    np.testing.assert_allclose(out.players_xy[frames[0], 0], (10.0, 20.0))


# -- the whole step, synthetic and real --------------------------------------------------------

def test_clean_tracking_keeps_contract_and_marks_sources():
    n = 400
    xy = moving_players(n, 4)
    ball_xy = rolling_ball(n, 8.0)
    ball_xy[100] = (100, 2)        # spike
    ball_xy[200:205] = np.nan      # short gap
    tr = make_tracking(xy, np.array([0, 0, 1, 1]), ball_xy)
    clean, report = clean_tracking(tr)
    assert report["ball"]["spikes_removed"] == 1
    assert clean.ball_source[100] == SRC_INTERP   # the spike's frame becomes a 1-frame gap, filled
    assert (clean.ball_source[200:205] == SRC_INTERP).all()
    assert (clean.ball_conf[200:205] == 0.5).all()
    clean.validate()


def test_unknown_step_rejected():
    tr = make_tracking(moving_players(10, 2), np.array([0, 1]), rolling_ball(10))
    with pytest.raises(KeyError, match="unknown clean step"):
        clean_tracking(tr, {"balls": {}})


def test_real_match(tracking):
    clean, report = clean_tracking(tracking)
    clean.validate()
    assert report["ball"]["spikes_removed"] > 1500
    # Jumps > 40 m/s between consecutive frames: ~2000 before, a handful after.
    step = np.hypot(*(clean.ball_xy[1:] - clean.ball_xy[:-1]).T) * FPS
    assert np.nansum(step > 40) < 50
    # Goalkeepers are kept: slots 24 and 46 still observed on most frames.
    assert report["frozen"]["action"] == "flag"
    assert report["players"]["observations_after"] > 0.99 * report["players"]["observations_before"]
    assert report["stitching"]["slots_after"] < report["stitching"]["slots_before"]
