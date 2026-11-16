"""scripts/train_gap_team_aug.py: pitch mirroring and home/away swapping of the gap-team features,
on synthetic Trackings (no data)."""

import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pompeu.experts import gap_team
from pompeu.features import collective, kinematics
from pompeu.io.contract import SRC_NONE, Tracking

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import train_gap_team_aug as aug  # noqa: E402

FPS = 25
PITCH = (105.0, 68.0)
H, A = 0, 1


def _tracking(players_xy, team, home_attack_sign=1):
    n, k = players_xy.shape[:2]
    return Tracking(players_xy=players_xy, players_team=np.broadcast_to(team, (n, k)),
                    players_conf=(~np.isnan(players_xy[..., 0])).astype(float),
                    ball_xy=np.full((n, 2), np.nan), ball_z=np.full(n, np.nan),
                    ball_conf=np.zeros(n), ball_source=np.full(n, SRC_NONE),
                    meta={"game": "SYN", "fps": FPS, "pitch": PITCH,
                          "home_attack_sign": home_attack_sign,
                          "halves": [(1, 0, n // 2 - 1), (2, n // 2, n - 1)]})


def _random_match(n=80, k=22, seed=0):
    rng = np.random.default_rng(seed)
    start = rng.uniform((5, 3), (100, 65), size=(k, 2))
    steps = rng.normal(0, 0.15, size=(n, k, 2))
    xy = start[None] + np.cumsum(steps, axis=0)
    xy[rng.random((n, k)) < 0.05] = np.nan                     # some unobserved players
    team = np.array([H] * (k // 2) + [A] * (k - k // 2))
    return _tracking(xy, team)


def _collective(tracking):
    players = kinematics.player_kinematics(tracking)
    return pd.DataFrame(collective.collective_features(tracking, players["vel"], players["speed"]))


def _assert_frames_equal(left, right):
    assert list(left.columns) == list(right.columns)
    for c in left.columns:
        np.testing.assert_allclose(left[c].to_numpy(float), right[c].to_numpy(float),
                                   atol=1e-9, equal_nan=True, err_msg=c)


def test_mirror_y_matches_features_of_the_mirrored_match():
    tr = _random_match()
    xy = tr.players_xy.copy()
    xy[..., 1] = PITCH[1] - xy[..., 1]
    mirrored = replace(tr, players_xy=xy)
    _assert_frames_equal(aug.mirror_y(_collective(tr), PITCH[1]), _collective(mirrored))


def test_swap_matches_features_of_the_relabelled_match():
    # The same players in the same places; who is called home changes, and so does
    # home_attack_sign, so every physical team keeps attacking the same goal.
    tr = _random_match(seed=1)
    swapped = replace(tr, players_team=1 - tr.players_team,
                      meta=dict(tr.meta, home_attack_sign=-tr.meta["home_attack_sign"]))
    _assert_frames_equal(aug.swap_teams(_collective(tr)), _collective(swapped))
    both = replace(swapped, players_xy=np.stack(
        [tr.players_xy[..., 0], PITCH[1] - tr.players_xy[..., 1]], axis=-1))
    _assert_frames_equal(aug.augment(_collective(tr), PITCH[1], "swap_mirror_y")[0],
                         _collective(both))


def test_hand_computed_values():
    # Two frames, first half (home attacks +x). Home: (60, 10), (70, 30); away: (20, 50), (40, 60).
    n = 4
    pos = np.array([[60, 10], [70, 30], [20, 50], [40, 60]], float)
    tr = _tracking(np.repeat(pos[None], n, axis=0), np.array([H, H, A, A]))
    F = _collective(tr)
    # Away attacks -x in the 1st half: its frame is rotated, x -> 105 - x, y -> 68 - y.
    assert F["cx_H"][0] == pytest.approx(65.0) and F["cy_H"][0] == pytest.approx(20.0)
    assert F["cx_A"][0] == pytest.approx(75.0) and F["cy_A"][0] == pytest.approx(13.0)
    assert F["line_gap"][0] == pytest.approx(70.0 - (105.0 - 85.0))   # def_line_H - (L - front_A)

    M = aug.mirror_y(F, PITCH[1])
    assert M["cy_H"][0] == pytest.approx(48.0) and M["cy_A"][0] == pytest.approx(55.0)
    assert M["cx_H"][0] == pytest.approx(65.0) and M["width_H"][0] == F["width_H"][0]

    S = aug.swap_teams(F)
    assert S["cx_H"][0] == pytest.approx(75.0) and S["cx_A"][0] == pytest.approx(65.0)
    assert S["centroid_gap"][0] == pytest.approx(F["centroid_gap"][0])      # cx_H + cx_A - L
    # def_line_A = 85 (second deepest of 65, 85), front_H = 70: 85 - (105 - 70)
    assert S["line_gap"][0] == pytest.approx(85.0 - (105.0 - 70.0))
    assert S["field_speed_mean"][0] == F["field_speed_mean"][0]


def test_merge_conventions_equals_three_copies():
    X_ = 3
    gts = np.array([[H, A, X_, H, X_], [H, A, A, X_, X_], [A, A, A, X_, X_]])
    keep, ybar, weight = aug.merge_conventions(gts, H, A)
    np.testing.assert_array_equal(keep, [True, True, True, True, False])
    np.testing.assert_allclose(ybar[:4], [1 / 3, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(weight, [3, 3, 2, 1, 0])

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(40, 3))
    y3 = (rng.random((3, 40)) < 1 / (1 + np.exp(-Z[:, 0]))).astype(float)
    w_copy, b_copy = gap_team.fit_logistic(np.tile(Z, (3, 1)), y3.ravel())
    w_merge, b_merge = aug.fit_logistic_weighted(Z, y3.mean(axis=0), np.full(40, 3.0))
    np.testing.assert_allclose(w_merge, w_copy, atol=1e-6)
    assert b_merge == pytest.approx(b_copy, abs=1e-6)


def _scene(rng, n, away_has_it):
    """Collective columns of n frames where the possessing team is up the pitch and moving on."""
    s = np.where(away_has_it, 1.0, -1.0)
    cols = {c: rng.normal(0, 1, n) for c in gap_team.BASE_COLUMNS}
    cols["cx_A"] = 50 + 8 * s + rng.normal(0, 4, n)
    cols["cx_H"] = 50 - 8 * s + rng.normal(0, 4, n)
    cols["cvx_A"] = 0.8 * s + rng.normal(0, 0.5, n)
    cols["cvx_H"] = -0.8 * s + rng.normal(0, 0.5, n)
    cols["cy_A"] = 34 + rng.normal(0, 6, n)
    cols["cy_H"] = 34 + 3 * s + rng.normal(0, 6, n)            # a home-side asymmetry to undo
    return pd.DataFrame(cols)


def test_mirrored_and_swapped_scene_gives_opposite_logit():
    rng = np.random.default_rng(3)
    away = rng.random(4000) < 0.4                              # unbalanced, as home advantage
    F = _scene(rng, len(away), away)
    Xs, ys = [], []
    for variant in aug.VARIANTS:
        Fv, swapped = aug.augment(F, PITCH[1], variant)
        Xs.append(gap_team.design_matrix(Fv, 1))
        ys.append((~away if swapped else away).astype(float))
    Z, mean, std = gap_team.standardise(np.concatenate(Xs))
    w, b = aug.fit_logistic_weighted(Z, np.concatenate(ys), np.ones(len(Z)))
    model = {"base_columns": list(gap_team.BASE_COLUMNS), "window_s": gap_team.WINDOW_S,
             "features": gap_team.feature_names(), "mean": list(mean), "std": list(std),
             "coef": list(w), "intercept": b}
    assert abs(b) < 0.05                                        # swap balances the classes

    test = _scene(np.random.default_rng(4), 200, rng.random(200) < 0.5)
    logit = gap_team.away_logit(test, 1, model)
    flipped = gap_team.away_logit(aug.augment(test, PITCH[1], "swap_mirror_y")[0], 1, model)
    strong = np.abs(logit) > 0.5
    assert strong.mean() > 0.8
    assert np.all(np.sign(flipped[strong]) == -np.sign(logit[strong]))
    np.testing.assert_allclose(flipped, -logit, atol=0.1 * np.abs(logit).max())
