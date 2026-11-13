"""pompeu.experts.gap_team and its HSMM hook, on synthetic inputs."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from pompeu import paths
from pompeu.experts import gap_team
from pompeu.models import hsmm

FPS = 25


class _Stub:
    """Just what team_evidence reads from a Tracking."""

    def __init__(self, half, ball_present):
        self._half = np.asarray(half, np.int8)
        self.ball_present = np.asarray(ball_present, bool)
        self.meta = {"fps": FPS}

    def half(self):
        return self._half


def _model(coef_by_feature, base_columns=("cx_A", "cx_H"), window_s=0.04, intercept=5.0):
    features = gap_team.feature_names(base_columns, window_s)
    return {"base_columns": list(base_columns), "window_s": window_s, "features": features,
            "mean": [0.0] * len(features), "std": [1.0] * len(features),
            "coef": [coef_by_feature.get(f, 0.0) for f in features], "intercept": intercept}


def test_feature_names_and_design_matrix_centred_mean():
    F = pd.DataFrame({"a": [0.0, 3.0, 6.0, np.nan, 12.0]})
    X = gap_team.design_matrix(F, fps=1, base_columns=("a", "missing"), window_s=3)
    assert gap_team.feature_names(("a", "missing"), 3) == ["a", "missing", "a_m3s", "missing_m3s"]
    assert X.shape == (5, 4)
    assert np.isnan(X[:, 1]).all() and np.isnan(X[:, 3]).all()        # absent column: NaN
    np.testing.assert_allclose(X[:, 2], [1.5, 3.0, 4.5, 9.0, 12.0])    # centred, NaN skipped
    assert len(gap_team.BASE_COLUMNS) == 31 and len(gap_team.feature_names()) == 62


def test_sign_home_positive_and_zero_where_ball_or_outside_halves():
    n = 6
    F = pd.DataFrame({"cx_A": [2.0, 2.0, -2.0, -2.0, 2.0, 2.0], "cx_H": np.zeros(n)})
    tr = _Stub(half=[1, 1, 1, 1, 1, 0], ball_present=[False, False, False, False, True, False])
    ev = gap_team.team_evidence(tr, F, _model({"cx_A": 1.0}))
    # away pushed up -> P(away) up -> evidence negative (home-positive, like hsmm's L_team);
    # the intercept is not used.
    np.testing.assert_allclose(ev[:4], [-2.0, -2.0, 2.0, 2.0])
    assert ev[4] == 0.0 and ev[5] == 0.0                              # ball seen / outside half


def test_nan_safe():
    F = pd.DataFrame({"cx_A": [np.nan, 3.0, np.nan], "cx_H": [np.nan, np.nan, 1.0]})
    tr = _Stub(half=[1, 1, 1], ball_present=[False, False, False])
    ev = gap_team.team_evidence(tr, F, _model({"cx_A": 1.0, "cx_H": -1.0}))
    assert np.all(np.isfinite(ev))
    assert ev[0] == 0.0                     # every feature missing: no evidence at all
    np.testing.assert_allclose(ev[1:], [-3.0, 1.0])     # a missing feature contributes nothing
    # A frame with no collective columns at all in F.
    ev = gap_team.team_evidence(tr, pd.DataFrame(index=range(3)), _model({"cx_A": 1.0}))
    assert np.all(ev == 0.0)


def test_load_model_needs_a_path():
    with pytest.raises(ValueError):
        gap_team.load_model(None)


def test_logistic_recovers_sign():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    y = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(float)
    Z, _, _ = gap_team.standardise(X)
    w, _ = gap_team.fit_logistic(Z, y)
    assert w[0] > 0 and w[1] < 0


def test_gap_team_evidence_never_moves_dead_versus_live():
    rng = np.random.default_rng(1)
    n = 40
    dead = rng.normal(0, 2, n)
    team = {k: rng.normal(0, 3, n) for k in ("proximity", "events", "tagging", "restart",
                                             "gap_team")}
    e = hsmm.emissions(dead, team, hsmm._settings(None))
    live = np.logaddexp(e[:, hsmm.HL], e[:, hsmm.AL])
    stopped = np.logaddexp(e[:, hsmm.HD], e[:, hsmm.AD])
    np.testing.assert_allclose(live - stopped, -dead, atol=1e-9)
    base = hsmm.emissions(dead, {k: v for k, v in team.items() if k != "gap_team"},
                          hsmm._settings(None))
    np.testing.assert_allclose((e - base)[:, hsmm.HL] - (e - base)[:, hsmm.AL], team["gap_team"])


def test_hsmm_hook_off_by_default_and_runs_when_on():
    from test_deadball import throw_in_scene
    tr = throw_in_scene()
    prep = hsmm.prepare(tr)
    default = hsmm.decode_prepared(prep)["labels"]
    off = hsmm.decode_prepared(prep, {"gap_team_weight": 0.0, "gap_team_model": "nope.json"})
    np.testing.assert_array_equal(default, off["labels"])
    assert "gap_team" not in off["evidence"]
    model = _model({"cx_H": 1.0}, window_s=gap_team.WINDOW_S)
    on = hsmm.decode_prepared(prep, {"gap_team_weight": 1.0, "gap_team_model": model})
    ev = on["evidence"]["gap_team"]
    assert ev.shape == (tr.n_frames,) and np.all(np.isfinite(ev))
    assert np.all(ev[np.asarray(tr.ball_present, bool)] == 0.0)
    assert set(on["labels"]) <= {"H", "A", "D"}


def test_shipped_model_is_consistent():
    path = os.path.join(paths.CONFIGS_DIR, "gap_team_v1.json")
    if not os.path.isfile(path):
        pytest.skip("configs/gap_team_v1.json not trained yet")
    with open(path, encoding="utf-8") as handle:
        model = json.load(handle)
    assert model["features"] == gap_team.feature_names(model["base_columns"], model["window_s"])
    assert len(model["coef"]) == len(model["mean"]) == len(model["std"]) == len(model["features"])
    assert all(m.startswith("dfl:") for m in model["training"]["matches"])
    assert model["training"]["split"] == "train"
