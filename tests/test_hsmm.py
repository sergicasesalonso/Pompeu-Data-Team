"""A5: the semi-Markov kernels against brute force, then the HSMM model."""

import itertools

import numpy as np
import pytest

from pompeu.decode import viterbi as sm


def all_segmentations(n, n_states, d_max):
    """Every (segments) of [0, n): compositions of n into parts <= d_max, times state labels."""
    def compositions(rest):
        if rest == 0:
            yield []
            return
        for d in range(1, min(d_max, rest) + 1):
            for tail in compositions(rest - d):
                yield [d] + tail
    for parts in compositions(n):
        for states in itertools.product(range(n_states), repeat=len(parts)):
            segments, t = [], 0
            for d, s in zip(parts, states):
                segments.append((t, t + d, s))
                t += d
            yield segments


def score(segments, emissions, log_dur, trans, start):
    total = start[segments[0][2]]
    for k, (a, b, s) in enumerate(segments):
        total += emissions[a:b, s].sum() + log_dur[s, b - a]
        if k:
            total += trans[a, segments[k - 1][2], s]
    return total


def random_problem(seed, n=7, n_states=3, d_max=3):
    rng = np.random.default_rng(seed)
    emissions = rng.normal(0, 1.5, (n, n_states))
    log_dur = np.log(rng.dirichlet(np.ones(d_max), n_states))
    log_dur = np.concatenate((np.full((n_states, 1), -np.inf), log_dur), axis=1)
    log_dur[0, 1] = -np.inf  # state 0 cannot last a single frame
    trans = np.log(rng.dirichlet(np.ones(n_states), (n + 1, n_states)))
    start = np.log(rng.dirichlet(np.ones(n_states)))
    return emissions, log_dur, trans, start


@pytest.mark.parametrize("seed", range(5))
def test_viterbi_and_posteriors_match_brute_force(seed):
    emissions, log_dur, trans, start = random_problem(seed)
    n, n_states = emissions.shape
    best, best_score = None, -np.inf
    occupancy = np.zeros((n, n_states))
    weights = []
    for segments in all_segmentations(n, n_states, log_dur.shape[1] - 1):
        value = score(segments, emissions, log_dur, trans, start)
        if not np.isfinite(value):
            continue
        weights.append(value)
        if value > best_score:
            best, best_score = segments, value
        for a, b, s in segments:
            occupancy[a:b, s] += np.exp(value)
    log_z = np.logaddexp.reduce(weights)
    occupancy /= np.exp(log_z)

    E = sm.cumulative(emissions)
    states, segments, got = sm.viterbi(E, log_dur, trans, start)
    assert got == pytest.approx(best_score, abs=1e-9)
    assert segments == best
    post, got_z = sm.posteriors(E, log_dur, trans, start)
    assert got_z == pytest.approx(log_z, abs=1e-9)
    np.testing.assert_allclose(post, occupancy, atol=1e-9)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-9)


def test_impossible_problem_raises():
    emissions, log_dur, trans, start = random_problem(0)
    log_dur[:] = -np.inf
    with pytest.raises(ValueError, match="no segmentation"):
        sm.viterbi(sm.cumulative(emissions), log_dur, trans, start)


# -- the model ---------------------------------------------------------------------------------

from pompeu.models import hsmm  # noqa: E402


def test_durations_are_normalised_and_respect_minimums():
    p = hsmm._settings(None)
    log_dur = hsmm.log_durations(p, 25)
    np.testing.assert_allclose(np.logaddexp.reduce(log_dur, axis=1), 0.0, atol=1e-9)
    assert np.isinf(log_dur[hsmm.HD, :25]).all() and np.isfinite(log_dur[hsmm.HD, 25])
    assert np.isinf(log_dur[hsmm.HL, :5]).all() and np.isfinite(log_dur[hsmm.HL, 5])
    assert log_dur.shape[1] == 120 * 25 + 1


def test_transitions_are_stochastic_and_restarts_keep_the_team():
    T = np.exp(hsmm.transition_matrix(hsmm._settings(None)))
    np.testing.assert_allclose(T.sum(axis=1), 1.0)
    assert T[hsmm.HD, hsmm.HL] > 0.95 and T[hsmm.AD, hsmm.AL] > 0.95
    assert T[hsmm.HL, hsmm.HL] < 1e-5 and T[hsmm.HD, hsmm.AD] < 1e-5


def test_team_evidence_never_moves_dead_versus_live():
    """The factorisation: P(live) / P(dead) per frame depends on L_dead only."""
    rng = np.random.default_rng(0)
    n = 50
    dead = rng.normal(0, 2, n)
    team = {k: rng.normal(0, 3, n) for k in ("proximity", "events", "tagging", "restart")}
    e = hsmm.emissions(dead, team, hsmm._settings(None))
    live = np.logaddexp(e[:, hsmm.HL], e[:, hsmm.AL])
    stopped = np.logaddexp(e[:, hsmm.HD], e[:, hsmm.AD])
    np.testing.assert_allclose(live - stopped, -dead, atol=1e-9)


def test_unknown_params_rejected():
    with pytest.raises(KeyError, match="unknown hsmm params"):
        hsmm._settings({"w_tags": 1.0})


# -- end to end on synthetic scenes ------------------------------------------------------------

def test_throw_in_scene_labels_and_posteriors():
    from test_deadball import FIRST_HALF, throw_in_scene
    out = hsmm.decode(throw_in_scene(), {"posteriors": True})
    labels = out["labels"]
    assert set(labels[:55]) == {"H"}
    # The stoppage stays D although away player 4 has the ball placed next to him for 2 s
    # (a control event for away): in the four-state model that is the restarting team (AD).
    assert (labels[70:995] == "D").all()
    assert "D" not in set(labels[1005:])            # play resumes at the throw
    restarting = [s for a, b, s in out["segments"] if a <= FIRST_HALF + 500 < b]
    assert restarting == ["AD"]
    post = out["posterior"]
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)
    assert post[500, 2] > 0.99 and post[20, 0] > 0.9
    # The throw lands among home players (the scene's geometry): home has it -- surely while
    # they are next to the ball, less surely once it rests with nobody near (no new evidence).
    assert labels[-1] == "H" and post[1050, 0] > 0.95 and post[-1, 0] > 0.5


def test_outside_the_halves_is_dead():
    from test_deadball import throw_in_scene
    from synthetic import make_tracking
    tr = throw_in_scene()
    before_kickoff = make_tracking(tr.players_xy, tr.players_team[0], tr.ball_xy, frame0=10000)
    labels = hsmm.predict(before_kickoff)
    assert (labels[:1185] == "D").all()   # video frames 10000..11184 precede the 1st half


def test_real_match(tracking):
    import pandas as pd
    import possession_metrics as pm
    from pompeu.clean import clean_tracking
    clean, _ = clean_tracking(tracking)
    out = hsmm.decode(clean)
    labels = out["labels"]
    table = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    assert pm.validate_submission(table, 199319) is None
    graded = clean.half() > 0
    assert 0.35 <= np.mean(labels[graded] == "D") <= 0.50
    codes = pm.label_codes(labels)
    stats = pm.possession_stats(codes, pm.valid_runs(np.where(graded, codes, pm.X)))
    assert 150 <= stats["n_spells"] <= 500
