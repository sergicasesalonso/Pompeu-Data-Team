"""Synthetic Smart Tagging for external matches (pompeu.external.synthetic_tags): synthetic inputs."""

import numpy as np

from pompeu.external import synthetic_tags as st
from pompeu.features import context
from pompeu.io.contract import SRC_ATD, Tracking

FPS = 25
H, A, D, X = 0, 1, 2, 3


def _labels():
    """Two halves of alternating 20 s spells with 4 s dead runs between some, X around."""
    seq = [X] * 50
    for half in (1, 2):
        for k in range(8):
            seq += [H if k % 2 == 0 else A] * (20 * FPS)
            if k % 3 == 2:
                seq += [D] * (4 * FPS)
        if half == 1:
            seq += [X] * 100
    seq += [X] * 50
    labels = np.array(seq, np.int8)
    half = np.zeros(len(labels), np.int8)
    inside = labels != X
    first_x = 50 + np.flatnonzero(labels[50:] == X)[0]
    half[:first_x][inside[:first_x]] = 1
    half[first_x:][inside[first_x:]] = 2
    return labels, half


def _perfect():
    model = st.load_error_model(None)
    model.update(missing_rate=0.0, wrong_team_rate=0.0, false_rate=0.0, min_spell_s=0.0,
                 start_offset_s={"median": 0.0, "iqr": 0.0},
                 end_offset_s={"median": 0.0, "iqr": 0.0}, pairing_rate=1.0,
                 truncate_last=True, set_piece_rate=0.0)
    return model


def _tracking(n, halves):
    return Tracking(players_xy=np.zeros((n, 1, 2)), players_team=np.zeros((n, 1)),
                    players_conf=np.ones((n, 1)), ball_xy=np.zeros((n, 2)),
                    ball_z=np.full(n, np.nan), ball_conf=np.ones(n),
                    ball_source=np.full(n, SRC_ATD),
                    meta={"game": "SYNTH-TAGS", "fps": FPS, "pitch": (105.0, 68.0),
                          "home_attack_sign": 1, "team_names": ("home", "away"),
                          "halves": halves})


def _halves(half):
    return [(h, int(np.flatnonzero(half == h)[0]), int(np.flatnonzero(half == h)[-1]))
            for h in (1, 2)]


def test_schema_and_determinism():
    labels, half = _labels()
    model = st.load_error_model(None)
    a = st.generate(labels, half, model, seed=3, fps=FPS)
    b = st.generate(labels, half, model, seed=3, fps=FPS)
    c = st.generate(labels, half, model, seed=4, fps=FPS)
    assert list(a.columns) == ["code", "start_frame", "end_frame", "team", "type", "half", "tags"]
    assert a.drop(columns="tags").equals(b.drop(columns="tags"))
    assert not a.drop(columns="tags").equals(c.drop(columns="tags"))
    assert (a["end_frame"] > a["start_frame"]).all()
    assert set(a["team"]) <= {-1, 0, 1}
    # never outside a half, never across halves
    for s, e, h in zip(a["start_frame"], a["end_frame"], a["half"]):
        assert h in (1, 2) and half[s] == h and (half[s:e] == h).all()


def test_perfect_model_votes_with_the_labels():
    labels, half = _labels()
    tags = st.generate(labels, half, _perfect(), seed=0, fps=FPS)
    vote, count = st.vote_arrays(tags, len(labels))
    live = np.isin(labels, (H, A))
    sign = np.where(labels == H, 1.0, -1.0)
    assert (count[live] == 2).all()                      # attack + paired defend everywhere
    assert (np.sign(vote[live]) == sign[live]).all()
    assert (count[labels == D] == 0).all()
    assert set(tags["code"]) <= set(st.ATTACK_CODES + st.DEFEND_CODES)


def test_wrong_team_and_missing_rates():
    labels, half = _labels()
    live = np.isin(labels, (H, A))
    sign = np.where(labels == H, 1.0, -1.0)
    flipped = st.generate(labels, half, dict(_perfect(), wrong_team_rate=1.0), seed=0, fps=FPS)
    vote, _ = st.vote_arrays(flipped, len(labels))
    assert (np.sign(vote[live]) == -sign[live]).all()
    none = st.generate(labels, half, dict(_perfect(), missing_rate=1.0), seed=0, fps=FPS)
    assert len(none) == 0
    short = st.generate(labels, half, dict(_perfect(), min_spell_s=30.0), seed=0, fps=FPS)
    assert len(short) == 0                               # every spell is 20 s


def test_sequential_tagger_blocks_spells_inside_a_running_tag():
    labels, half = _labels()
    model = dict(_perfect(), truncate_last=False,
                 phase_duration_s={c: {"median": 50.0, "iqr": 0.0} for c in st.DEFAULTS["attack_mix"]})
    tags = st.generate(labels, half, model, seed=0, fps=FPS)
    attack = tags[tags["code"].isin(st.ATTACK_CODES)].sort_values("start_frame")
    # possession phases never overlap one another (within a half)
    for _, g in attack.groupby("half"):
        assert (g["start_frame"].to_numpy()[1:] >= g["end_frame"].to_numpy()[:-1]).all()
    spells = st.spells(labels, half)
    assert len(attack) < len(spells)                    # some spells were swallowed


def test_set_pieces_and_false_positives():
    labels, half = _labels()
    model = dict(_perfect(), set_piece_rate=1.0, set_piece_team_na=0.0,
                 set_piece_offset_s={"median": -1.0, "iqr": 0.0},
                 set_piece_duration_s={"median": 5.0, "iqr": 0.0})
    tags = st.generate(labels, half, model, seed=0, fps=FPS)
    sp = tags[tags["code"] == "SET PIECES"]
    dead = st.dead_runs(labels, half, 2.0, FPS)
    assert len(sp) == len(dead)
    for (a, b), (_, row) in zip(dead, sp.sort_values("start_frame").iterrows()):
        assert row["start_frame"] == b - FPS
        assert row["team"] == labels[b] and row["type"] in model["set_piece_types"]
    fp = st.generate(labels, half, dict(_perfect(), false_rate=1.0), seed=0, fps=FPS)
    starts = fp["start_frame"].to_numpy()
    assert np.isin(labels[starts], (D,)).sum() > 0      # false tags start inside dead runs


def test_context_features_on_an_external_tracking():
    labels, half = _labels()
    tracking = _tracking(len(labels), _halves(half))
    tags = st.generate(labels, half, _perfect(), seed=0, fps=FPS)
    F = st.context_features(tracking, tags)
    vote, count = st.vote_arrays(tags, len(labels))
    assert np.allclose(F["st_vote"], vote) and np.allclose(F["st_vote_n"], count)
    assert set(F) >= set(context.TAGGING_COLUMNS)
    import games
    assert "SYNTH-TAGS" not in games.GAMES                # registration undone

    from pompeu.features import build
    with st.injected({"SYNTH-TAGS": tags}):
        G = build.build_features(tracking, with_windows=False, groups=("context",))
    assert np.allclose(G["st_vote"].to_numpy(), vote)
    G0 = build.build_features(tracking, with_windows=False, groups=("context",))
    assert np.isnan(G0["st_vote"].to_numpy()).all()       # patch undone


def test_statistics_of_a_perfect_tagging():
    labels, half = _labels()
    tags = st.generate(labels, half, _perfect(), seed=0, fps=FPS)
    team = np.where(labels == H, 0, np.where(labels == A, 1, -1))
    live = team >= 0
    s = st.tag_statistics(tags, team, live, labels == D, half > 0, st.spells(labels, half), FPS)
    assert s["coverage_live"] == 1.0 and s["frame_agreement"] == 1.0
    assert s["missing_rate"] == 0.0 and s["wrong_team_rate"] == 0.0
    assert s["coverage_dead"] == 0.0
    assert s["start_offset_s"]["median"] == 0.0
