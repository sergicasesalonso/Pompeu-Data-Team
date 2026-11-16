"""A7: the sequence model's inputs and network, and A3's learned classifier hook."""

import numpy as np
import pandas as pd
import pytest

from pompeu.external import examples
from pompeu.models import seqmodel
from synthetic import make_tracking

torch = pytest.importorskip("torch")


def test_band_mask_limits_attention():
    mask = seqmodel.band_mask(10, 2, "cpu")
    assert mask[5, 3] == 0 and mask[5, 7] == 0
    assert mask[5, 2] == float("-inf") and mask[5, 8] == float("-inf")


def test_upsample_keeps_probabilities():
    steps = np.log(np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]))
    out = seqmodel.upsample(steps, 10)
    np.testing.assert_allclose(np.exp(out).sum(axis=1), 1.0, atol=1e-9)
    assert np.exp(out[0, 0]) > 0.7 and np.exp(out[-1, 1]) > 0.7


def test_pad_players_keeps_the_busiest_slots():
    tokens = {"players": np.ones((4, 3, 8), np.float32),
              "mask": np.array([[1, 0, 1]] * 4, dtype=bool)}
    players, mask = seqmodel.pad_players(tokens, 5)
    assert players.shape == (4, 5, 8) and mask[:, :3].sum() == 8 and not mask[:, 3:].any()
    players, mask = seqmodel.pad_players(tokens, 2)
    assert mask.all()                                  # the empty slot is the one dropped


def test_network_shapes_and_player_order_does_not_matter():
    torch.manual_seed(0)
    model = seqmodel.build_model(12, dict(seqmodel.CONFIG, window=20, band=5)).eval()
    rng = np.random.default_rng(0)
    players = rng.normal(size=(1, 20, 6, 8)).astype(np.float32)
    players[..., 0] = np.sign(players[..., 0])
    mask = np.ones((1, 20, 6), dtype=bool)
    ball = rng.normal(size=(1, 20, 6)).astype(np.float32)
    features = rng.normal(size=(1, 20, 12)).astype(np.float32)
    with torch.no_grad():
        logits, boundary = model(*[torch.as_tensor(a) for a in (players, mask, ball, features)])
        order = rng.permutation(6)
        shuffled, _ = model(*[torch.as_tensor(a) for a in (players[:, :, order], mask, ball,
                                                             features)])
    assert logits.shape == (1, 20, 3) and boundary.shape == (1, 20)
    np.testing.assert_allclose(logits.numpy(), shuffled.numpy(), atol=1e-4)


def test_feature_inputs_standardise_and_flag_missing():
    F = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0] * 2, "b": np.arange(10.0)})
    norm = {"columns": ["a", "b"], "mean": [3.0, 4.5], "std": [1.0, 2.0], "nan_columns": ["a"]}
    x = seqmodel.feature_inputs(F, norm)
    assert x.shape == (2, 3)                           # every 5th frame; 2 values + 1 flag
    np.testing.assert_allclose(x[:, 0], [-2.0, -2.0])
    assert (x[:, 2] == 0).all()


def test_player_tokens_and_team_swap():
    n = 50
    xy = np.tile(np.array([[30.0, 30.0], [60.0, 40.0]]), (n, 1, 1))
    tr = make_tracking(xy, np.array([0, 1]), np.tile([31.0, 30.0], (n, 1)), frame0=20000)
    tokens = examples.player_tokens(tr)
    assert tokens["players"].shape == (10, 2, 8) and tokens["ball"].shape == (10, 6)
    assert tokens["players"][0, 0, 0] == 1.0 and tokens["players"][0, 1, 0] == -1.0
    labels = np.array([0] * 20 + [1] * 20 + [2] * 10, dtype=np.int8)
    swapped, swapped_labels = examples.swap_teams(tr, labels)
    assert (swapped.players_team[0] == [1, 0]).all()
    assert swapped.meta["home_attack_sign"] == -tr.meta["home_attack_sign"]
    np.testing.assert_array_equal(swapped_labels, [1] * 20 + [0] * 20 + [2] * 10)


def test_gbm_cue_off_by_default():
    from pompeu.events import deadball
    p = deadball._settings(None)
    assert p["gbm_weight"] == 0.0 and deadball.gbm_cue(None, None, p) is None
