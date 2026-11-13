"""Synthetic checks of method hsmm_tcn (no match data): tag filter, wiring, determinism."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from pompeu import pipeline  # noqa: E402
from pompeu.experimental import hsmm_tcn, tcn  # noqa: E402
from pompeu.io.contract import Tracking  # noqa: E402


def _tags(rows):
    out = pd.DataFrame(rows, columns=["code", "type"])
    out["start_frame"], out["end_frame"] = 0, 10
    out["team"], out["half"], out["tags"] = 0, 1, [{}] * len(out)
    return out


def test_like_synthetic_keeps_possession_and_known_set_pieces_only():
    from pompeu.external.synthetic_tags import ATTACK_CODES, load_error_model
    known = load_error_model()["set_piece_types"][0]
    tags = _tags([(ATTACK_CODES[0], None), ("SET PIECES", known), ("SET PIECES", "no-such-type"),
                  ("GOAL", None), ("KICK OFF", None)])
    kept = tcn.like_synthetic(tags)
    assert list(kept["code"]) == [ATTACK_CODES[0], "SET PIECES"]
    assert kept.iloc[1]["type"] == known


def test_empty_tagging_has_the_columns():
    empty = tcn.empty_tagging()
    assert len(empty) == 0
    assert {"code", "start_frame", "end_frame", "team", "type", "half", "tags"} <= set(empty.columns)


def test_method_is_registered():
    assert pipeline.METHODS["hsmm_tcn"] is hsmm_tcn.predict


def test_wrong_weights_hash_is_refused(tmp_path):
    import torch  # noqa: F401
    model = tcn.build_model(4, dict(tcn.CONFIG, channels=8, dilations=(1, 2)))
    path = str(tmp_path / "m.pt")
    tcn.save(model, {"names": list("abcd"), "mean": [0] * 4, "std": [1] * 4, "nan_cols": []}, path)
    tr = Tracking(players_xy=np.zeros((10, 2, 2)), players_team=np.zeros((10, 2), int),
                  players_conf=np.ones((10, 2)), ball_xy=np.zeros((10, 2)), ball_z=np.full(10, np.nan),
                  ball_conf=np.ones(10), ball_source=np.ones(10),
                  meta={"pitch": (105.0, 68.0), "fps": 25})
    with pytest.raises(ValueError, match="hash"):
        hsmm_tcn.tcn_log_probs(tr, None, path, sha="0" * 16)
