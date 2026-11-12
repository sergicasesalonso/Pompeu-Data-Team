"""Line B guards, synthetic only: properties the B9 proposal relies on (equip/B9-PROPOSTA.md)."""

import glob
import os

import numpy as np
import pandas as pd

from pompeu import paths
from pompeu.vision import fuse
from test_fuse import FIRST_GRADED, GAP, N, _cands, _scene


def _same(a, b):
    for name in ("ball_xy", "ball_z", "ball_conf", "ball_source"):
        x, y = getattr(a, name), getattr(b, name)
        assert np.array_equal(x, y, equal_nan=True), name


def test_augment_ball_is_reproducible():
    tr = _scene()
    cands = [_cands(np.arange(*GAP), 0.8), _cands(np.arange(0, 200, 2), 0.6, source="broadcast")]
    a, ra = fuse.augment_ball(tr, cands)
    b, rb = fuse.augment_ball(tr, cands)
    _same(a, b)
    assert ra == rb


def test_candidate_order_does_not_change_the_result():
    tr = _scene()
    c1, c2 = _cands(np.arange(*GAP), 0.8), _cands(np.arange(*GAP), 0.5, offset=(0.3, 0.0))
    a, _ = fuse.augment_ball(tr, [c1, c2])
    b, _ = fuse.augment_ball(tr, [c1.sample(frac=1, random_state=0), c2.sample(frac=1, random_state=1)])
    _same(a, b)


def test_second_pass_adds_nothing():
    tr = _scene()
    once, _ = fuse.augment_ball(tr, [_cands(np.arange(*GAP))])
    twice, r = fuse.augment_ball(once, [])
    assert r["accepted"] == 0
    assert np.array_equal(once.ball_source, twice.ball_source)


def test_empty_candidates_leave_the_tracking_unchanged():
    tr = _scene()
    out, r = fuse.augment_ball(tr, [])
    _same(tr, out)
    assert r["accepted"] == 0


def test_all_candidates_outside_the_tracking_are_ignored():
    tr = _scene()
    far = pd.DataFrame({"frame": [FIRST_GRADED + N + 10, FIRST_GRADED - 5], "x_m": [1.0, 2.0],
                        "y_m": [1.0, 2.0], "conf": [0.9, 0.9], "source": "video"})
    out, r = fuse.augment_ball(tr, [far])
    _same(tr, out)


def test_atdplus_configs_do_not_clean_again():
    # build_atd_plus saves an ALREADY CLEANED Tracking; a `clean:` key would clean it twice.
    files = sorted(glob.glob(os.path.join(paths.ROOT, "configs", "*_atdplus.yaml")))
    assert files
    for f in files:
        with open(f) as fh:
            keys = [line.split(":")[0].strip() for line in fh if line.strip() and not line.startswith((" ", "#"))]
        assert "clean" not in keys, f


# -- tracking_hash (scripts/build_atd_plus.py) -----------------------------------------------------

import hashlib
import importlib.util

from pompeu.io.contract import ARRAY_FIELDS

_PATH = os.path.join(paths.ROOT, "scripts", "build_atd_plus.py")
_spec = importlib.util.spec_from_file_location("build_atd_plus", _PATH)
build_atd_plus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_atd_plus)


def _legacy_hash(tracking):
    """The hash as it was before NaN/zero canonicalisation."""
    h = hashlib.sha256()
    h.update(repr(int(tracking.frame0)).encode())
    h.update(repr(tracking.meta).encode())
    for name in ARRAY_FIELDS:
        a = np.ascontiguousarray(getattr(tracking, name))
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(repr(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _with_nan_bits(tr, bits):
    """Copy of tr whose ball_xy NaNs carry a different bit pattern."""
    xy = tr.ball_xy.copy()
    raw = xy.view(np.uint64)
    raw[np.isnan(xy)] = bits
    assert np.isnan(xy).any() and np.isnan(xy[np.isnan(xy)]).all()
    return type(tr)(**{**{f: getattr(tr, f) for f in ARRAY_FIELDS}, "ball_xy": xy},
                    frame0=tr.frame0, meta=tr.meta)


def test_hash_ignores_nan_payload_and_sign():
    tr = _scene()
    other = _with_nan_bits(tr, np.uint64(0xFFF8000000000001))     # negative NaN with a payload
    assert not np.array_equal(tr.ball_xy.view(np.uint64), other.ball_xy.view(np.uint64))
    assert build_atd_plus.tracking_hash(tr) == build_atd_plus.tracking_hash(other)


def test_hash_ignores_negative_zero():
    tr = _scene()
    other = type(tr)(**{**{f: getattr(tr, f) for f in ARRAY_FIELDS},
                        "ball_z": np.where(np.isnan(tr.ball_z), -0.0, tr.ball_z)},
                     frame0=tr.frame0, meta=tr.meta)
    same = type(tr)(**{**{f: getattr(tr, f) for f in ARRAY_FIELDS},
                       "ball_z": np.where(np.isnan(tr.ball_z), 0.0, tr.ball_z)},
                    frame0=tr.frame0, meta=tr.meta)
    assert build_atd_plus.tracking_hash(other) == build_atd_plus.tracking_hash(same)


def test_hash_still_detects_real_changes():
    tr = _scene()
    xy = tr.ball_xy.copy()
    xy[0, 0] += 1e-9
    other = type(tr)(**{**{f: getattr(tr, f) for f in ARRAY_FIELDS}, "ball_xy": xy},
                     frame0=tr.frame0, meta=tr.meta)
    assert build_atd_plus.tracking_hash(tr) != build_atd_plus.tracking_hash(other)


def test_hash_equals_legacy_when_arrays_are_canonical():
    # So the hashes already recorded (ESTAT.md) stay valid whenever the arrays had canonical NaN.
    tr = _scene()
    assert build_atd_plus.tracking_hash(tr) == _legacy_hash(tr)
