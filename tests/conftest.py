import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pompeu import paths  # noqa: E402,F401  (games and possession_metrics importable from here on)
import games  # noqa: E402


def _have_data():
    try:
        games.require(["atd_meta", "atd_raw"])
        return True
    except IOError:
        return False


needs_data = pytest.mark.skipif(not _have_data(), reason="match data not in data/ (gated)")


@pytest.fixture(scope="session")
def atd():
    if not _have_data():
        pytest.skip("match data not in data/ (gated)")
    from pompeu.io.atd import load_atd
    return load_atd()


@pytest.fixture(scope="session")
def tracking(atd):
    from pompeu.io.contract import Tracking
    return Tracking.from_atd(atd)
