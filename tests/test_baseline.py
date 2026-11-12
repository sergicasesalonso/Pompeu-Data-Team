"""The baseline on the Tracking contract reproduces the organisers' committed files exactly.

This is the end-to-end check of stage 0: loader, contract and pipeline together reading the same
numbers starter/baseline.py reads from the raw file.
"""

import os

import numpy as np
import pandas as pd
import pytest

from pompeu import paths
from pompeu.models import baseline


@pytest.mark.parametrize("name, window", [("baseline_raw", 0), ("baseline_smooth25", 25),
                                          ("baseline_smooth50", 50), ("baseline_smooth125", 125)])
def test_reproduces_published_baseline(tracking, name, window):
    published = pd.read_csv(os.path.join(paths.EVALUATION_DIR, "results", name + ".csv"))
    ours = baseline.predict(tracking, window)
    np.testing.assert_array_equal(ours, published["label"].values.astype("<U1"))


def test_crop_with_context_matches_full_run(tracking):
    """What the stress bench relies on: a window plus context gives the full-match answer."""
    full = baseline.predict(tracking, 125)
    crop = tracking.crop(11185 - 1500, 18685 + 1500)
    part = baseline.predict(crop, 125)
    np.testing.assert_array_equal(part[1500:-1500], full[11185:18685])
