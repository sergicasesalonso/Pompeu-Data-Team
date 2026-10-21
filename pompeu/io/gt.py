"""Ground truth and label arrays, through the organisers' possession_metrics -- never re-parsed."""

import numpy as np

from pompeu import paths
import possession_metrics as pm


def load_gt_codes(path=paths.PUBLIC_GT):
    """(N,) int8 codes H=0, A=1, D=2, X=3 of a frame,label CSV, validated as a submission."""
    frame = pm.read_label_csv(path)
    reason = pm.validate_submission(frame, len(frame))
    if reason:
        raise ValueError("{}: {}".format(path, reason))
    return pm.label_codes(frame)


def graded_window(gt_codes):
    """[(start, stop)] half-open spans where the ground truth is not X (pm.valid_runs)."""
    return pm.valid_runs(gt_codes)


def labels_to_codes(labels):
    return pm.label_codes(np.asarray(labels))


def codes_to_labels(codes):
    return np.array(pm.LABELS, dtype="<U1")[np.asarray(codes)]
