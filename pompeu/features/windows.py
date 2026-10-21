"""A2 multi-scale context: centred rolling mean, min and max at 0.5 s, 2 s and 5 s.

Centred because the whole match is available (offline): what happens just after a frame is as
informative as what happened just before. NaN-aware -- a window's statistic uses whatever frames in
it are defined, and is NaN only if none are -- so a ball feature keeps a value through short gaps.
"""

import pandas as pd

# Params
WINDOWS_S = (0.5, 2.0, 5.0)
STATS = ("mean", "min", "max")
# The features that get the treatment: the ones whose recent trend says more than their value.
WINDOWED = ("ball_speed", "ball_acc", "d1_H", "d1_A", "d1_diff", "n5_H", "n5_A", "pc_ball_H",
            "field_speed_mean", "stopped_frac", "cvx_H", "cvx_A")


def window_features(frame, fps, columns=WINDOWED, windows_s=WINDOWS_S, stats=STATS):
    """DataFrame of `<column>_w<seconds>_<stat>` columns, same index as `frame`."""
    out = {}
    for seconds in windows_s:
        width = max(1, int(round(seconds * fps)))
        rolling = frame[list(columns)].rolling(width, center=True, min_periods=1)
        for stat in stats:
            values = getattr(rolling, stat)()
            for column in columns:
                out["{}_w{:g}_{}".format(column, seconds, stat)] = values[column]
    return pd.DataFrame(out, index=frame.index)
