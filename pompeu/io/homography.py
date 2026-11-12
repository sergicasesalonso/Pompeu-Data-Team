"""Per-frame homography, metres in and pixels out.

starter/load_homography.py reads the file and maps NORMALISED pitch coordinates. Everything in
pompeu is in metres, so this divides by the pitch before projecting -- the one conversion that is
easy to forget and silently wrong when forgotten.
"""

import numpy as np

from pompeu import paths  # noqa: F401
import games
from starter import load_homography

load = load_homography.load


def project_m(matrix, x_m, y_m, game=games.GAME):
    """Pitch metres -> image pixels."""
    length, width = games.GAMES[game]["pitch"]
    return load_homography.project(matrix, np.asarray(x_m) / length, np.asarray(y_m) / width)


def unproject_m(matrix, pixel_x, pixel_y, game=games.GAME):
    """Image pixels -> pitch metres."""
    length, width = games.GAMES[game]["pitch"]
    x, y = load_homography.unproject(matrix, pixel_x, pixel_y)
    return x * length, y * width
