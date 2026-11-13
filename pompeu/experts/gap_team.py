"""Team evidence inside ball gaps from team shape and motion (hsmm_v2 proposal).

When the ball is missing (41% of graded frames on BAR-ATH) the HSMM's team experts -- proximity,
touches, controls -- go silent and the duration prior just carries the last team, so a change of
possession inside a gap is never found. scripts/experiments/gap_team.py measured, on external
val (DFL J03WN1), that a logistic regression on A2's collective group (no ball needed: centroids,
centroid velocity, lines, width/depth, pressing/recovering counts, speeds) plus their 3 s centred
means names the post-change team in 75% of the gaps where possession changes; carrying the last
team names it in 0%.

This module turns that model into an HSMM emission expert:

    model = load_model("configs/gap_team_v1.json")     # scripts/train_gap_team.py
    llr = team_evidence(tracking, F, model)            # (N,) log-odds HOME vs away

SIGN. Positive = home has the ball, as hsmm's L_team. The logistic model predicts P(away), so the
evidence is minus its logit.

WHERE. Only on frames inside a half without a (cleaned) ball: elsewhere the ball-based experts
speak and this one is 0. Frames whose collective features are all missing give 0 too; a single
missing feature contributes its training mean (0 after standardising).

NO INTERCEPT. The intercept only encodes the home/away base rate of the training matches, which
says nothing about another match; the evidence is the feature part of the logit only.

The model is fitted on public external matches (DFL), never on BAR-ATH, and ships as a small
JSON (coefficients, feature list, standardisation): see scripts/train_gap_team.py.
"""

import json
import os

import numpy as np
import pandas as pd

from pompeu import paths

# Params
WINDOW_S = 3.0      # centred mean added to every collective column
RIDGE = 1.0         # L2 on standardised coefficients (not on the intercept)
# A2's collective group (pompeu.features.collective), in a fixed order.
BASE_COLUMNS = tuple(
    "{}_{}".format(name, team)
    for team in ("H", "A")
    for name in ("n_vis", "cx", "cy", "cvx", "cvy", "width", "depth", "def_line", "front",
                 "compact", "n_recover", "n_press", "speed_mean")
) + ("field_speed_mean", "field_speed_median", "stopped_frac", "centroid_gap", "line_gap")

_MODELS = {}


# -- features ----------------------------------------------------------------------------------

def feature_names(base_columns=BASE_COLUMNS, window_s=WINDOW_S):
    return list(base_columns) + ["{}_m{:g}s".format(c, window_s) for c in base_columns]


def design_matrix(F, fps, base_columns=BASE_COLUMNS, window_s=WINDOW_S):
    """(N, 2 * len(base_columns)) float64: the columns and their centred `window_s` means.

    A column absent from F is all NaN (it then contributes nothing). Training and inference both
    build their matrix here, from the same F that hsmm.prepare computes.
    """
    n = len(F)
    raw = pd.DataFrame({c: (F[c].to_numpy(np.float64) if c in F.columns else np.full(n, np.nan))
                        for c in base_columns})
    rolled = raw.rolling(max(1, int(round(window_s * fps))), center=True, min_periods=1).mean()
    return np.column_stack([raw.to_numpy(np.float64), rolled.to_numpy(np.float64)])


def standardise(X, mean=None, std=None):
    """(Z, mean, std); NaN -> 0 after standardising, constant columns -> std 1."""
    if mean is None:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        std = np.where(~(std > 1e-9), 1.0, std)
    Z = (X - np.asarray(mean)) / np.asarray(std)
    return np.where(np.isnan(Z), 0.0, Z), mean, std


def fit_logistic(Z, y, ridge=RIDGE, iters=50, tol=1e-8):
    """(w, b) of a ridge-penalised logistic regression by Newton's method. y in {0, 1}."""
    n, d = Z.shape
    Xb = np.column_stack([Z, np.ones(n)])
    beta = np.zeros(d + 1)
    penalty = np.full(d + 1, ridge)
    penalty[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        grad = Xb.T @ (p - y) + penalty * beta
        hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(penalty) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < tol:
            break
    return beta[:-1], beta[-1]


# -- model -------------------------------------------------------------------------------------

def load_model(path):
    """The JSON written by scripts/train_gap_team.py (relative paths: from the repo root)."""
    if path is None:
        raise ValueError("gap_team_weight > 0 needs gap_team_model (path to the JSON)")
    if isinstance(path, dict):
        return path
    full = path if os.path.isabs(path) else os.path.join(paths.ROOT, path)
    if full not in _MODELS:
        with open(full, encoding="utf-8") as handle:
            _MODELS[full] = json.load(handle)
    return _MODELS[full]


def away_logit(F, fps, model):
    """(N,) feature part of the logit of P(away has the ball); NaN where every feature is NaN."""
    X = design_matrix(F, fps, model["base_columns"], model["window_s"])
    if list(model["features"]) != feature_names(model["base_columns"], model["window_s"]):
        raise ValueError("gap_team model features do not match its base columns")
    Z, _, _ = standardise(X, np.asarray(model["mean"], float), np.asarray(model["std"], float))
    logit = Z @ np.asarray(model["coef"], float)
    return np.where(np.all(np.isnan(X), axis=1), np.nan, logit)


def team_evidence(tracking, F, model):
    """(N,) log-odds home vs away (+ = home, as hsmm's L_team) on in-half frames without ball,
    0 elsewhere and wherever the features are missing."""
    fps = tracking.meta.get("fps", 25)
    model = load_model(model)
    evidence = -away_logit(F, fps, model)
    gap = (tracking.half() > 0) & ~np.asarray(tracking.ball_present, bool)
    evidence = np.where(gap & np.isfinite(evidence), evidence, 0.0)
    return evidence.astype(np.float64)
