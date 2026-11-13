"""A3 step 3: the live/dead classifier -- gradient boosting on A2 features, trained on external matches.

    from pompeu.models import deadball_gbm
    model = deadball_gbm.load()                   # artifacts/deadball_gbm_v<version>.txt
    p_dead = deadball_gbm.predict(model, F)       # (N,) P(ball dead) per frame

Trained by scripts/train_deadball_gbm.py on the `train` external matches (degraded like the ATD,
several seeds each), early-stopped on `val`, measured on `test`; never on BAR-ATH. Inputs are
examples.model_columns: every A2 feature except Smart Tagging, whose columns are always missing
on the external matches. LightGBM handles the missing ball natively (NaN), which is the point:
the classifier learns what a stoppage looks like with and without a ball.

In A3 it is one more expert (deadball PARAMS gbm_weight): its per-frame log-odds, scaled for
autocorrelation like every other cue, are added to the rule cues before the two-state HMM.

The model file is small and trained on open data only (IDSSE CC-BY 4.0, Metrica sample data), so
it is committed: the repository regenerates the submission without retraining. Training is
deterministic (fixed seed, deterministic=True, a fixed thread count).
"""

import os

import numpy as np

from pompeu import paths

# Params
# 1: the committed artifact (lightgbm 4.6.0), used by every measured config. A v2 retrained on
# examples v2 (ATD-like ball blur) was planned but never trained: bump only with its artifact.
VERSION = 1
ARTIFACT = os.path.join(paths.ROOT, "artifacts", "deadball_gbm_v{}.txt".format(VERSION))
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 8,
    "verbose": -1,
}
MAX_ROUNDS = 3000
EARLY_STOP = 100

_CACHE = {}


def load(path=ARTIFACT):
    """The trained booster (cached per path)."""
    import lightgbm as lgb
    if path not in _CACHE:
        if not os.path.isfile(path):
            raise IOError("{} not found -- run scripts/train_deadball_gbm.py".format(path))
        _CACHE[path] = lgb.Booster(model_file=path)
    return _CACHE[path]


def predict(model, F):
    """(N,) P(dead) for the frames of a feature table (any extra columns are ignored)."""
    columns = model.feature_name()
    missing = [c for c in columns if c not in F.columns]
    if missing:
        raise ValueError("features missing for the dead-ball model: {}".format(missing[:5]))
    return model.predict(F[columns].to_numpy(dtype=np.float32))


def train(X, y_dead, X_val, y_val, params=None, log=print):
    """A booster trained on (X, y_dead), early-stopped on validation log-loss."""
    import lightgbm as lgb
    p = dict(PARAMS, **(params or {}))
    train_set = lgb.Dataset(X, label=y_dead.astype(np.float32), free_raw_data=True)
    val_set = lgb.Dataset(X_val, label=y_val.astype(np.float32), reference=train_set)
    booster = lgb.train(p, train_set, num_boost_round=MAX_ROUNDS, valid_sets=[val_set],
                        valid_names=["val"],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                   lgb.log_evaluation(200)])
    log("best iteration {}  val logloss {:.4f}".format(
        booster.best_iteration, booster.best_score["val"]["binary_logloss"]))
    return booster
