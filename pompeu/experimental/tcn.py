"""Experimental: a dilated temporal CNN frame classifier (H / A / D) over A2 features plus
synthetic Smart Tagging channels. Not part of the submission pipeline.

    from pompeu.experimental import tcn
    item = tcn.build_item("dfl:J03WOH", seed=0, swap=True)      # 5 Hz inputs + labels at 2/5/8 s
    model, norm = tcn.load(path)
    log_p = tcn.predict_frames(model, norm, item["X"], item["names"], n_frames)  # (N, 3)

INPUTS, per 5 Hz step (every 5th frame of the cleaned, degraded match):
  * A2's features (build_features(tagging=None), examples.model_columns: no tagging columns);
  * the Smart Tagging context columns (pompeu.features.context.TAGGING_COLUMNS) computed from
    SYNTHETIC tags (pompeu.external.synthetic_tags, error model measured on BAR-ATH's real tags,
    generated from the 2 s labels). On BAR-ATH the same columns come from the real tags.
  Standardised with train statistics, NaN -> 0 plus a missing flag per column that can be NaN.

AUGMENTATION (each recomputed from the CLEAN match, never translated column by column):
  team swap (examples.swap_teams: H <-> A, attack direction reversed) and y-mirror (the pitch
  flipped across its long axis). Degradation (pompeu.external.degrade defaults: ball_missing 0.41,
  dead_weight 5) and tags are drawn per (match, seed).

NETWORK: 1x1 input projection, residual blocks of non-causal dilated convolutions (kernel 3,
dilations 1..128 at 5 Hz: about +-51 s of context), channel LayerNorm, GELU, dropout; three heads,
one per labelling convention (min_possession_s 2, 5, 8), cross-entropy ignoring X. Inference
averages the heads' log-probabilities, then interpolates 5 Hz -> 25 Hz.

Deterministic: seeded numpy/torch, torch.use_deterministic_algorithms(True),
CUBLAS_WORKSPACE_CONFIG=:4096:8 (set by set_determinism()).
"""

import json
import os
from dataclasses import replace

import numpy as np

from pompeu import paths

# Params
STEP = 5                    # frames per step: 25 fps -> 5 Hz
VALUES = (2.0, 5.0, 8.0)    # min_possession_s of each head
X_CODE = 3
TAG_SEED = 7919
CACHE = os.path.join(paths.CACHE_DIR, "tcn")
CONFIG = {
    "channels": 96,
    "dilations": (1, 2, 4, 8, 16, 32, 64, 128),
    "kernel": 3,
    "dropout": 0.15,
    "input_dropout": 0.1,
}


# -- inputs --------------------------------------------------------------------------------------

def mirror_y(tracking):
    """The match flipped across the pitch's long axis (y -> width - y); labels unchanged."""
    width = float(tracking.meta["pitch"][1])
    players = tracking.players_xy.copy()
    players[..., 1] = width - players[..., 1]
    ball = tracking.ball_xy.copy()
    ball[:, 1] = width - ball[:, 1]
    return replace(tracking, players_xy=players, ball_xy=ball)


def item_path(name, seed, swap=False, mirror=False, tag="v1"):
    safe = name.replace(":", "_") + ("_swap" if swap else "") + ("_mir" if mirror else "")
    return os.path.join(CACHE, "{}_s{}_{}.npz".format(safe, seed, tag))


def prepared(name, swap=False, mirror=False):
    """(clean external Tracking, labels, (3, N) aligned labels at VALUES) after augmentation."""
    from pompeu import external
    from pompeu.external import examples
    tracking, labels = external.load(name)
    if swap:
        tracking, labels = examples.swap_teams(tracking, labels)
    if mirror:
        tracking = mirror_y(tracking)
    fps = tracking.meta.get("fps", 25)
    graded = np.where(tracking.half() > 0, labels, X_CODE).astype(np.int8)
    ys = np.stack([external.align_labels(graded, v, fps) for v in VALUES]).astype(np.int8)
    return tracking, labels, ys


def inputs_from(clean, tags, F=None):
    """(X (T, C) float32 at 5 Hz, column names) from a cleaned Tracking and its tags."""
    from pompeu.external import examples, synthetic_tags as st
    from pompeu.features import context
    from pompeu.features.build import build_features
    if F is None:
        F = build_features(clean, tagging=None)
    rows = np.arange(0, clean.n_frames, STEP)
    columns = examples.model_columns(F.columns)
    feats = F[columns].to_numpy(dtype=np.float32)[rows]
    ctx = st.context_features(clean, tags)
    tag_cols = [c for c in context.TAGGING_COLUMNS if c in ctx]
    tagx = np.column_stack([np.asarray(ctx[c], dtype=np.float32)[rows] for c in tag_cols])
    names = list(columns) + ["tag:" + c for c in tag_cols]
    return np.concatenate([feats, tagx], axis=1), names


def build_item(name, seed, swap=False, mirror=False, run_lengths=None, degrade_seed=None,
               tag_seed=None, return_clean=False):
    """{'X', 'names', 'y' (3, T), 'n_frames'} for one (match, degradation, augmentation)."""
    from pompeu.clean import clean_tracking
    from pompeu.external import degrade, synthetic_tags as st
    if run_lengths is None:
        from pompeu.io.contract import load_tracking
        from pompeu.validate.stress import missing_run_lengths
        run_lengths = missing_run_lengths(load_tracking("atd"))
    tracking, labels, ys = prepared(name, swap, mirror)
    fps = tracking.meta.get("fps", 25)
    dseed = seed if degrade_seed is None else degrade_seed
    degraded = degrade.degrade(tracking, labels, dseed, run_lengths, None)
    clean, _ = clean_tracking(degraded, {})
    tags = st.generate(ys[0], tracking.half(), st.load_error_model(),
                       seed=TAG_SEED + dseed if tag_seed is None else tag_seed, fps=fps)
    X, names = inputs_from(clean, tags)
    out = {"X": X, "names": names, "y": ys[:, ::STEP], "n_frames": clean.n_frames}
    if return_clean:
        out.update(clean=clean, degraded=degraded, tags=tags, ys=ys)
    return out


def save_item(item, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as handle:
        np.savez_compressed(handle, X=item["X"], y=item["y"], names=np.array(item["names"]),
                 n_frames=item["n_frames"])
    os.replace(path + ".part", path)


def load_item(path):
    with np.load(path) as z:
        return {"X": z["X"], "y": z["y"], "names": list(z["names"]), "n_frames": int(z["n_frames"])}


def fit_normalisation(items):
    names = items[0]["names"]
    stacked = np.concatenate([it["X"][::5] for it in items])
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    nan_cols = [i for i, any_nan in enumerate(np.isnan(stacked).any(axis=0)) if any_nan]
    return {"names": names, "mean": mean.tolist(), "std": std.tolist(), "nan_cols": nan_cols}


def normalise(X, names, norm):
    if list(names) != list(norm["names"]):
        index = {n: i for i, n in enumerate(names)}
        X = np.column_stack([X[:, index[n]] if n in index else np.full(len(X), np.nan, np.float32)
                             for n in norm["names"]])
    z = (X - np.asarray(norm["mean"], np.float32)) / np.asarray(norm["std"], np.float32)
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)
    flags = np.isnan(X[:, norm["nan_cols"]]).astype(np.float32)
    return np.concatenate([z, flags], axis=1).astype(np.float32)


# -- network -------------------------------------------------------------------------------------

def set_determinism(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(n_in, config=CONFIG):
    import torch
    nn = torch.nn

    class ChannelNorm(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.norm = nn.LayerNorm(c)

        def forward(self, x):                       # (B, C, T)
            return self.norm(x.transpose(1, 2)).transpose(1, 2)

    class Block(nn.Module):
        def __init__(self, c, d):
            super().__init__()
            k = config["kernel"]
            self.conv = nn.Conv1d(c, c, k, dilation=d, padding=d * (k - 1) // 2)
            self.norm = ChannelNorm(c)
            self.act = nn.GELU()
            self.drop = nn.Dropout(config["dropout"])
            self.mix = nn.Conv1d(c, c, 1)

        def forward(self, x):
            return x + self.mix(self.drop(self.act(self.norm(self.conv(x)))))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            c = config["channels"]
            self.config = dict(config, n_in=n_in)
            self.in_drop = nn.Dropout(config["input_dropout"])
            self.inp = nn.Conv1d(n_in, c, 1)
            self.blocks = nn.Sequential(*[Block(c, d) for d in config["dilations"]])
            self.norm = ChannelNorm(c)
            self.out = nn.Conv1d(c, 3 * len(VALUES), 1)

        def forward(self, x):                       # (B, T, n_in) -> (B, heads, T, 3)
            h = self.inp(self.in_drop(x).transpose(1, 2))
            h = self.norm(self.blocks(h))
            o = self.out(h)                         # (B, 3 * heads, T)
            b, _, t = o.shape
            return o.view(b, len(VALUES), 3, t).transpose(2, 3)

    return TCN()


def save(model, norm, path, extra=None):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": model.state_dict(), "config": model.config}, path)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(dict(norm, **(extra or {})), handle)


def load(path, device="cpu"):
    import torch
    blob = torch.load(path, map_location=device)
    config = dict(blob["config"])
    n_in = config.pop("n_in")
    config["dilations"] = tuple(config["dilations"])
    model = build_model(n_in, config)
    model.load_state_dict(blob["state"])
    model.to(device).eval()
    with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as handle:
        norm = json.load(handle)
    return model, norm


def predict_steps(model, Z):
    """(T, 3) log-probabilities at 5 Hz from normalised inputs Z (T, C): heads averaged."""
    import torch
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.as_tensor(Z[None], device=device)).float()
        log_p = torch.log_softmax(out, dim=-1)[0].mean(dim=0)        # mean over heads
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
    return log_p.cpu().numpy().astype(np.float64)


def upsample(step_log_probs, n_frames):
    steps = np.arange(len(step_log_probs)) * STEP
    frames = np.arange(n_frames)
    out = np.column_stack([np.interp(frames, steps, step_log_probs[:, c])
                           for c in range(step_log_probs.shape[1])])
    return out - np.logaddexp.reduce(out, axis=1, keepdims=True)


def predict_frames(model, norm, X, names, n_frames):
    """(n_frames, 3) per-frame log P(H), log P(A), log P(D)."""
    return upsample(predict_steps(model, normalise(X, names, norm)), n_frames)


def smooth_labels(log_p, window):
    """Argmax after a centred moving average of the probabilities over `window` frames."""
    p = np.exp(log_p)
    if window > 1:
        kernel = np.ones(window) / window
        p = np.column_stack([np.convolve(p[:, c], kernel, mode="same") for c in range(3)])
    return np.argmax(p, axis=1).astype(np.int8)
