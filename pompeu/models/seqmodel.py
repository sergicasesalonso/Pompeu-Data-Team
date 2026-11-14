"""A7: the sequence model -- who has the ball, especially when the ball is not there.

    from pompeu.models import seqmodel
    model, norm = seqmodel.load()                         # artifacts/seqmodel_v<version>.pt
    log_probs = seqmodel.predict_tracking(model, norm, clean_tracking)   # (N, 3) per frame

ARCHITECTURE (plan, stage A7; sized for a 4 GB GPU):

  1. Set encoder, permutation-invariant over players: every player is a token (team, position and
     velocity in the home-attacks-+x frame, offset to the ball, confidence -- examples.player_tokens),
     the ball a token of its own, masked when missing. Two Transformer layers (4 heads, d 128)
     over [ball, players]; outputs pooled per team (masked mean) plus the ball token. Slot order
     never matters, so tracklets that change slot are no problem.
  2. A2's hand-made features of the frame (examples.model_columns, standardised, NaN -> 0 with a
     missing flag per column that can be missing) projected to d and added: hand-made features
     plus learning, steadier with little data.
  3. Temporal encoder, BIDIRECTIONAL: 4 Transformer layers over 5 Hz steps with attention limited
     to +/- 30 s (a band mask). Bidirectional because who attacks five seconds after a gap tells
     who won the ball inside it.
  4. Heads: P(H / A / D) per step (focal loss, class weights) and P(boundary) per step (BCE on a
     Gaussian of +/- 1 step around every label change -- the head that matters for spellF1).

TRAINING (scripts/train_seqmodel.py): the external `train` matches, each in 8 degradations and
its team-swapped twin (examples: features and tokens recomputed, never translated), 120-s windows
sampled at random, AdamW + cosine schedule, mixed precision, early stopping on `val`.

INFERENCE: overlapping windows (stride half a window), logits averaged where they overlap, then
linearly interpolated from 5 Hz back to every frame. Output: per-frame log P(H), log P(A),
log P(D), which hsmm_v2 uses as one more expert.

Not built (documented in ARQUITECTURA.md): the 25 Hz dilated TCN branch (the 5 Hz output is
interpolated instead) and the restart-type auxiliary head (the external labels have no restart
types).
"""

import json
import os

import numpy as np

from pompeu import paths
from pompeu.external import examples

# Params
# 1: the committed artifact (artifacts/seqmodel_v1.pt), trained on external train/val. A v2 on
# examples v2 (ATD-like ball blur) was planned but never trained: bump only with its artifact.
VERSION = 1
ARTIFACT = os.path.join(paths.ROOT, "artifacts", "seqmodel_v{}.pt".format(VERSION))
CONFIG = {
    "d_model": 128,
    "set_layers": 2,
    "time_layers": 4,
    "heads": 4,
    "dropout": 0.1,
    "window": 600,          # steps at 5 Hz = 120 s
    "band": 150,            # attention reach in steps = +/- 30 s
    "max_players": 40,
}
N_CLASSES = 3               # H, A, D


def _torch():
    import torch
    return torch


def build_model(n_features, config=CONFIG):
    """The network (torch.nn.Module) for n_features hand-made inputs (values + missing flags)."""
    torch = _torch()
    nn = torch.nn
    d = config["d_model"]

    class SeqModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.player_in = nn.Sequential(nn.Linear(len(examples.TOKEN_FEATURES), d), nn.GELU(),
                                           nn.Linear(d, d))
            self.ball_in = nn.Sequential(nn.Linear(len(examples.BALL_FEATURES), d), nn.GELU(),
                                         nn.Linear(d, d))
            layer = nn.TransformerEncoderLayer(d, config["heads"], 2 * d, config["dropout"],
                                               batch_first=True, norm_first=True)
            self.set_encoder = nn.TransformerEncoder(layer, config["set_layers"])
            self.features_in = nn.Sequential(nn.Linear(n_features, d), nn.GELU(), nn.Linear(d, d))
            self.merge = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.LayerNorm(d))
            time_layer = nn.TransformerEncoderLayer(d, config["heads"], 4 * d, config["dropout"],
                                                    batch_first=True, norm_first=True)
            self.time_encoder = nn.TransformerEncoder(time_layer, config["time_layers"])
            self.classes = nn.Linear(d, N_CLASSES)
            self.boundary = nn.Linear(d, 1)
            self.config = config

        def encode_sets(self, players, mask, ball):
            """(B, T, K, F), (B, T, K) bool, (B, T, Fb) -> (B, T, 3 d)."""
            b, t, k, _ = players.shape
            tokens = torch.cat([self.ball_in(ball).unsqueeze(2), self.player_in(players)], dim=2)
            # The ball token is always attended to (a missing ball is all zeros + its flag).
            pad = torch.cat([torch.zeros(b, t, 1, dtype=torch.bool, device=mask.device), ~mask],
                            dim=2)
            out = self.set_encoder(tokens.reshape(b * t, k + 1, -1),
                                   src_key_padding_mask=pad.reshape(b * t, k + 1))
            out = out.reshape(b, t, k + 1, -1)
            ball_out, player_out = out[:, :, 0], out[:, :, 1:]
            home = (players[..., 0] > 0) & mask
            away = (players[..., 0] < 0) & mask

            def pooled(select):
                weight = select.unsqueeze(-1).to(player_out.dtype)
                return (player_out * weight).sum(2) / weight.sum(2).clamp(min=1.0)

            return torch.cat([pooled(home), pooled(away), ball_out], dim=-1)

        def forward(self, players, mask, ball, features):
            """-> (class logits (B, T, 3), boundary logits (B, T))."""
            t = players.shape[1]
            x = torch.cat([self.encode_sets(players, mask, ball), self.features_in(features)],
                          dim=-1)
            x = self.merge(x)
            x = x + positional(t, x.shape[-1], x.device, x.dtype)
            band = band_mask(t, self.config["band"], x.device)
            x = self.time_encoder(x, mask=band)
            return self.classes(x), self.boundary(x).squeeze(-1)

    return SeqModel()


def positional(t, d, device, dtype):
    torch = _torch()
    position = torch.arange(t, device=device, dtype=torch.float32).unsqueeze(1)
    rate = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32)
                     * (-np.log(10000.0) / d))
    out = torch.zeros(t, d, device=device)
    out[:, 0::2] = torch.sin(position * rate)
    out[:, 1::2] = torch.cos(position * rate)
    return out.to(dtype)


def band_mask(t, band, device):
    """(t, t) float mask: 0 within +/- band steps, -inf beyond (local attention)."""
    torch = _torch()
    index = torch.arange(t, device=device)
    far = (index[None, :] - index[:, None]).abs() > band
    return torch.zeros(t, t, device=device).masked_fill(far, float("-inf"))


# -- inputs ------------------------------------------------------------------------------------

def feature_inputs(F, norm):
    """(T, n) float32 model inputs at 5 Hz from a 25 Hz feature table: standardised values with
    NaN -> 0, then one missing flag per column in norm['nan_columns']."""
    rows = np.arange(0, len(F), examples.TOKEN_STEP)
    values = F[norm["columns"]].to_numpy(dtype=np.float32)[rows]
    flags = np.isnan(F[norm["nan_columns"]].to_numpy(dtype=np.float32)[rows]).astype(np.float32)
    z = (values - np.asarray(norm["mean"], np.float32)) / np.asarray(norm["std"], np.float32)
    z = np.clip(np.nan_to_num(z), -5.0, 5.0)
    return np.concatenate([z, flags], axis=1)


def fit_normalisation(frames):
    """{'columns', 'mean', 'std', 'nan_columns'} from a list of training feature tables."""
    columns = examples.model_columns(frames[0].columns)
    stacked = np.concatenate([f[columns].to_numpy(dtype=np.float32)[::25] for f in frames])
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    nan_columns = [c for c, any_nan in zip(columns, np.isnan(stacked).any(axis=0)) if any_nan]
    return {"columns": columns, "mean": mean.tolist(), "std": std.tolist(),
            "nan_columns": nan_columns}


def pad_players(tokens, k):
    """Players and mask padded (or cut) to k slots, so every sequence has the same shape."""
    players, mask = tokens["players"], tokens["mask"]
    t, current = mask.shape
    if current >= k:
        order = np.argsort(-mask.sum(axis=0), kind="stable")[:k]  # keep the busiest slots
        return players[:, order], mask[:, order]
    out = np.zeros((t, k, players.shape[2]), dtype=players.dtype)
    out_mask = np.zeros((t, k), dtype=bool)
    out[:, :current], out_mask[:, :current] = players, mask
    return out, out_mask


# -- saving, loading, predicting ---------------------------------------------------------------

def save(model, norm, path=ARTIFACT, extra=None):
    torch = _torch()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": model.state_dict(), "config": model.config,
                "n_features": model.features_in[0].in_features}, path)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(dict(norm, **(extra or {})), handle)


def load(path=ARTIFACT, device=None):
    """(model in eval mode, normalisation dict)."""
    torch = _torch()
    if not os.path.isfile(path):
        raise IOError("{} not found -- run scripts/train_seqmodel.py".format(path))
    # CPU unless asked: inference is canonical on CPU (rule 10); CPU and CUDA are not bit-identical.
    device = device or "cpu"
    blob = torch.load(path, map_location=device)
    model = build_model(blob["n_features"], blob["config"])
    model.load_state_dict(blob["state"])
    model.to(device).eval()
    with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as handle:
        norm = json.load(handle)
    return model, norm


def predict_steps(model, players, mask, ball, features, temperature=1.0):
    """(T, 3) log-probabilities at 5 Hz for one whole sequence, by overlapping windows."""
    torch = _torch()
    device = next(model.parameters()).device
    window = model.config["window"]
    stride = window // 2
    t = len(players)
    total = np.zeros((t, N_CLASSES), dtype=np.float64)
    weight = np.zeros(t)
    starts = list(range(0, max(1, t - window + 1), stride))
    if starts[-1] + window < t:
        starts.append(max(0, t - window))
    with torch.no_grad():
        for start in starts:
            stop = min(t, start + window)
            batch = [torch.as_tensor(a[start:stop][None], device=device)
                     for a in (players, mask, ball, features)]
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, _ = model(*batch)
            logits = logits.float()[0].cpu().numpy() / temperature
            # Triangular weights: the middle of a window has context on both sides.
            ramp = 1.0 - np.abs(np.linspace(-1, 1, stop - start)) * 0.9
            total[start:stop] += logits * ramp[:, None]
            weight[start:stop] += ramp
    logits = total / weight[:, None]
    return logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)


def upsample(step_log_probs, n_frames):
    """5 Hz log-probabilities -> every frame, linear interpolation of the logits."""
    steps = np.arange(len(step_log_probs)) * examples.TOKEN_STEP
    frames = np.arange(n_frames)
    out = np.column_stack([np.interp(frames, steps, step_log_probs[:, c])
                           for c in range(step_log_probs.shape[1])])
    return out - np.logaddexp.reduce(out, axis=1, keepdims=True)


def predict_tracking(model, norm, tracking, F=None, temperature=1.0):
    """(N, 3) per-frame log P(H), log P(A), log P(D) for a cleaned Tracking."""
    from pompeu.features.build import build_features
    if F is None or not set(norm["columns"]) <= set(F.columns):
        F = build_features(tracking, tagging=None)
    tokens = examples.player_tokens(tracking)
    players, mask = pad_players(tokens, model.config["max_players"])
    features = feature_inputs(F, norm)
    steps = predict_steps(model, players, mask, tokens["ball"], features, temperature)
    return upsample(steps, tracking.n_frames)
