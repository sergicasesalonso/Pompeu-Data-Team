"""Small temporal heatmap network. PyTorch is optional outside Line B.

An original encoder/decoder, trained from scratch; no pretrained weights or
third-party model code. Inputs are RGB clips (B,T,3,H,W), scaled to [0,1].
The output is one full-resolution logit map for each clip's centre frame.
"""

import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def set_deterministic(seed=0, threads=1):
    """Repeatability within one pinned software/hardware environment, not CPU=GPU."""
    if not isinstance(seed, int) or not 0 <= seed < 2**32 or threads < 1:
        raise ValueError("invalid seed or thread count")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class TemporalHeatmapNet(nn.Module):
    """Two downsampling stages with skip connections; no batch-state statistics."""

    def __init__(self, n_frames=3, width=16):
        super().__init__()
        if n_frames not in (3, 5) or not isinstance(width, int) or width < 2:
            raise ValueError("expected 3 or 5 frames and width >= 2")
        self.config = {"n_frames": n_frames, "width": width}
        self.enc1 = _block(n_frames * 3, width)
        self.enc2 = _block(width, width * 2)
        self.bridge = _block(width * 2, width * 4)
        self.dec2 = _block(width * 6, width * 2)
        self.dec1 = _block(width * 3, width)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.constant_(self.head.bias, -4.6)  # sparse positives at initialisation

    def forward(self, clips):
        if (clips.ndim != 5 or clips.shape[1:3] != (self.config["n_frames"], 3)
                or min(clips.shape[-2:]) < 4 or not clips.is_floating_point()):
            raise ValueError("expected floating RGB clips (B,T,3,H,W), H,W >= 4")
        a = self.enc1(clips.flatten(1, 2))
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.bridge(F.avg_pool2d(b, 2))
        d = self.dec2(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="nearest"), b), 1))
        return self.head(self.dec1(torch.cat(
            (F.interpolate(d, size=a.shape[-2:], mode="nearest"), a), 1)))


def heatmap_targets(points, height, width, sigma=2.0):
    """(N,2) crop-pixel centres -> (N,1,H,W) Gaussians; NaN pairs mean no ball.

    Centres use nearest-pixel rounding so every positive has a target exactly 1.
    A missing ATD ball must NOT be passed here as a known-negative training label.
    """
    points = torch.as_tensor(points, dtype=torch.float32)
    if (points.ndim != 2 or points.shape[1] != 2 or height < 1 or width < 1
            or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("invalid points or target dimensions")
    present = torch.isfinite(points).all(1)
    absent = torch.isnan(points).all(1)
    inside = (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
    if not ((present & inside) | absent).all():
        raise ValueError("each target must be inside the crop or a NaN pair")
    centre = torch.floor(torch.nan_to_num(points, nan=0.0) + 0.5)
    centre[:, 0].clamp_(0, width - 1)
    centre[:, 1].clamp_(0, height - 1)
    y = torch.arange(height, device=points.device)[None, :, None]
    x = torch.arange(width, device=points.device)[None, None, :]
    distance = (x - centre[:, 0, None, None])**2 + (y - centre[:, 1, None, None])**2
    return (torch.exp(-distance / (2 * sigma**2)) * present[:, None, None])[:, None]


def focal_heatmap_loss(logits, target, weights=None):
    """Stable focal loss, normalised per image; soft Gaussian shoulders reduce negatives."""
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("expected equal (B,1,H,W) logits and targets")
    if not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("targets must be finite and in [0,1]")
    p = logits.sigmoid()
    positive = target.eq(1)
    loss = (-F.logsigmoid(logits) * (1-p)**2 * positive
            - F.logsigmoid(-logits) * p**2 * (1-target)**4 * ~positive)
    per_image = loss.flatten(1).sum(1) / positive.flatten(1).sum(1).clamp_min(1)
    if weights is None:
        return per_image.mean()
    weights = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
    if weights.shape != per_image.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("expected one finite positive weight per image")
    return (per_image * weights).sum() / weights.sum()
