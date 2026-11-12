"""Consume B1's labels unchanged; crop and augment only on the local machine."""

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .model import heatmap_targets


def augment_clip(tensor, point, rng):
    """One shared geometric/photometric transform for all frames and the label."""
    point = list(point)
    tensor = (tensor * float(rng.uniform(0.7, 1.3))).clamp(0, 1)
    if rng.random() < 0.5:
        tensor = tensor.flip(-1)
        point[0] = tensor.shape[-1]-1-point[0]
    if rng.random() < 0.5:
        horizontal = rng.random() < 0.5
        shape = (1, 3) if horizontal else (3, 1)
        kernel = torch.full((3, 1, *shape), 1/3)
        padding = (0, 1) if horizontal else (1, 0)
        tensor = F.conv2d(tensor, kernel, padding=padding, groups=3)
    return tensor.contiguous(), point


class B1ClipDataset(Dataset):
    """`read_clip(frame, n_frames)` returns consecutive RGB uint8 frames (T,H,W,3).

    B1 owns the split. `block_frames` must match B1's block length: this only
    discards contexts crossing a block boundary; it never creates new splits.
    Labels without an observed ball are not negative examples. Every crop here
    contains the observed ball. Validation measures agreement with B1, not GT.
    """

    def __init__(self, table, read_clip, split, *, crop_shape=(256, 448), n_frames=3,
                 block_frames=1500, seed=0, augment=False):
        required = {"frame", "x_px", "y_px", "crowded", "split"}
        if not required.issubset(table.columns) or split not in ("train", "val"):
            raise ValueError("expected B1 columns and a train/val split")
        if n_frames not in (3, 5) or not isinstance(block_frames, int) or block_frames < n_frames:
            raise ValueError("invalid temporal context or B1 block length")
        if len(crop_shape) != 2 or any(not isinstance(v, int) or v < 8 for v in crop_shape):
            raise ValueError("crop dimensions must be integers >= 8")
        if (not np.issubdtype(table.frame.dtype, np.integer) or table.frame.duplicated().any()
                or (table.frame < 0).any() or not table.split.isin(["train", "val"]).all()
                or not np.isfinite(table[["x_px", "y_px"]].to_numpy()).all()
                or not table.crowded.isin([True, False]).all()):
            raise ValueError("invalid B1 table")
        if augment and split != "train":
            raise ValueError("validation augmentation is forbidden")
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("invalid seed")
        if table.groupby(table.frame // block_frames)["split"].nunique().gt(1).any():
            raise ValueError("B1 block length does not match the supplied split")
        radius = n_frames // 2
        frames = table.frame.to_numpy()
        safe = (frames >= radius) & ((frames-radius)//block_frames == (frames+radius)//block_frames)
        self.table = table.loc[safe & (table.split == split)].sort_values("frame").reset_index(drop=True)
        self.read_clip, self.split, self.crop_shape = read_clip, split, tuple(crop_shape)
        self.n_frames, self.seed, self.augment, self.epoch = n_frames, seed, augment, 0
        self.block_frames = block_frames

    def __len__(self):
        return len(self.table)

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0 or (self.split == "val" and epoch != 0):
            raise ValueError("invalid epoch; validation crops must remain fixed")
        self.epoch = epoch

    def __getitem__(self, index):
        row = self.table.iloc[index]
        frame = int(row.frame)
        clip = np.asarray(self.read_clip(frame, self.n_frames))
        if (clip.ndim != 4 or clip.shape[0] != self.n_frames or clip.shape[-1] != 3
                or clip.dtype != np.uint8 or min(clip.shape[1:3]) < 1):
            raise ValueError("clip reader must return temporal RGB uint8 data")
        h, w = clip.shape[1:3]
        x, y = float(row.x_px), float(row.y_px)
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("B1 coordinate is outside the decoded image")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, frame]))
        ch, cw = self.crop_shape

        def origin(coordinate, length, crop):
            low = max(0, int(np.floor(coordinate)) - crop + 1)
            high = min(int(np.floor(coordinate)), max(0, length-crop))
            return int(rng.integers(low, high+1))

        left, top = origin(x, w, cw), origin(y, h, ch)
        pixels = np.zeros((self.n_frames, ch, cw, 3), dtype=np.uint8)
        cut = clip[:, top:top+ch, left:left+cw]
        pixels[:, :cut.shape[1], :cut.shape[2]] = cut
        tensor = torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy()).float() / 255
        point = [float(np.clip(x-left, 0, cw-1)), float(np.clip(y-top, 0, ch-1))]
        if self.augment:
            tensor, point = augment_clip(tensor, point, rng)
        return {"clip": tensor.contiguous(), "target": heatmap_targets([point], ch, cw)[0],
                "point": torch.tensor(point), "weight": 0.5 if row.crowded else 1.0,
                "frame": frame}
