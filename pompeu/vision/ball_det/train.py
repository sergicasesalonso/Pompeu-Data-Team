"""Train only B1 train rows; select checkpoints only on temporal validation.

Returns aggregate history. Checkpoints and any per-frame products stay private.
No external/public possession labels or clean_tracking calls enter this module.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import decode_heatmaps
from .checkpoint import save_checkpoint
from .model import focal_heatmap_loss


@torch.inference_mode()
def evaluate(model, loader, device="cpu", threshold=0.5, top_k=3):
    """Crop-level agreement with held-out pseudo-labels; not human-GT accuracy."""
    model.eval()
    losses, examples, positives, candidates, hits = 0.0, 0, 0, 0, {5: 0, 10: 0}
    for batch in loader:
        logits = model(batch["clip"].to(device))
        loss = focal_heatmap_loss(logits, batch["target"].to(device), batch["weight"].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite validation loss")
        size = len(logits)
        losses += float(loss) * size
        examples += size
        positives += int(torch.isfinite(batch["point"]).all(dim=1).sum())
        table = decode_heatmaps(logits.sigmoid()[:, 0].cpu().numpy(), np.arange(size), top_k, threshold)
        candidates += len(table)
        for index in range(size):
            xy = table.loc[table.frame == index, ["x_px", "y_px"]].to_numpy()
            distance = np.linalg.norm(xy - batch["point"][index].numpy(), axis=1)
            for radius in hits:
                hits[radius] += int((distance <= radius).any())
    if not examples:
        raise ValueError("empty validation set")
    metrics = {"val_loss": losses/examples, "val_crops": examples, "val_positive_crops": positives,
               "val_negative_crops": examples-positives, "val_candidates": candidates}
    for radius, count in hits.items():
        metrics[f"crop_pseudo_precision_{radius}px"] = count/candidates if candidates else 0.0
        metrics[f"crop_pseudo_recall_{radius}px"] = count/positives if positives else 0.0
    return metrics


def fit(model, train, val, checkpoint, *, seed=0, epochs=10, batch_size=8, lr=1e-3,
        device="cpu", metadata=None, report=None):
    """Seed model construction before calling; no resume or implicit GPU selection.

    num_workers=0 and explicit shuffle generator fix data order. Validation is
    never augmented. The caller's metadata should include input hashes/config.
    """
    if (train.split != "train" or val.split != "val" or val.augment
            or not len(train) or not len(val)):
        raise ValueError("nonempty train/val B1 datasets without validation augmentation required")
    if set(train.table.frame) & set(val.table.frame):
        raise ValueError("train and validation frames overlap")
    if (train.n_frames != model.config["n_frames"] or val.n_frames != train.n_frames
            or train.crop_shape != val.crop_shape or train.block_frames != val.block_frames):
        raise ValueError("model and dataset context/crop mismatch")
    if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("invalid training settings")
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, best = [], float("inf")
    for epoch in range(epochs):
        train.set_epoch(epoch)
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["clip"].to(device))
            loss = focal_heatmap_loss(logits, batch["target"].to(device), batch["weight"].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(logits)
        metrics = {"epoch": epoch+1, "train_loss": total/len(train), **evaluate(model, val_loader, device)}
        if metrics["val_loss"] < best:
            best = metrics["val_loss"]
            save_checkpoint(model, checkpoint, {**(metadata or {}), "seed": seed, "epoch": epoch+1,
                                               "crop_shape": list(train.crop_shape), "metrics": metrics,
                                               "training": {"epochs": epochs, "batch_size": batch_size,
                                                            "lr": lr, "device": str(device)}})
        history.append(metrics)
        if report is not None:
            report(metrics)
    return history
