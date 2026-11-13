"""Synthetic training resource/determinism smoke. Never loads or stores match data."""

import hashlib
import time


def training_smoke(cfg, device="cpu", steps=2, threads=1):
    import torch
    from .model import TemporalHeatmapNet, focal_heatmap_loss, heatmap_targets, set_deterministic
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("positive smoke step count required")
    def run():
        set_deterministic(cfg["seed"], threads=threads)
        model = TemporalHeatmapNet(**cfg["model"]).to(device)
        batch, frames = cfg["train"]["batch_size"], cfg["model"]["n_frames"]
        h, w = cfg["data"]["crop_shape"]
        x = torch.rand(batch, frames, 3, h, w, device=device)
        target = heatmap_targets([[w//2, h//2]]*batch, h, w).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = focal_heatmap_loss(model(x), target)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite synthetic smoke loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return (digest.hexdigest(), losses, seconds/steps,
                torch.cuda.max_memory_allocated() if device == "cuda" else None,
                torch.cuda.max_memory_reserved() if device == "cuda" else None)
    first, second = run(), run()
    return {"synthetic_training_smoke": True, "device": device, "steps_per_run": steps,
            "batch_size": cfg["train"]["batch_size"], "torch_version": str(torch.__version__),
            "state_sha256": first[0], "repeat_state_sha256": second[0],
            "exact_repeatability": first[:2] == second[:2], "seconds_per_step": first[2],
            "peak_cuda_allocated_bytes": max(first[3], second[3]) if device == "cuda" else None,
            "peak_cuda_reserved_bytes": max(first[4], second[4]) if device == "cuda" else None}
