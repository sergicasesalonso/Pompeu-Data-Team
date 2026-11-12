"""Local B2 runner. All clips, weights and detections stay under the private cache.

    python scripts/ball_detector.py prepare --video <local tactical video>
    python scripts/ball_detector.py train
    python scripts/ball_detector.py detect --video <local tactical video>

Requires the separate vision environment and Marti's B0/B1 (PR #3) for video
preparation/detection. CPU is the default; shared GPU use needs team coordination.
Prints aggregate metrics and hashes only. It never uses public possession GT.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "detect"])
    parser.add_argument("--config", default="configs/b2_temporal.yaml")
    parser.add_argument("--video", help="local video; required for prepare/detect")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args(argv)
    if args.stage != "train" and not args.video:
        parser.error("--video is required for prepare/detect")
    if args.stage == "prepare" and args.device != "cpu":
        parser.error("crop preparation uses CPU only")
    import yaml
    from pompeu import paths
    from pompeu.vision.ball_det.model import set_deterministic
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_deterministic(cfg["seed"])
    cfg_hash = digest(args.config)
    commit = subprocess.run(["git", "-c", f"safe.directory={paths.ROOT}", "rev-parse", "HEAD"],
                            cwd=paths.ROOT, capture_output=True, text=True, check=True).stdout.strip()
    root = Path(paths.CACHE_DIR)/"ball_det"/cfg_hash[:16]
    root.mkdir(parents=True, exist_ok=True)
    clips, checkpoint = root/"clips", root/"best.pt"
    if args.stage in ("prepare", "detect"):
        try:
            from pompeu.vision import decode, pseudolabels
        except ImportError:
            raise RuntimeError("B0/B1 from Marti's PR #3 must be available before video processing") from None
        from pompeu.vision.ball_det.video import iter_video
        info = decode.probe(args.video, count_packets=True)
    if args.stage == "prepare":
        import pandas as pd
        from pompeu.vision.ball_det.cache import prepare_cache
        label_path = pseudolabels.cache_path()
        table = pd.read_parquet(label_path)
        metadata = {"video_sha256": digest(args.video), "b1_sha256": digest(label_path),
                    "config_sha256": cfg_hash, "fps": info["fps"]}
        counts = prepare_cache(table, iter_video(args.video, info), clips, metadata=metadata,
                               seed=cfg["seed"], n_frames=cfg["model"]["n_frames"],
                               crop_shape=tuple(cfg["data"]["crop_shape"]),
                               stride=cfg["data"]["stride"],
                               block_frames=int(round(info["fps"]*pseudolabels.PARAMS["block_s"])))
        print(json.dumps({"prepared": counts, "config_sha256": cfg_hash}))
    elif args.stage == "train":
        from pompeu.vision.ball_det.cache import CachedB1Dataset
        from pompeu.vision.ball_det.model import TemporalHeatmapNet
        from pompeu.vision.ball_det.train import fit
        train, val = CachedB1Dataset(clips, "train", augment=True), CachedB1Dataset(clips, "val")
        if train.metadata.get("config_sha256") != cfg_hash:
            raise ValueError("crop cache configuration hash differs")
        model = TemporalHeatmapNet(**cfg["model"])
        print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "device": args.device}))
        fit(model, train, val, checkpoint, seed=cfg["seed"], device=args.device,
            metadata={**train.metadata, "code_commit": commit},
            report=lambda metrics: print(json.dumps(metrics)), **cfg["train"])
        print(json.dumps({"checkpoint_sha256": digest(checkpoint)}))
    else:
        from pompeu.vision.ball_det import detect_ball
        from pompeu.vision.ball_det.checkpoint import load_model
        _, metadata = load_model(checkpoint)
        if metadata.get("video_sha256") != digest(args.video):
            raise ValueError("video hash differs from training input")
        expected = checkpoint.with_suffix(".pt.sha256").read_text(encoding="ascii").strip()
        detections = detect_ball(iter_video(args.video, info), checkpoint=checkpoint,
                                 expected_sha256=expected, device=args.device, **cfg["inference"])
        output = root/"detections.parquet"
        detections.to_parquet(output, index=False)
        print(json.dumps({"candidates": len(detections), "frames_with_candidates": int(detections.frame.nunique()),
                          "detections_sha256": digest(output), "device": args.device}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Avoid tracebacks that could expose rows/coordinates from local files.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "hint": "Check local inputs, B0/B1 availability and the vision environment."}), file=sys.stderr)
        sys.exit(1)
