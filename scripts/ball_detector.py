"""Local B2 runner. All clips, weights and detections stay under the private cache.

    python scripts/ball_detector.py prepare --video <local tactical video>
    python scripts/ball_detector.py train
    python scripts/ball_detector.py detect --video <local tactical video>

Requires the separate vision environment and Marti's B0/B1 (PR #3) for video
preparation/detection. CPU is the default; shared GPU use needs team coordination.
Prints aggregate metrics and hashes only. It never uses public possession GT.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    from pompeu.vision.ball_det.provenance import file_sha256
    return file_sha256(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "validate", "detect", "benchmark"])
    parser.add_argument("--config", default="configs/b2_temporal.yaml")
    parser.add_argument("--video", help="local video; required for prepare/detect")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--pseudolabels", help="local B1 parquet (default private cache)")
    parser.add_argument("--threads", type=int, help="override CPU thread count; recorded in checkpoints")
    args = parser.parse_args(argv)
    if args.stage in ("prepare", "validate", "detect") and not args.video:
        parser.error("--video is required for prepare/detect")
    if args.stage == "prepare" and args.device != "cpu":
        parser.error("crop preparation uses CPU only")
    if args.stage in ("validate", "detect") and args.device != "cpu":
        parser.error("canonical validation and inference require CPU")
    import yaml
    from pompeu import paths
    from pompeu.vision.ball_det.model import set_deterministic
    from pompeu.vision.ball_det.provenance import checkpoint_pin, configuration_hash, verify_sources
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    threads = args.threads if args.threads is not None else cfg.get("execution", {}).get("threads", 1)
    set_deterministic(cfg["seed"], threads=threads)
    cfg_hash = configuration_hash(cfg)
    if args.stage == "benchmark":
        benchmark(cfg, args.device, threads)
        return
    commit = subprocess.run(["git", "-c", f"safe.directory={paths.ROOT}", "rev-parse", "HEAD"],
                            cwd=paths.ROOT, capture_output=True, text=True, check=True).stdout.strip()
    root = Path(paths.CACHE_DIR)/"ball_det"/cfg_hash[:16]
    root.mkdir(parents=True, exist_ok=True)
    clips, checkpoint = root/"clips", root/"best.pt"
    if args.pseudolabels:
        label_path = Path(args.pseudolabels)
    else:
        from pompeu.vision import pseudolabels
        label_path = Path(pseudolabels.cache_path())
    if args.stage in ("prepare", "validate", "detect"):
        try:
            from pompeu.vision import decode, pseudolabels
        except ImportError:
            raise RuntimeError("B0/B1 from Marti's PR #3 must be available before video processing") from None
        from pompeu.vision.ball_det.video import decoder_version, iter_video
        info = decode.probe(args.video, count_packets=True)
    if args.stage == "prepare":
        import pandas as pd
        from pompeu.vision.ball_det.cache import prepare_cache
        table = pd.read_parquet(label_path)
        metadata = {"video_sha256": digest(args.video), "b1_sha256": digest(label_path),
                    "config_sha256": cfg_hash, "fps": info["fps"], "ffmpeg_version": decoder_version()}
        counts = prepare_cache(table, iter_video(args.video, info), clips, metadata=metadata,
                               seed=cfg["seed"], n_frames=cfg["model"]["n_frames"],
                               crop_shape=tuple(cfg["data"]["crop_shape"]),
                               stride=cfg["data"]["stride"],
                               negative_crops=cfg["data"].get("negative_crops", 0),
                               negative_margin=cfg["data"].get("negative_margin", 16),
                               block_frames=int(round(info["fps"]*pseudolabels.PARAMS["block_s"])))
        print(json.dumps({"prepared": counts, "config_sha256": cfg_hash}))
    elif args.stage == "train":
        from pompeu.vision.ball_det.cache import CachedB1Dataset
        from pompeu.vision.ball_det.model import TemporalHeatmapNet
        from pompeu.vision.ball_det.train import fit
        train, val = CachedB1Dataset(clips, "train", augment=True), CachedB1Dataset(clips, "val")
        verify_sources(train.metadata, cfg_hash, label_path, args.video)
        model = TemporalHeatmapNet(**cfg["model"])
        print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "device": args.device}))
        fit(model, train, val, checkpoint, seed=cfg["seed"], device=args.device,
            metadata={**train.metadata, "code_commit": commit, "execution": {"threads": threads}},
            report=lambda metrics: print(json.dumps(metrics)), **cfg["train"])
        print(json.dumps({"checkpoint_sha256": digest(checkpoint)}))
    else:
        from pompeu.vision.ball_det import detect_ball
        from pompeu.vision.ball_det.checkpoint import load_model
        expected = checkpoint_pin(cfg)
        model, metadata = load_model(checkpoint, expected)
        verify_sources(metadata, cfg_hash, label_path, args.video)
        if metadata.get("video_sha256") != digest(args.video):
            raise ValueError("video hash differs from training input")
        if metadata.get("ffmpeg_version") != decoder_version():
            raise ValueError("FFmpeg version differs from preparation")
        if metadata.get("execution", {}).get("threads") != threads:
            raise ValueError("CPU threads differ from the checkpoint execution settings")
        if args.stage == "validate":
            import pandas as pd
            from pompeu.vision.ball_det.predict import BallDetector
            from pompeu.vision.ball_det.validation import evaluate_frames
            table = pd.read_parquet(label_path)
            detector = BallDetector(model, crop_shape=tuple(cfg["data"]["crop_shape"]), **cfg["inference"])
            metrics = evaluate_frames(detector, iter_video(args.video, info), table,
                                       block_frames=int(round(info["fps"]*pseudolabels.PARAMS["block_s"])))
            print(json.dumps(metrics))
            return
        detections = detect_ball(iter_video(args.video, info), checkpoint=checkpoint,
                                 expected_sha256=expected, device=args.device, **cfg["inference"])
        output = root/"detections.parquet"
        detections.to_parquet(output, index=False)
        print(json.dumps({"candidates": len(detections), "frames_with_candidates": int(detections.frame.nunique()),
                          "detections_sha256": digest(output), "device": args.device}))


def benchmark(cfg, device, threads):
    """Synthetic compute smoke only: warmup plus two 1080p centre frames."""
    import numpy as np
    import torch
    from pompeu.vision.ball_det.model import TemporalHeatmapNet
    from pompeu.vision.ball_det.predict import BallDetector
    model = TemporalHeatmapNet(**cfg["model"])
    detector = BallDetector(model, crop_shape=tuple(cfg["data"]["crop_shape"]), device=device,
                             **{**cfg["inference"], "frame_stride": 1})
    video = np.zeros((cfg["model"]["n_frames"]+2, 1080, 1920, 3), np.uint8)
    timings, start = [], time.perf_counter()
    for _frame, _rows in detector.iter_detections(video):
        if device == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        timings.append(now-start)
        start = now
    print(json.dumps({"synthetic_untrained_benchmark": True, "device": device, "threads": threads,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "warmup_s": timings[0], "steady_frame_s": sum(timings[1:])/len(timings[1:]),
                      "torch_version": str(torch.__version__)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Avoid tracebacks that could expose rows/coordinates from local files.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "hint": "Check local inputs, B0/B1 availability and the vision environment."}), file=sys.stderr)
        sys.exit(1)
