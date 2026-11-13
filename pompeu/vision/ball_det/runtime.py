"""Read-only dependency checks plus a disposable cache write probe; no match rows."""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from .provenance import file_sha256


def check_lock(path, installed=None):
    """Compare every pinned requirement; platform-only extra packages are reported."""
    normal = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    if installed is None:
        installed = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}
    packages = {normal(k): v for k, v in installed.items()}
    expected = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9+_.-]+)", line)
        if match is None:
            raise ValueError("unsupported lock entry")
        expected[normal(match[1])] = match[2]
    missing = sorted(set(expected)-set(packages))
    mismatches = [{"package": k, "expected": v, "actual": packages[k]}
                  for k, v in sorted(expected.items()) if k in packages and packages[k] != v]
    blob = json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    return {"lock_sha256": file_sha256(path), "locked_packages": len(expected),
            "missing_packages": missing, "version_mismatches": mismatches,
            "extra_package_count": len(set(packages)-set(expected)),
            "environment_sha256": hashlib.sha256(blob).hexdigest()}


def runtime_status(lock_path, cache_dir, device="cpu"):
    if device not in ("cpu", "cuda"):
        raise ValueError("unsupported device")
    report = check_lock(lock_path)
    report.update(python_version=sys.version.split()[0], python_311=sys.version_info[:2] == (3, 11), device=device)
    writable = False
    try:
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory, prefix=".b2-preflight-") as handle:
            handle.write(b"synthetic write probe")
            handle.flush()
        writable = True
        report["cache_free_bytes"] = shutil.disk_usage(directory).free
    except OSError:
        report["cache_free_bytes"] = None
    report["cache_writable"] = writable
    ready = report["python_311"] and not report["missing_packages"] and not report["version_mismatches"] and writable
    # An incompatible active environment (e.g. the TCN venv) is not imported or changed.
    if ready:
        try:
            import torch
            report["torch_version"], report["cuda_build"] = str(torch.__version__), torch.version.cuda
            if device == "cuda":
                report["cuda_available"] = bool(torch.cuda.is_available())
                ready = report["cuda_available"]
                executable = shutil.which("nvidia-smi")
                driver = None
                if executable:
                    result = subprocess.run([executable, "--query-gpu=driver_version", "--format=csv,noheader"],
                                            capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and result.stdout.strip():
                        driver = result.stdout.splitlines()[0].strip()
                report["driver_version"] = driver
                report["driver_560_or_newer"] = bool(driver and driver.split(".")[0].isdigit()
                                                       and int(driver.split(".")[0]) >= 560)
                ready = ready and report["driver_560_or_newer"]
        except (ImportError, OSError, RuntimeError, subprocess.TimeoutExpired):
            report["torch_runtime_status"] = "unavailable"
            ready = False
    report["ready"] = bool(ready)
    return report


def input_plan(table, cfg, fps, block_s):
    """Count eligible B1 examples and raw payload without returning per-frame data."""
    from .dataset import B1ClipDataset
    selected = table.loc[table.frame % cfg["data"]["stride"] == 0]
    datasets = [B1ClipDataset(selected, None, split, crop_shape=tuple(cfg["data"]["crop_shape"]),
                              n_frames=cfg["model"]["n_frames"], block_frames=int(round(fps*block_s)),
                              negative_crops=cfg["data"].get("negative_crops", 0),
                              negative_margin=cfg["data"].get("negative_margin", 16)) for split in ("train", "val")]
    samples = {data.split: len(data) for data in datasets}
    payload = sum(samples.values())*cfg["model"]["n_frames"]*3
    for length in cfg["data"]["crop_shape"]:
        payload *= length
    val_stride = cfg.get("validation", {}).get("frame_stride", cfg.get("inference", {}).get("frame_stride", 1))
    if not isinstance(val_stride, int) or val_stride < 1:
        raise ValueError("positive validation stride required")
    full_val = B1ClipDataset(table, None, "val", crop_shape=tuple(cfg["data"]["crop_shape"]),
                             n_frames=cfg["model"]["n_frames"], block_frames=int(round(fps*block_s))).table
    return {"eligible_b1_centres": {data.split: len(data.table) for data in datasets},
            "crop_samples": samples, "raw_crop_payload_bytes": payload,
            "full_frame_validation_centres": int((full_val.frame % val_stride == 0).sum()),
            "validation_frame_stride": val_stride,
            "train_batches_per_epoch": (samples["train"]+cfg["train"]["batch_size"]-1)//cfg["train"]["batch_size"]}


def preflight(cfg, lock_path, run_root, device="cpu", video=None, b1_path=None):
    report = runtime_status(lock_path, run_root, device)
    report["inputs_checked"] = video is not None
    if video is None:
        return report
    report["video_present"] = Path(video).is_file()
    report["b1_present"] = b1_path is not None and Path(b1_path).is_file()
    report["ffmpeg_present"] = shutil.which("ffmpeg") is not None
    report["ffprobe_present"] = shutil.which("ffprobe") is not None
    if not report["ready"] or not all(report[k] for k in ("video_present", "b1_present", "ffmpeg_present", "ffprobe_present")):
        report["ready"] = False
        return report
    import pandas as pd
    from pompeu.vision import decode, pseudolabels
    from .video import decoder_version
    info = decode.probe(video, count_packets=False)
    table = pd.read_parquet(b1_path)
    report.update(input_plan(table, cfg, info["fps"], pseudolabels.PARAMS["block_s"]))
    report["video_sha256"], report["b1_sha256"] = file_sha256(video), file_sha256(b1_path)
    report["ffmpeg_version"] = decoder_version()
    ready = Path(run_root)/"clips"/"ready.json"
    report["cache_exists"] = (Path(run_root)/"clips").exists()
    report["cache_complete"] = False
    if ready.is_file():
        from .provenance import configuration_hash
        try:
            cached = json.loads(ready.read_text(encoding="utf-8"))
            expected = {"video_sha256": report["video_sha256"], "b1_sha256": report["b1_sha256"],
                        "config_sha256": configuration_hash(cfg), "ffmpeg_version": report["ffmpeg_version"]}
            report["cache_complete"] = (cached.get("version") == 2
                and all(cached.get("metadata", {}).get(k) == v for k, v in expected.items())
                and cached.get("counts") == report["crop_samples"])
        except (OSError, ValueError):
            pass
    report["minimum_new_cache_free_bytes"] = report["raw_crop_payload_bytes"] + 1024**3
    # Existing complete caches are checked against their sources during train.
    report["space_for_new_cache"] = report["cache_free_bytes"] >= report["minimum_new_cache_free_bytes"]
    report["ready"] = (all(v > 0 for v in report["crop_samples"].values())
                       and (report["cache_complete"] or
                            (not report["cache_exists"] and report["space_for_new_cache"])))
    return report
