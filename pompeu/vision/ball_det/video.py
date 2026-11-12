"""Sequential local FFmpeg decoding by ordinal frame; no timestamp seeking."""

import os
import shutil
import subprocess

import numpy as np


def iter_video(path, info):
    """Yield (zero-based frame, RGB uint8), given metadata from Marti's B0 probe.

    No frames or decoder stderr are printed. Closing the iterator terminates the
    decoder. A short raw frame or nonzero exit is an error, never silent EOF.
    """
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("FFmpeg is required for local B2 video processing")
    width, height = info["width"], info["height"]
    if min(width, height) < 1:
        raise ValueError("invalid B0 video dimensions")
    command = [executable, "-nostdin", "-v", "error", "-threads", "1", "-i", str(path),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0", "-threads", "1",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    count, size = 0, width*height*3
    try:
        while True:
            data = bytearray()
            while len(data) < size:
                part = process.stdout.read(size-len(data))
                if not part:
                    break
                data.extend(part)
            if not data:
                break
            if len(data) != size:
                raise RuntimeError("incomplete decoded video frame")
            yield count, np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()
            count += 1
        if process.wait() != 0:
            raise RuntimeError("local video decoding failed")
        expected = info.get("frames_counted") or info.get("frames_header")
        if expected is not None and count != expected:
            raise RuntimeError("decoded frame count differs from B0")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
