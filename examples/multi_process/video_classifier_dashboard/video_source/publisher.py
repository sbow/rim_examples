#!/usr/bin/env python3
"""Process 1 — Python SHM video publisher (RoboticsIpcModule peer ``video_source``).

Pulls a YouTube stream with ``yt-dlp`` + ``ffmpeg`` (no OpenCV), decodes it to
raw RGB24 frames, drops each frame into a shared-memory ring, and publishes a
tiny 64 B RouterFrame *notification* through the RIM router. The notification's
sideband descriptor (``sideband_idx``/``sideband_seq``/``sideband_len``) tells
the downstream ML peer exactly which SHM slot to read — the bulk pixels never
touch the router. This is the ADR 0005 control-plane/bulk split in ~120 lines.

Dependencies (external — RIM itself stays dependency-free):
  * ``yt-dlp`` and ``ffmpeg`` on ``PATH``
  * Python 3.11+ (for ``tomllib``) — or ``pip install tomli`` on 3.10.

The RIM Python bridge (``rim_router_peer``/``rim_router_frame``) is reused
verbatim from ``RoboticsIpcModule/examples/bridges/python_peer``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# --- locate the RIM Python bridge + the shared SHM helper --------------------
_HERE = Path(__file__).resolve().parent
_COMMON = _HERE.parent / "common"
_RIM_DIR = Path(os.environ.get("RIM_DIR", _HERE.parents[4] / "RoboticsIpcModule"))
sys.path.insert(0, str(_COMMON))
sys.path.insert(0, str(_RIM_DIR / "examples" / "bridges" / "python_peer"))

from frame_shm import FrameShm  # noqa: E402
from rim_router_frame import (  # noqa: E402
    FLAG_HAS_SIDEBAND,
    FLAG_KEYFRAME,
    RouterFrame,
)
from rim_router_peer import RouterPeer, router_now_ns  # noqa: E402

PEER_ID = 1
TOPIC_RAW_FRAME = 10
# Packed into the 32 B inline payload: width, height, channels, capture_ns, fps.
_META_FMT = "<IIIQf"


def _uds_path(addr: str) -> str:
    """Strip the ``uds:`` scheme from a topology address."""
    return addr.split(":", 1)[1] if addr.startswith("uds:") else addr


def resolve_stream_url(youtube_url: str, fmt: str) -> str:
    """Resolve a playable media URL with ``yt-dlp -g`` (handles live + VOD)."""
    out = subprocess.check_output(
        ["yt-dlp", "-q", "--no-warnings", "-f", fmt, "-g", youtube_url],
        text=True,
    )
    # For muxed/HLS, -g prints one URL; for split A/V it prints two — take the
    # first (video) line, which ffmpeg will decode for frames.
    return out.strip().splitlines()[0]


def start_ffmpeg(url: str, width: int, height: int, fps: int) -> subprocess.Popen:
    """Spawn ffmpeg decoding ``url`` to rawvideo rgb24 on stdout."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        # Pace input at its native frame rate. Without this, ffmpeg races through
        # each freshly-downloaded HLS segment (faster-than-realtime burst) and
        # then stalls while fetching the next one — the "chunky" sprint/freeze
        # cycle. -re emits frames steadily and lets the next segment prefetch.
        "-re",
        "-i", url,
        "-an",                                   # no audio
        "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)


def _read_frame(stream, n: int) -> bytes:
    """Read exactly ``n`` bytes (one full RGB frame) from ffmpeg's stdout.

    ffmpeg's stdout is an unbuffered pipe, so a single ``read(n)`` returns only
    what one ``read()`` syscall yields — usually a partial frame (~one pipe
    buffer). Loop until a whole frame is assembled, or return short on real EOF
    (ffmpeg exited / stream ended).
    """
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_HERE.parent / "topology.toml"))
    ap.add_argument("--url", default=None, help="override [video].youtube_url")
    args = ap.parse_args()

    if not shutil.which("yt-dlp") or not shutil.which("ffmpeg"):
        sys.exit("error: yt-dlp and ffmpeg must be installed and on PATH")

    cfg = tomllib.loads(Path(args.config).read_text())
    video = cfg.get("video", {})
    width = int(video.get("width", 640))
    height = int(video.get("height", 360))
    fps = int(video.get("fps", 15))
    youtube_url = args.url or video["youtube_url"]
    yt_format = video.get("yt_format", "best[height<=720]")

    peer = next(p for p in cfg["peers"] if p["id"] == PEER_ID)
    peer_path = _uds_path(peer["local"])
    router_path = _uds_path(cfg["router"]["listen"])
    shm_name = peer["sideband"][0]["name"]

    slot_bytes = width * height * 3
    print(f"[video_source] {youtube_url} -> {width}x{height}@{fps} "
          f"shm={shm_name} ({slot_bytes} B/frame)", flush=True)

    ring = FrameShm(shm_name, width, height, channels=3, slot_count=4, create=True)
    peer = RouterPeer(router_path=router_path, peer_path=peer_path, peer_id=PEER_ID)

    stop = False

    def _sig(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    proc = None
    frame_index = 0
    last_fps_t = time.monotonic()
    fps_count = 0
    cur_fps = 0.0
    try:
        try:
            url = resolve_stream_url(youtube_url, yt_format)
        except subprocess.CalledProcessError:
            print(
                f"[video_source] yt-dlp could not extract a stream from {youtube_url}\n"
                "  - the stream may have ended or be region-locked / members-only, or\n"
                "  - yt-dlp is out of date (YouTube changes often). Try, in order:\n"
                "      .venv/bin/pip install -U yt-dlp\n"
                "      .venv/bin/pip install -U --pre 'yt-dlp[default]'   # nightly\n"
                "  - or pick another source:  ./launch.sh <youtube_url>",
                file=sys.stderr, flush=True)
            return 1
        proc = start_ffmpeg(url, width, height, fps)
        assert proc.stdout is not None

        while not stop:
            buf = _read_frame(proc.stdout, slot_bytes)
            if len(buf) < slot_bytes:
                print("[video_source] stream ended/stalled, exiting", flush=True)
                break

            shm_seq = ring.write(buf)

            now = router_now_ns()
            meta = struct.pack(_META_FMT, width, height, 3, now, cur_fps)
            frame = RouterFrame.make(
                source=PEER_ID,
                topic_id=TOPIC_RAW_FRAME,
                seq=frame_index & 0xFFFFFFFF,
                timestamp_ns=now,
                flags=FLAG_HAS_SIDEBAND | FLAG_KEYFRAME,
                payload=meta,
                sideband_idx=0,
                sideband_seq=shm_seq,
                sideband_len=slot_bytes,
            )
            peer.send_frame(frame)

            frame_index += 1
            fps_count += 1
            dt = time.monotonic() - last_fps_t
            if dt >= 1.0:
                cur_fps = fps_count / dt
                fps_count = 0
                last_fps_t = time.monotonic()
                print(f"[video_source] published {frame_index} frames "
                      f"({cur_fps:.1f} fps)", flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        peer.close()
        ring.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
