"""
Process 1: Video Publisher

Captures frames from an online video source and publishes them to shared memory
for consumption by downstream processes (classifier, dashboard).
"""

import argparse
import ctypes
import mmap
import os
import signal
import struct
import sys
import time
from typing import Optional

import cv2
import numpy as np

# Shared memory constants (must match router/main.cpp)
SHM_RAW_FRAMES = "/vcf_raw_frames"
SHM_HEADER_SIZE = 64  # Matches ShmHeader struct size with padding

# ShmHeader layout:
#   uint64_t write_sequence       (8 bytes)
#   uint64_t frame_timestamp_ns   (8 bytes)
#   uint32_t frame_width          (4 bytes)
#   uint32_t frame_height         (4 bytes)
#   uint32_t frame_channels       (4 bytes)
#   uint32_t data_size            (4 bytes)
#   sem_t write_lock              (32 bytes on Linux)
#   sem_t read_ready              (32 bytes on Linux)
HEADER_FORMAT = "<QQIIIi"  # Partial header for the fields we write
HEADER_FIELDS_SIZE = struct.calcsize(HEADER_FORMAT)

running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    running = False


class ShmPublisher:
    """Publishes video frames to a POSIX shared memory segment."""

    def __init__(self, shm_name: str, segment_size: int):
        self.shm_name = shm_name
        self.segment_size = segment_size
        self.fd: Optional[int] = None
        self.mm: Optional[mmap.mmap] = None
        self.sequence = 0

    def connect(self) -> bool:
        """Connect to an existing shared memory segment created by the router."""
        try:
            self.fd = os.open(
                f"/dev/shm{self.shm_name}", os.O_RDWR
            )
            self.mm = mmap.mmap(self.fd, self.segment_size)
            print(f"[VideoPublisher] Connected to SHM: {self.shm_name}")
            return True
        except (OSError, ValueError) as e:
            print(f"[VideoPublisher] Failed to connect to SHM: {e}")
            return False

    def publish_frame(self, frame: np.ndarray) -> bool:
        """Write a video frame into shared memory."""
        if self.mm is None:
            return False

        height, width, channels = frame.shape
        data = frame.tobytes()
        data_size = len(data)

        # Ensure frame fits in segment (minus header)
        if data_size + SHM_HEADER_SIZE > self.segment_size:
            print(
                f"[VideoPublisher] Frame too large: {data_size} bytes "
                f"(max {self.segment_size - SHM_HEADER_SIZE})"
            )
            return False

        self.sequence += 1
        timestamp_ns = int(time.time_ns())

        # Write header fields
        header = struct.pack(
            HEADER_FORMAT,
            self.sequence,
            timestamp_ns,
            width,
            height,
            channels,
            data_size,
        )

        self.mm.seek(0)
        self.mm.write(header)

        # Write frame data after header
        self.mm.seek(SHM_HEADER_SIZE)
        self.mm.write(data)

        return True

    def close(self):
        """Close the shared memory mapping."""
        if self.mm:
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class VideoCapture:
    """Captures frames from an online video source."""

    def __init__(self, source: str):
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.stream_url: Optional[str] = None

    def open(self) -> bool:
        """Open the video source. Supports direct URLs and YouTube via yt-dlp."""
        url = self._resolve_url(self.source)
        if url is None:
            return False

        self.stream_url = url
        self.cap = cv2.VideoCapture(url)

        if not self.cap.isOpened():
            print(f"[VideoPublisher] Failed to open video stream: {url}")
            return False

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(
            f"[VideoPublisher] Opened stream: {width}x{height} @ {fps:.1f} FPS"
        )
        return True

    def read(self) -> Optional[np.ndarray]:
        """Read the next frame from the video source."""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            # Attempt reconnection for live streams
            print("[VideoPublisher] Stream ended or error, reconnecting...")
            self.cap.release()
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            if not ret:
                return None

        return frame

    def release(self):
        """Release the video capture."""
        if self.cap:
            self.cap.release()
            self.cap = None

    @staticmethod
    def _is_youtube_url(url: str) -> bool:
        """Check if a URL is a YouTube link by parsing the hostname."""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            return hostname in (
                "youtube.com",
                "www.youtube.com",
                "m.youtube.com",
                "youtu.be",
            )
        except ValueError:
            return False

    @staticmethod
    def _resolve_url(source: str) -> Optional[str]:
        """Resolve a video URL. Uses yt-dlp for YouTube links."""
        if VideoCapture._is_youtube_url(source):
            try:
                import yt_dlp

                ydl_opts = {
                    "format": "best[height<=720]",
                    "quiet": True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source, download=False)
                    url = info.get("url")
                    if url:
                        print(f"[VideoPublisher] Resolved YouTube URL")
                        return url
            except ImportError:
                print("[VideoPublisher] yt-dlp not installed, trying direct URL")
            except Exception as e:
                print(f"[VideoPublisher] yt-dlp error: {e}")
                return None

        # Direct URL (RTSP, HTTP stream, file path)
        return source


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="Video Publisher - Stream video to shared memory"
    )
    parser.add_argument(
        "--source",
        default=os.environ.get(
            "VIDEO_SOURCE",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ),
        help="Video source URL (YouTube, RTSP, HTTP, or file path)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target publish rate in frames per second",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=640,
        help="Resize frame width (0 = no resize)",
    )
    args = parser.parse_args()

    print("=== Video Classifier Dashboard - Video Publisher (Process 1) ===")
    print(f"[VideoPublisher] Source: {args.source}")
    print(f"[VideoPublisher] Target FPS: {args.fps}")

    # Connect to shared memory created by router
    segment_size = int(os.environ.get("SHM_SEGMENT_SIZE", 8 * 1024 * 1024))
    publisher = ShmPublisher(SHM_RAW_FRAMES, segment_size)

    # Retry connection until router is ready
    while running and not publisher.connect():
        print("[VideoPublisher] Waiting for router...")
        time.sleep(1.0)

    if not running:
        return

    # Open video source
    capture = VideoCapture(args.source)
    if not capture.open():
        print("[VideoPublisher] Failed to open video source. Exiting.")
        publisher.close()
        return

    # Publishing loop
    frame_interval = 1.0 / args.fps
    frames_published = 0
    start_time = time.time()

    print("[VideoPublisher] Publishing frames...")

    while running:
        loop_start = time.time()

        frame = capture.read()
        if frame is None:
            print("[VideoPublisher] No frame available, retrying...")
            time.sleep(0.5)
            continue

        # Resize if requested
        if args.resize_width > 0:
            h, w = frame.shape[:2]
            scale = args.resize_width / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (args.resize_width, new_h))

        # Publish to shared memory
        if publisher.publish_frame(frame):
            frames_published += 1

            if frames_published % 100 == 0:
                elapsed = time.time() - start_time
                actual_fps = frames_published / elapsed if elapsed > 0 else 0
                print(
                    f"[VideoPublisher] Published {frames_published} frames "
                    f"({actual_fps:.1f} FPS actual)"
                )

        # Rate limiting
        elapsed = time.time() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Cleanup
    capture.release()
    publisher.close()
    elapsed = time.time() - start_time
    print(
        f"[VideoPublisher] Finished. Published {frames_published} frames "
        f"in {elapsed:.1f}s ({frames_published / elapsed:.1f} FPS)"
    )


if __name__ == "__main__":
    main()
