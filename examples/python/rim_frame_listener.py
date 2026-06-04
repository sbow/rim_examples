#!/usr/bin/env python3
import argparse
import os
import socket
import struct

from rich.console import Console
from rich.table import Table

console = Console()


def decode_frame(frame: bytes) -> tuple[int, int, int, int, str]:
    source = frame[0]
    topic_id = struct.unpack_from("<H", frame, 2)[0]
    seq = struct.unpack_from("<I", frame, 4)[0]
    ts_ns = struct.unpack_from("<Q", frame, 8)[0]
    payload = frame[32:64].rstrip(b"\0").decode("utf-8", errors="replace")
    return source, topic_id, seq, ts_ns, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Receive RouterFrame packets over UDS datagrams")
    parser.add_argument("--path", default="/tmp/rim_examples_python_listener.sock")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    if os.path.exists(args.path):
        os.unlink(args.path)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(args.path)

    table = Table(title=f"RIM frames from {args.path}")
    table.add_column("source")
    table.add_column("topic")
    table.add_column("seq")
    table.add_column("timestamp_ns")
    table.add_column("payload")

    try:
        for _ in range(args.count):
            data, _ = sock.recvfrom(256)
            if len(data) < 64:
                continue
            source, topic_id, seq, ts_ns, payload = decode_frame(data[:64])
            table.add_row(str(source), str(topic_id), str(seq), str(ts_ns), payload)
    finally:
        sock.close()
        if os.path.exists(args.path):
            os.unlink(args.path)

    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
