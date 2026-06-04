"""Cross-language shared-memory frame ring (Python port of ``frame_shm.hpp``).

This is the bulk-data side of the RoboticsIpcModule control-plane/bulk split:
the 64 B RouterFrame announces "frame N ready"; the RGB pixels themselves live
in a named ``/dev/shm`` region declared as a ``[[peers.sideband]]`` in the
topology (ADR 0005). The layout is byte-identical to the C++ and Node ports so
all three languages share one ring.

Layout (little-endian)::

    offset  size  field
    ------  ----  --------------------------------------------------------
       0     4    magic        b'RVF1'
       4     4    version
       8     4    width
      12     4    height
      16     4    channels
      20     4    slot_count
      24     4    slot_bytes   = width*height*channels
      28     4    reserved
      32     8    latest_seq   1-based; 0 = no frame yet (the seqlock)
      40    24    reserved (header padded to 64 B)
      64   slot_count*slot_bytes   frame slots
"""

from __future__ import annotations

import mmap
import os
import struct

MAGIC = 0x31465652  # b'RVF1' little-endian
VERSION = 1
HEADER_BYTES = 64
_HEADER_FMT = "<8I"  # magic..reserved0 (8 x uint32); latest_seq read separately
_LATEST_OFFSET = 32


def _dev_shm_path(name: str) -> str:
    """Map a POSIX shm name ('/rim_vcd_source_frames') to its /dev/shm path."""
    return "/dev/shm/" + name.lstrip("/")


class FrameShm:
    """Single-writer / many-reader RGB frame ring backed by ``/dev/shm``.

    :param name: POSIX shm name, e.g. ``/rim_vcd_source_frames``.
    :param width: frame width in pixels (producer only).
    :param height: frame height in pixels (producer only).
    :param channels: bytes per pixel, 3 for RGB24 (producer only).
    :param slot_count: ring depth; use >= 3 so readers never tear (producer).
    :param create: ``True`` on the producer (creates + sizes the region),
        ``False`` on a consumer (opens an existing region and reads its header).
    """

    def __init__(self, name, width=0, height=0, channels=3, slot_count=3, create=True):
        self.name = name
        self._owns = create
        path = _dev_shm_path(name)

        if create:
            self.slot_bytes = width * height * channels
            total = HEADER_BYTES + self.slot_bytes * slot_count
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, total)
            self._mm = mmap.mmap(fd, total)
            os.close(fd)
            self.width, self.height, self.channels, self.slot_count = (
                width, height, channels, slot_count)
            struct.pack_into(_HEADER_FMT, self._mm, 0, MAGIC, VERSION, width,
                             height, channels, slot_count, self.slot_bytes, 0)
            self._write_latest(0)
        else:
            fd = os.open(path, os.O_RDWR)
            size = os.fstat(fd).st_size
            self._mm = mmap.mmap(fd, size)
            os.close(fd)
            (magic, _ver, self.width, self.height, self.channels,
             self.slot_count, self.slot_bytes, _r) = struct.unpack_from(
                _HEADER_FMT, self._mm, 0)
            if magic != MAGIC:
                raise RuntimeError(f"frame ring {name} has bad magic")

        self._next_seq = 1

    def _write_latest(self, seq: int) -> None:
        struct.pack_into("<Q", self._mm, _LATEST_OFFSET, seq)

    def _read_latest(self) -> int:
        return struct.unpack_from("<Q", self._mm, _LATEST_OFFSET)[0]

    def _slot_offset(self, slot: int) -> int:
        return HEADER_BYTES + slot * self.slot_bytes

    def write(self, pixels) -> int:
        """Publish one frame.

        :param pixels: ``bytes``/``memoryview`` of exactly ``slot_bytes``.
        :returns: the 1-based sequence number to advertise as the RouterFrame
            ``sideband_seq`` so the consumer fetches the matching slot.
        """
        if len(pixels) != self.slot_bytes:
            raise ValueError(f"expected {self.slot_bytes} bytes, got {len(pixels)}")
        seq = self._next_seq
        self._next_seq += 1
        slot = (seq - 1) % self.slot_count
        off = self._slot_offset(slot)
        self._mm[off:off + self.slot_bytes] = pixels
        self._write_latest(seq)
        return seq

    def read(self, want: int = 0):
        """Copy a frame out tear-free.

        :param want: sequence advertised by the RouterFrame, or 0 for "latest".
        :returns: ``bytes`` of ``slot_bytes`` length, or ``None`` if no coherent
            frame is available yet.
        """
        for _ in range(8):
            s0 = self._read_latest()
            if s0 == 0:
                return None
            seq = want if (want and want <= s0) else s0
            slot = (seq - 1) % self.slot_count
            off = self._slot_offset(slot)
            data = bytes(self._mm[off:off + self.slot_bytes])
            s1 = self._read_latest()
            if s1 - seq < self.slot_count:
                return data
        return None

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            if self._owns:
                try:
                    os.unlink(_dev_shm_path(self.name))
                except FileNotFoundError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
