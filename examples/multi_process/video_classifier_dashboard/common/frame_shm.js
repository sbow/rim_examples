// Cross-language shared-memory frame ring (Node port of `frame_shm.hpp`).
//
// The bulk-data half of RoboticsIpcModule's control-plane/bulk split: the 64 B
// RouterFrame announces "frame N ready"; the RGB pixels live in a named
// /dev/shm region declared as a `[[peers.sideband]]` in the topology (ADR 0005).
// Layout is byte-identical to the C++ and Python ports.
//
// Node has no mmap in stdlib, but /dev/shm files are ordinary tmpfs files, so a
// positioned `fs.readSync` is enough for a consumer (this demo never writes from
// Node). Reads are page-cache hits — no disk I/O.
//
// Layout (little-endian):
//   offset size field
//   0      4    magic 'RVF1'
//   4      4    version
//   8      4    width
//   12     4    height
//   16     4    channels
//   20     4    slot_count
//   24     4    slot_bytes
//   28     4    reserved
//   32     8    latest_seq (1-based; 0 = none)
//   64     slot_count*slot_bytes  frame slots

import fs from 'node:fs';

export const MAGIC = 0x31465652; // 'RVF1'
export const HEADER_BYTES = 64;
const LATEST_OFFSET = 32;

function devShmPath(name) {
    return '/dev/shm/' + name.replace(/^\/+/, '');
}

/** Read-only consumer view of a frame ring backed by /dev/shm. */
export class FrameShmReader {
    /**
     * @param {string} name POSIX shm name, e.g. "/rim_vcd_annotated_frames".
     */
    constructor(name) {
        this.name = name;
        this.path = devShmPath(name);
        this.fd = null;
        this.header = null;
        this._hbuf = Buffer.alloc(HEADER_BYTES);
    }

    /** Open the region and parse its header. Returns true once it exists. */
    open() {
        if (this.fd !== null) return true;
        try {
            this.fd = fs.openSync(this.path, 'r');
        } catch {
            return false;
        }
        fs.readSync(this.fd, this._hbuf, 0, HEADER_BYTES, 0);
        if (this._hbuf.readUInt32LE(0) !== MAGIC) {
            this.close();
            return false;
        }
        this.header = {
            width: this._hbuf.readUInt32LE(8),
            height: this._hbuf.readUInt32LE(12),
            channels: this._hbuf.readUInt32LE(16),
            slotCount: this._hbuf.readUInt32LE(20),
            slotBytes: this._hbuf.readUInt32LE(24),
        };
        this._slot = Buffer.alloc(this.header.slotBytes);
        return true;
    }

    /** Latest published sequence (0 until the producer writes its first frame). */
    latestSeq() {
        fs.readSync(this.fd, this._hbuf, 0, 8, LATEST_OFFSET);
        return this._hbuf.readBigUInt64LE(0);
    }

    /**
     * Copy a frame out tear-free.
     * @param {bigint|number} want sequence from the RouterFrame, or 0 for latest.
     * @returns {Buffer|null} a slotBytes-length RGB buffer, or null if none yet.
     */
    read(want = 0n) {
        if (this.fd === null && !this.open()) return null;
        const wantBig = typeof want === 'bigint' ? want : BigInt(want);
        const sc = BigInt(this.header.slotCount);
        for (let attempt = 0; attempt < 8; attempt += 1) {
            const s0 = this.latestSeq();
            if (s0 === 0n) return null;
            const seq = (wantBig !== 0n && wantBig <= s0) ? wantBig : s0;
            const slot = Number((seq - 1n) % sc);
            const off = HEADER_BYTES + slot * this.header.slotBytes;
            fs.readSync(this.fd, this._slot, 0, this.header.slotBytes, off);
            const s1 = this.latestSeq();
            if (s1 - seq < sc) return this._slot;
        }
        return null;
    }

    close() {
        if (this.fd !== null) {
            try { fs.closeSync(this.fd); } catch { /* already gone */ }
            this.fd = null;
        }
    }
}
