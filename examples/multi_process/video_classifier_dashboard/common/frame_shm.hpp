#pragma once

/**
 * @file frame_shm.hpp
 * @brief Cross-language shared-memory frame ring used as a RIM *sideband region*.
 *
 * RoboticsIpcModule keeps bulk data (camera frames, tensors) out of the 64 B
 * RouterFrame and in a named shared-memory region the topology declares as a
 * `[[peers.sideband]]` (see ADR 0005 / sideband.hpp). The header-only core only
 * carries the *descriptor*; the producer and consumer agree on the in-region
 * layout themselves. This file is that agreement, ported byte-for-byte to
 * Python (`frame_shm.py`) and Node (`frame_shm.js`).
 *
 * Layout (all little-endian, x86_64 / aarch64):
 * @code
 *   offset  size  field
 *   ------  ----  ---------------------------------------------------------
 *      0     4    magic        'R','V','F','1'
 *      4     4    version      = kFrameShmVersion
 *      8     4    width        pixels
 *     12     4    height       pixels
 *     16     4    channels     3 (RGB24)
 *     20     4    slot_count   ring depth (>= 3 for a tear-free reader)
 *     24     4    slot_bytes   width * height * channels
 *     28     4    reserved
 *     32     8    latest_seq   1-based; 0 = no frame yet (the seqlock)
 *     40    24    reserved (pad header to 64 B)
 *     64   slot_count * slot_bytes   the frame slots
 * @endcode
 *
 * Concurrency: single writer, many readers, "newest frame wins" — the same
 * lock-free philosophy as RIM's SHM transport. The writer fills slot
 * `(seq-1) % slot_count` then publishes `latest_seq = seq`. A reader snapshots
 * `latest_seq`, copies the slot, re-reads `latest_seq`, and accepts the copy iff
 * the writer has not lapped the ring meanwhile (`s1 - s0 < slot_count`).
 */

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rim_vcd {

constexpr uint32_t kFrameShmMagic   = 0x31465652u;  ///< 'R','V','F','1' little-endian
constexpr uint32_t kFrameShmVersion = 1u;
constexpr uint32_t kFrameShmHeaderBytes = 64u;

/** @brief POD header mapped at offset 0 of every frame-ring region. */
struct FrameShmHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t slot_count;
    uint32_t slot_bytes;
    uint32_t reserved0;
    uint64_t latest_seq;   ///< accessed as std::atomic via std::atomic_ref-style fences
    uint64_t reserved1[3];
};
static_assert(sizeof(FrameShmHeader) == kFrameShmHeaderBytes,
              "FrameShmHeader must be 64 B to match the Python/Node ports");

/**
 * @brief RAII map of a frame ring, either creating it (producer) or opening an
 *        existing one (consumer).
 *
 * @note `name` is a POSIX shm name ("/rim_vcd_source_frames"); it appears on
 *       disk as `/dev/shm/rim_vcd_source_frames`, which is what the Python and
 *       Node ports open directly.
 */
class FrameShm {
public:
    /** @brief Open or create the region. @param create true on the producer. */
    static FrameShm open(const std::string& name, uint32_t width, uint32_t height,
                         uint32_t channels, uint32_t slot_count, bool create) {
        const uint32_t slot_bytes = width * height * channels;
        const size_t total = kFrameShmHeaderBytes +
                             static_cast<size_t>(slot_bytes) * slot_count;

        const int flags = create ? (O_CREAT | O_RDWR) : O_RDWR;
        const int fd = ::shm_open(name.c_str(), flags, 0666);
        if (fd < 0) {
            throw std::runtime_error("shm_open(" + name + ") failed: " +
                                     std::strerror(errno));
        }
        if (create && ::ftruncate(fd, static_cast<off_t>(total)) != 0) {
            ::close(fd);
            throw std::runtime_error("ftruncate(" + name + ") failed");
        }
        void* base = ::mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED) {
            throw std::runtime_error("mmap(" + name + ") failed");
        }

        FrameShm ring;
        ring.name_ = name;
        ring.base_ = static_cast<uint8_t*>(base);
        ring.total_ = total;
        ring.owns_ = create;
        ring.hdr_ = reinterpret_cast<FrameShmHeader*>(ring.base_);

        if (create) {
            std::memset(ring.base_, 0, kFrameShmHeaderBytes);
            ring.hdr_->magic = kFrameShmMagic;
            ring.hdr_->version = kFrameShmVersion;
            ring.hdr_->width = width;
            ring.hdr_->height = height;
            ring.hdr_->channels = channels;
            ring.hdr_->slot_count = slot_count;
            ring.hdr_->slot_bytes = slot_bytes;
            std::atomic_thread_fence(std::memory_order_release);
        } else if (ring.hdr_->magic != kFrameShmMagic) {
            ::munmap(base, total);
            throw std::runtime_error("frame ring " + name + " has bad magic");
        }
        return ring;
    }

    FrameShm() = default;
    FrameShm(const FrameShm&) = delete;
    FrameShm& operator=(const FrameShm&) = delete;
    FrameShm(FrameShm&& o) noexcept { *this = std::move(o); }
    FrameShm& operator=(FrameShm&& o) noexcept {
        if (this != &o) {
            unmap();
            name_ = std::move(o.name_); base_ = o.base_; total_ = o.total_;
            owns_ = o.owns_; hdr_ = o.hdr_;
            o.base_ = nullptr; o.hdr_ = nullptr; o.total_ = 0; o.owns_ = false;
        }
        return *this;
    }
    ~FrameShm() { unmap(); }

    uint32_t width() const { return hdr_->width; }
    uint32_t height() const { return hdr_->height; }
    uint32_t channels() const { return hdr_->channels; }
    uint32_t slot_count() const { return hdr_->slot_count; }
    uint32_t slot_bytes() const { return hdr_->slot_bytes; }

    /**
     * @brief Producer: publish one frame.
     * @param pixels  slot_bytes() of tightly packed RGB24.
     * @return the published 1-based sequence number (use as RouterFrame
     *         sideband_seq so the consumer fetches the matching slot).
     */
    uint64_t write(const uint8_t* pixels) {
        const uint64_t seq = next_seq_++;
        const uint32_t slot = static_cast<uint32_t>((seq - 1) % hdr_->slot_count);
        std::memcpy(slot_ptr(slot), pixels, hdr_->slot_bytes);
        std::atomic_thread_fence(std::memory_order_release);
        store_latest(seq);
        return seq;
    }

    /**
     * @brief Consumer: copy a frame out tear-free.
     * @param want  the sequence advertised by the RouterFrame, or 0 for "latest".
     * @param out   destination buffer of at least slot_bytes().
     * @return true if a coherent frame was copied.
     */
    bool read(uint64_t want, uint8_t* out) const {
        for (int attempt = 0; attempt < 8; ++attempt) {
            const uint64_t s0 = load_latest();
            if (s0 == 0) return false;
            const uint64_t seq = (want != 0 && want <= s0) ? want : s0;
            const uint32_t slot = static_cast<uint32_t>((seq - 1) % hdr_->slot_count);
            std::atomic_thread_fence(std::memory_order_acquire);
            std::memcpy(out, slot_ptr(slot), hdr_->slot_bytes);
            std::atomic_thread_fence(std::memory_order_acquire);
            const uint64_t s1 = load_latest();
            if (s1 - seq < hdr_->slot_count) return true;  // slot not lapped
        }
        return false;
    }

private:
    uint8_t* slot_ptr(uint32_t slot) const {
        return base_ + kFrameShmHeaderBytes +
               static_cast<size_t>(slot) * hdr_->slot_bytes;
    }
    uint64_t load_latest() const {
        return reinterpret_cast<const std::atomic<uint64_t>*>(&hdr_->latest_seq)
            ->load(std::memory_order_relaxed);
    }
    void store_latest(uint64_t v) {
        reinterpret_cast<std::atomic<uint64_t>*>(&hdr_->latest_seq)
            ->store(v, std::memory_order_relaxed);
    }
    void unmap() {
        if (base_) {
            ::munmap(base_, total_);
            if (owns_) ::shm_unlink(name_.c_str());
            base_ = nullptr;
        }
    }

    std::string name_;
    uint8_t* base_ = nullptr;
    size_t total_ = 0;
    bool owns_ = false;
    FrameShmHeader* hdr_ = nullptr;
    uint64_t next_seq_ = 1;
};

}  // namespace rim_vcd
