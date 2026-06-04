#pragma once

/**
 * @file wire.hpp
 * @brief Demo payload layouts carried in the 32 B RouterFrame inline payload
 *        and in the detection sideband region.
 *
 * These mirror the packing in `video_source/publisher.py` (raw meta) and
 * `dashboard/server.js` (annotated stats + detections). All little-endian.
 */

#include <cstdint>
#include <cstring>

namespace rim_vcd {

constexpr uint16_t kTopicRawFrame = 10;
constexpr uint16_t kTopicAnnotatedFrame = 20;

constexpr int kMaxDetections = 64;
constexpr int kDetRecordBytes = 24;  ///< 5 floats + 1 uint32

/** @brief Inline payload published by video_source (topic 10). Matches "<IIIQf". */
#pragma pack(push, 1)
struct RawMeta {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint64_t capture_ns;
    float source_fps;
};

/** @brief Inline payload published by ml_inference (topic 20). Matches "<HHHHfffI". */
struct AnnStats {
    uint16_t width;
    uint16_t height;
    uint16_t num_boxes;
    uint16_t pad;
    float infer_ms;
    float overlay_ms;
    float e2e_ms;
    uint32_t source_seq;
};

/** @brief One record in the detection sideband region. */
struct DetRecord {
    float x, y, w, h, score;
    uint32_t cls;
};
#pragma pack(pop)

static_assert(sizeof(RawMeta) == 24, "RawMeta layout");
static_assert(sizeof(AnnStats) == 24, "AnnStats layout");
static_assert(sizeof(DetRecord) == kDetRecordBytes, "DetRecord layout");

}  // namespace rim_vcd
