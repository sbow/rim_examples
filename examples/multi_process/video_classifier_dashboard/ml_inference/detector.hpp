#pragma once

/**
 * @file detector.hpp
 * @brief Detection types + COCO label/colour tables shared by every backend.
 */

#include <cstdint>
#include <vector>

namespace rim_vcd {

/** @brief One detected object in pixel coordinates of the source frame. */
struct Detection {
    float x = 0;      ///< top-left x (pixels)
    float y = 0;      ///< top-left y (pixels)
    float w = 0;      ///< width (pixels)
    float h = 0;      ///< height (pixels)
    float score = 0;  ///< confidence 0..1
    int cls = 0;      ///< COCO class id (0..79)
};

/** @brief Timing + count summary returned by a pipeline pass. */
struct PassResult {
    int num_boxes = 0;
    float infer_ms = 0;    ///< detector wall time
    float overlay_ms = 0;  ///< box/label draw wall time
};

/// 80 COCO class names (YOLOv8 default training set).
inline const char* coco_class_name(int cls) {
    static const char* kNames[80] = {
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
        "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
        "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
        "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
        "potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush",
    };
    return (cls >= 0 && cls < 80) ? kNames[cls] : "object";
}

/// Deterministic, well-separated RGB colour for a class id (golden-ratio hue).
inline void coco_class_color(int cls, uint8_t& r, uint8_t& g, uint8_t& b) {
    const float h = static_cast<float>((cls * 0.61803398875f) - static_cast<int>(cls * 0.61803398875f)) * 6.0f;
    const int i = static_cast<int>(h);
    const float f = h - i;
    const float q = 1.0f - f;
    auto to8 = [](float v) { return static_cast<uint8_t>(v * 230.0f + 25.0f); };
    switch (i % 6) {
        case 0: r = to8(1); g = to8(f); b = to8(0); break;
        case 1: r = to8(q); g = to8(1); b = to8(0); break;
        case 2: r = to8(0); g = to8(1); b = to8(f); break;
        case 3: r = to8(0); g = to8(q); b = to8(1); break;
        case 4: r = to8(f); g = to8(0); b = to8(1); break;
        default: r = to8(1); g = to8(0); b = to8(q); break;
    }
}

}  // namespace rim_vcd
