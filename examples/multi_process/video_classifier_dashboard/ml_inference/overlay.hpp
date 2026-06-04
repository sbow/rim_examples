#pragma once

/**
 * @file overlay.hpp
 * @brief Host-side bounding-box overlay shared by the CUDA and CPU backends.
 *
 * The annotated frame is downloaded to host memory before it is written to the
 * SHM ring, so the (cheap, O(boxes)) rectangle draw happens here once, on the
 * CPU, regardless of which detector produced the boxes. Crisp class/score text
 * is rendered by the browser dashboard from the detection sideband — keeping
 * this code font-free.
 */

#include "detector.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rim_vcd {

/** @brief Paint one RGB pixel if in bounds. */
inline void put_px(uint8_t* rgb, int W, int H, int x, int y,
                   uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    uint8_t* p = rgb + (static_cast<size_t>(y) * W + x) * 3;
    p[0] = r; p[1] = g; p[2] = b;
}

/** @brief Draw a `thickness`-pixel rectangle outline in the given colour. */
inline void draw_rect(uint8_t* rgb, int W, int H, int x0, int y0, int x1, int y1,
                      uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
    x0 = std::clamp(x0, 0, W - 1); x1 = std::clamp(x1, 0, W - 1);
    y0 = std::clamp(y0, 0, H - 1); y1 = std::clamp(y1, 0, H - 1);
    for (int t = 0; t < thickness; ++t) {
        for (int x = x0; x <= x1; ++x) {
            put_px(rgb, W, H, x, y0 + t, r, g, b);
            put_px(rgb, W, H, x, y1 - t, r, g, b);
        }
        for (int y = y0; y <= y1; ++y) {
            put_px(rgb, W, H, x0 + t, y, r, g, b);
            put_px(rgb, W, H, x1 - t, y, r, g, b);
        }
    }
}

/** @brief Overlay every detection's box (class colour) onto an RGB frame. */
inline void draw_boxes(uint8_t* rgb, int W, int H,
                       const std::vector<Detection>& dets) {
    for (const Detection& d : dets) {
        uint8_t r, g, b;
        coco_class_color(d.cls, r, g, b);
        const int x0 = static_cast<int>(d.x);
        const int y0 = static_cast<int>(d.y);
        const int x1 = static_cast<int>(d.x + d.w);
        const int y1 = static_cast<int>(d.y + d.h);
        draw_rect(rgb, W, H, x0, y0, x1, y1, r, g, b, 2);
        // A short solid colour tab at the top-left anchors the browser label.
        for (int yy = y0 - 4; yy < y0; ++yy)
            for (int xx = x0; xx < x0 + 18; ++xx)
                put_px(rgb, W, H, xx, yy, r, g, b);
    }
}

}  // namespace rim_vcd
