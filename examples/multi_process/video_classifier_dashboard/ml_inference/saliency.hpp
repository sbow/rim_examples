#pragma once

/**
 * @file saliency.hpp
 * @brief Model-free fallback "detector" so the pipeline runs with no weights.
 *
 * When no YOLO engine is configured, the ML peer still needs to emit plausible
 * boxes to exercise the whole RIM pipeline end-to-end. This computes a coarse
 * edge-energy grid over the frame and turns the most energetic cells into
 * boxes. It is intentionally simple and clearly *not* a classifier — class ids
 * are assigned round-robin purely so the colour/label plumbing has something to
 * show. The energy grid itself is computed on the GPU in the CUDA backend
 * (see @ref pipeline_cuda.cu) and on the CPU in the fallback backend.
 */

#include "detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rim_vcd {

constexpr int kGridX = 20;  ///< energy grid columns
constexpr int kGridY = 12;  ///< energy grid rows

/** @brief CPU reference for the per-cell edge energy (CUDA mirrors this). */
inline void compute_energy_host(const uint8_t* rgb, int W, int H, float* energy) {
    const int cw = W / kGridX, ch = H / kGridY;
    for (int gy = 0; gy < kGridY; ++gy) {
        for (int gx = 0; gx < kGridX; ++gx) {
            float acc = 0;
            int n = 0;
            for (int y = gy * ch; y < (gy + 1) * ch && y < H - 1; y += 2) {
                for (int x = gx * cw; x < (gx + 1) * cw && x < W - 1; x += 2) {
                    const uint8_t* p = rgb + (static_cast<size_t>(y) * W + x) * 3;
                    const int lum = p[0] + 2 * p[1] + p[2];
                    const uint8_t* px = p + 3;
                    const uint8_t* py = p + static_cast<size_t>(W) * 3;
                    const int lx = px[0] + 2 * px[1] + px[2];
                    const int ly = py[0] + 2 * py[1] + py[2];
                    acc += std::abs(lum - lx) + std::abs(lum - ly);
                    ++n;
                }
            }
            energy[gy * kGridX + gx] = n ? acc / (n * 1024.0f) : 0.0f;
        }
    }
}

/**
 * @brief Turn the energy grid into up to `max_boxes` detections.
 * @param conf threshold in 0..1 relative to the peak cell energy.
 */
inline void pick_boxes(const float* energy, int W, int H, float conf,
                       std::vector<Detection>& out, int max_boxes = 12) {
    out.clear();
    float peak = 1e-6f;
    for (int i = 0; i < kGridX * kGridY; ++i) peak = std::max(peak, energy[i]);
    const float thresh = conf * peak;

    const int cw = W / kGridX, ch = H / kGridY;
    int cls_cycle = 0;
    for (int gy = 0; gy < kGridY && (int)out.size() < max_boxes; ++gy) {
        for (int gx = 0; gx < kGridX && (int)out.size() < max_boxes; ++gx) {
            const float e = energy[gy * kGridX + gx];
            if (e < thresh) continue;
            // Skip a cell whose neighbour to the left/up is already a box to
            // crudely merge runs into fewer, larger boxes.
            const bool left = gx > 0 && energy[gy * kGridX + gx - 1] >= thresh;
            const bool up = gy > 0 && energy[(gy - 1) * kGridX + gx] >= thresh;
            if (left || up) continue;
            int gx2 = gx;
            while (gx2 + 1 < kGridX && energy[gy * kGridX + gx2 + 1] >= thresh) ++gx2;
            int gy2 = gy;
            while (gy2 + 1 < kGridY && energy[(gy2 + 1) * kGridX + gx] >= thresh) ++gy2;
            Detection d;
            d.x = static_cast<float>(gx * cw);
            d.y = static_cast<float>(gy * ch);
            d.w = static_cast<float>((gx2 - gx + 1) * cw);
            d.h = static_cast<float>((gy2 - gy + 1) * ch);
            d.score = std::min(1.0f, e / peak);
            d.cls = (cls_cycle++ * 7) % 80;  // spread across COCO classes
            out.push_back(d);
        }
    }
}

}  // namespace rim_vcd
