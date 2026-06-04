/**
 * @file pipeline_cpu.cpp
 * @brief Pure-host fallback backend (no CUDA) — lets the demo run anywhere.
 *
 * Compiled only when CUDA is unavailable (or `-DVCD_FORCE_CPU=ON`). It runs the
 * same edge-energy saliency detector as the CUDA fallback, just on the CPU.
 */

#include "overlay.hpp"
#include "pipeline.hpp"
#include "saliency.hpp"

#include <chrono>
#include <cstring>

namespace rim_vcd {
namespace {

using Clock = std::chrono::steady_clock;
inline float ms_since(Clock::time_point t) {
    return std::chrono::duration<float, std::milli>(Clock::now() - t).count();
}

class CpuPipeline final : public Pipeline {
public:
    CpuPipeline(int w, int h, float conf) : w_(w), h_(h), conf_(conf) {}

    PassResult process(const uint8_t* src, uint8_t* dst,
                       std::vector<Detection>& dets) override {
        PassResult res;
        const auto t0 = Clock::now();
        compute_energy_host(src, w_, h_, energy_);
        pick_boxes(energy_, w_, h_, conf_, dets);
        res.infer_ms = ms_since(t0);

        const auto t1 = Clock::now();
        std::memcpy(dst, src, static_cast<size_t>(w_) * h_ * 3);
        draw_boxes(dst, w_, h_, dets);
        res.overlay_ms = ms_since(t1);
        res.num_boxes = static_cast<int>(dets.size());
        return res;
    }

    const char* backend() const override { return "cpu-saliency"; }

private:
    int w_, h_;
    float conf_;
    float energy_[kGridX * kGridY] = {};
};

}  // namespace

std::unique_ptr<Pipeline> make_pipeline(int width, int height,
                                        const std::string& /*model_path*/,
                                        float conf) {
    return std::make_unique<CpuPipeline>(width, height, conf);
}

}  // namespace rim_vcd
