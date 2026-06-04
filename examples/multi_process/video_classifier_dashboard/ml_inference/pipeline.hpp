#pragma once

/**
 * @file pipeline.hpp
 * @brief Backend-agnostic detect+overlay interface for the ML peer.
 *
 * Exactly one implementation is linked at build time:
 *   * @ref pipeline_cuda.cu — CUDA edge-energy detector, plus an optional
 *     YOLOv8 TensorRT backend (`-DVCD_WITH_TENSORRT=ON` + a model in the TOML).
 *   * @ref pipeline_cpu.cpp — pure-host fallback so the demo builds/runs with
 *     no GPU at all.
 *
 * Either way the contract is identical, so @ref ml_main.cpp never sees CUDA.
 */

#include "detector.hpp"

#include <memory>
#include <string>
#include <vector>

namespace rim_vcd {

/** @brief Owns whatever device/model state a backend needs across frames. */
class Pipeline {
public:
    virtual ~Pipeline() = default;

    /**
     * @brief Detect objects in @p src_rgb and write an annotated copy to @p dst_rgb.
     * @param src_rgb tightly packed RGB24 of the configured WxH.
     * @param dst_rgb destination of the same size (boxes overlaid).
     * @param dets   filled with this frame's detections (for the metadata sideband).
     * @return timing + count summary.
     */
    virtual PassResult process(const uint8_t* src_rgb, uint8_t* dst_rgb,
                               std::vector<Detection>& dets) = 0;

    /** @brief Human-readable backend tag, e.g. "cuda-yolov8-trt". */
    virtual const char* backend() const = 0;
};

/**
 * @brief Construct the compiled-in backend.
 * @param model_path YOLO engine/ONNX path; empty → built-in saliency detector.
 */
std::unique_ptr<Pipeline> make_pipeline(int width, int height,
                                        const std::string& model_path, float conf);

}  // namespace rim_vcd
