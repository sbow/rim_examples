#pragma once

/**
 * @file yolo_trt.hpp
 * @brief Optional YOLOv8 TensorRT backend (compiled only with VCD_WITH_TENSORRT).
 *
 * Loads a serialized TensorRT engine (a `.engine` built from a YOLOv8 ONNX
 * export) and runs it on the GPU. Pre-processing (letterbox + normalize) and
 * the YOLOv8 output decode + NMS are kept compact; this is reference code, not
 * a tuned production detector. See the README for the export recipe.
 */

#include "detector.hpp"

#include <memory>
#include <string>
#include <vector>

namespace rim_vcd {

/** @brief Thin wrapper over a TensorRT YOLOv8 engine. */
class YoloTrt {
public:
    /**
     * @param engine_path serialized TensorRT engine (.engine).
     * @param net_size    square network input (e.g. 640).
     * @param conf        confidence threshold 0..1.
     */
    YoloTrt(const std::string& engine_path, int net_size, float conf);
    ~YoloTrt();

    /** @brief Run detection on a host RGB24 frame, mapping boxes back to WxH. */
    std::vector<Detection> infer(const uint8_t* rgb, int W, int H);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rim_vcd
