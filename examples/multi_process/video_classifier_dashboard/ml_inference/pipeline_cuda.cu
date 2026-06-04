/**
 * @file pipeline_cuda.cu
 * @brief CUDA backend: GPU edge-energy saliency by default, YOLOv8 TensorRT when
 *        a model is configured and VCD_WITH_TENSORRT is enabled.
 *
 * The detector runs on the GPU; the (cheap) box overlay is shared host code
 * (@ref overlay.hpp). The energy grid kernel is a genuine CUDA reduction so the
 * default, model-free build still demonstrates GPU compute in the hot path.
 */

#include "overlay.hpp"
#include "pipeline.hpp"
#include "saliency.hpp"

#ifdef VCD_WITH_TENSORRT
#include "yolo_trt.hpp"
#endif

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

namespace rim_vcd {
namespace {

using Clock = std::chrono::steady_clock;
inline float ms_since(Clock::time_point t) {
    return std::chrono::duration<float, std::milli>(Clock::now() - t).count();
}

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("cuda: ") + what + ": " +
                                 cudaGetErrorString(e));
}

/// One block per grid cell; reduce |dLum/dx|+|dLum/dy| over the cell.
__global__ void energy_kernel(const uint8_t* rgb, int W, int H, float* energy) {
    const int gx = blockIdx.x, gy = blockIdx.y;
    const int cw = W / kGridX, ch = H / kGridY;
    const int x0 = gx * cw, y0 = gy * ch;

    __shared__ float partial[256];
    float acc = 0; int n = 0;
    for (int y = y0 + threadIdx.y; y < y0 + ch && y < H - 1; y += blockDim.y) {
        for (int x = x0 + threadIdx.x; x < x0 + cw && x < W - 1; x += blockDim.x) {
            const uint8_t* p = rgb + (static_cast<size_t>(y) * W + x) * 3;
            const int lum = p[0] + 2 * p[1] + p[2];
            const uint8_t* px = p + 3;
            const uint8_t* py = p + static_cast<size_t>(W) * 3;
            acc += fabsf(float(lum - (px[0] + 2 * px[1] + px[2]))) +
                   fabsf(float(lum - (py[0] + 2 * py[1] + py[2])));
            ++n;
        }
    }
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    partial[tid] = (n ? acc / n : 0.0f);
    __syncthreads();
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        energy[gy * kGridX + gx] = partial[0] / (blockDim.x * blockDim.y) / 1024.0f;
}

class CudaPipeline final : public Pipeline {
public:
    CudaPipeline(int w, int h, const std::string& model_path, float conf)
        : w_(w), h_(h), conf_(conf) {
        cuda_check(cudaMalloc(&d_rgb_, static_cast<size_t>(w) * h * 3), "malloc rgb");
        cuda_check(cudaMalloc(&d_energy_, sizeof(float) * kGridX * kGridY),
                   "malloc energy");
#ifdef VCD_WITH_TENSORRT
        if (!model_path.empty()) {
            yolo_ = std::make_unique<YoloTrt>(model_path, 640, conf);
            tag_ = "cuda-yolov8-trt";
        }
#else
        if (!model_path.empty())
            std::fprintf(stderr, "[ml] model set but built without TensorRT; "
                                 "using CUDA saliency\n");
#endif
    }

    ~CudaPipeline() override {
        if (d_rgb_) cudaFree(d_rgb_);
        if (d_energy_) cudaFree(d_energy_);
    }

    PassResult process(const uint8_t* src, uint8_t* dst,
                       std::vector<Detection>& dets) override {
        PassResult res;
        const auto t0 = Clock::now();
#ifdef VCD_WITH_TENSORRT
        if (yolo_) {
            dets = yolo_->infer(src, w_, h_);
        } else
#endif
        {
            cuda_check(cudaMemcpy(d_rgb_, src, static_cast<size_t>(w_) * h_ * 3,
                                  cudaMemcpyHostToDevice), "h2d");
            dim3 grid(kGridX, kGridY), block(16, 16);
            energy_kernel<<<grid, block>>>(d_rgb_, w_, h_, d_energy_);
            cuda_check(cudaGetLastError(), "energy launch");
            cuda_check(cudaMemcpy(energy_, d_energy_, sizeof(energy_),
                                  cudaMemcpyDeviceToHost), "d2h");
            pick_boxes(energy_, w_, h_, conf_, dets);
        }
        res.infer_ms = ms_since(t0);

        const auto t1 = Clock::now();
        std::memcpy(dst, src, static_cast<size_t>(w_) * h_ * 3);
        draw_boxes(dst, w_, h_, dets);
        res.overlay_ms = ms_since(t1);
        res.num_boxes = static_cast<int>(dets.size());
        return res;
    }

    const char* backend() const override { return tag_; }

private:
    int w_, h_;
    float conf_;
    uint8_t* d_rgb_ = nullptr;
    float* d_energy_ = nullptr;
    float energy_[kGridX * kGridY] = {};
    const char* tag_ = "cuda-saliency";
#ifdef VCD_WITH_TENSORRT
    std::unique_ptr<YoloTrt> yolo_;
#endif
};

}  // namespace

std::unique_ptr<Pipeline> make_pipeline(int width, int height,
                                        const std::string& model_path, float conf) {
    return std::make_unique<CudaPipeline>(width, height, model_path, conf);
}

}  // namespace rim_vcd
