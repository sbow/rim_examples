/**
 * @file yolo_trt.cu
 * @brief YOLOv8 TensorRT backend implementation (built only with VCD_WITH_TENSORRT).
 *
 * Reference integration: letterbox+normalize on the GPU, a single TensorRT
 * enqueue, then a host-side decode + greedy NMS of the YOLOv8 `[1,84,8400]`
 * output (84 = 4 box + 80 class scores; v8 has no objectness channel).
 */

#include "yolo_trt.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <NvInfer.h>

namespace rim_vcd {
namespace {

class TrtLogger : public nvinfer1::ILogger {
    void log(Severity sev, const char* msg) noexcept override {
        if (sev <= Severity::kWARNING) fprintf(stderr, "[trt] %s\n", msg);
    }
};
TrtLogger g_logger;

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("cuda: ") + what + ": " +
                                 cudaGetErrorString(e));
}

/// HWC-RGB(u8) -> CHW-RGB(float, 0..1) into a letterboxed square net input.
__global__ void preprocess_kernel(const uint8_t* src, int W, int H,
                                  float* dst, int net, float scale,
                                  int pad_x, int pad_y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= net || y >= net) return;
    const int sx = static_cast<int>((x - pad_x) / scale);
    const int sy = static_cast<int>((y - pad_y) / scale);
    float r = 0.5f, g = 0.5f, b = 0.5f;  // grey letterbox
    if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
        const uint8_t* p = src + (static_cast<size_t>(sy) * W + sx) * 3;
        r = p[0] / 255.0f; g = p[1] / 255.0f; b = p[2] / 255.0f;
    }
    const int plane = net * net;
    dst[0 * plane + y * net + x] = r;
    dst[1 * plane + y * net + x] = g;
    dst[2 * plane + y * net + x] = b;
}

float iou(const Detection& a, const Detection& b) {
    const float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.w, b.x + b.w);
    const float y2 = std::min(a.y + a.h, b.y + b.h);
    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

}  // namespace

struct YoloTrt::Impl {
    int net;
    float conf;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* ctx = nullptr;
    void* d_in = nullptr;     // [3*net*net] float
    void* d_out = nullptr;    // [84*8400] float
    uint8_t* d_rgb = nullptr; // [W*H*3]
    int out_attrs = 84, out_anchors = 8400;
    std::vector<float> h_out;
    cudaStream_t stream{};
    int alloc_w = 0, alloc_h = 0;

    void ensure_rgb(int W, int H) {
        if (W * H <= alloc_w * alloc_h) return;
        if (d_rgb) cudaFree(d_rgb);
        cuda_check(cudaMalloc(&d_rgb, static_cast<size_t>(W) * H * 3), "malloc rgb");
        alloc_w = W; alloc_h = H;
    }
};

YoloTrt::YoloTrt(const std::string& engine_path, int net_size, float conf)
    : impl_(std::make_unique<Impl>()) {
    impl_->net = net_size;
    impl_->conf = conf;

    std::ifstream f(engine_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open engine " + engine_path);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});

    impl_->runtime = nvinfer1::createInferRuntime(g_logger);
    impl_->engine = impl_->runtime->deserializeCudaEngine(blob.data(), blob.size());
    if (!impl_->engine) throw std::runtime_error("deserializeCudaEngine failed");
    impl_->ctx = impl_->engine->createExecutionContext();
    cuda_check(cudaStreamCreate(&impl_->stream), "stream");

    const int n = net_size;
    cuda_check(cudaMalloc(&impl_->d_in, sizeof(float) * 3 * n * n), "malloc in");
    cuda_check(cudaMalloc(&impl_->d_out,
                          sizeof(float) * impl_->out_attrs * impl_->out_anchors),
               "malloc out");
    impl_->h_out.resize(impl_->out_attrs * impl_->out_anchors);
}

YoloTrt::~YoloTrt() {
    if (!impl_) return;
    if (impl_->d_in) cudaFree(impl_->d_in);
    if (impl_->d_out) cudaFree(impl_->d_out);
    if (impl_->d_rgb) cudaFree(impl_->d_rgb);
    if (impl_->stream) cudaStreamDestroy(impl_->stream);
    if (impl_->ctx) delete impl_->ctx;
    if (impl_->engine) delete impl_->engine;
    if (impl_->runtime) delete impl_->runtime;
}

std::vector<Detection> YoloTrt::infer(const uint8_t* rgb, int W, int H) {
    Impl& s = *impl_;
    const int net = s.net;
    const float scale = std::min(net / float(W), net / float(H));
    const int pad_x = static_cast<int>((net - W * scale) / 2);
    const int pad_y = static_cast<int>((net - H * scale) / 2);

    s.ensure_rgb(W, H);
    cuda_check(cudaMemcpyAsync(s.d_rgb, rgb, static_cast<size_t>(W) * H * 3,
                              cudaMemcpyHostToDevice, s.stream), "h2d");

    dim3 block(16, 16), grid((net + 15) / 16, (net + 15) / 16);
    preprocess_kernel<<<grid, block, 0, s.stream>>>(
        s.d_rgb, W, H, static_cast<float*>(s.d_in), net, scale, pad_x, pad_y);

    void* bindings[] = {s.d_in, s.d_out};
    s.ctx->enqueueV2(bindings, s.stream, nullptr);
    cuda_check(cudaMemcpyAsync(s.h_out.data(), s.d_out,
                              sizeof(float) * s.out_attrs * s.out_anchors,
                              cudaMemcpyDeviceToHost, s.stream), "d2h");
    cuda_check(cudaStreamSynchronize(s.stream), "sync");

    // Decode YOLOv8 [84, 8400], attribute-major: row a, anchor i -> [a*8400+i].
    const int A = s.out_anchors;
    std::vector<Detection> cand;
    for (int i = 0; i < A; ++i) {
        int best = -1; float bestp = s.conf;
        for (int c = 0; c < 80; ++c) {
            const float p = s.h_out[(4 + c) * A + i];
            if (p > bestp) { bestp = p; best = c; }
        }
        if (best < 0) continue;
        const float cx = s.h_out[0 * A + i], cy = s.h_out[1 * A + i];
        const float bw = s.h_out[2 * A + i], bh = s.h_out[3 * A + i];
        Detection d;
        d.x = ((cx - bw / 2) - pad_x) / scale;
        d.y = ((cy - bh / 2) - pad_y) / scale;
        d.w = bw / scale; d.h = bh / scale;
        d.score = bestp; d.cls = best;
        cand.push_back(d);
    }

    std::sort(cand.begin(), cand.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> keep;
    std::vector<char> dead(cand.size(), 0);
    for (size_t i = 0; i < cand.size(); ++i) {
        if (dead[i]) continue;
        keep.push_back(cand[i]);
        for (size_t j = i + 1; j < cand.size(); ++j)
            if (!dead[j] && cand[j].cls == cand[i].cls && iou(cand[i], cand[j]) > 0.45f)
                dead[j] = 1;
        if (keep.size() >= 64) break;
    }
    return keep;
}

}  // namespace rim_vcd
