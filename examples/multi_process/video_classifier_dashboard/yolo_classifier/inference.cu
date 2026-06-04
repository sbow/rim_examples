/**
 * CUDA preprocessing kernels for YOLO inference.
 *
 * Performs GPU-accelerated image preprocessing including:
 * - Letterbox resize to model input dimensions
 * - BGR to RGB conversion
 * - Normalization to [0, 1] range
 * - HWC to CHW format conversion
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

namespace {

/**
 * Bilinear interpolation resize + normalize kernel.
 * Converts HWC uint8 input to CHW float32 output normalized to [0, 1].
 */
__global__ void preprocess_kernel(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int src_width, int src_height, int src_channels,
    int dst_width, int dst_height,
    float scale_x, float scale_y,
    int pad_x, int pad_y) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) return;

    // Check if this pixel is in the padded (letterbox) region
    int src_x_base = x - pad_x;
    int src_y_base = y - pad_y;

    float r = 0.5f, g = 0.5f, b = 0.5f;  // Gray padding

    int effective_w = dst_width - 2 * pad_x;
    int effective_h = dst_height - 2 * pad_y;

    if (src_x_base >= 0 && src_x_base < effective_w &&
        src_y_base >= 0 && src_y_base < effective_h) {

        // Map to source coordinates with bilinear interpolation
        float fx = static_cast<float>(src_x_base) * scale_x;
        float fy = static_cast<float>(src_y_base) * scale_y;

        int x0 = static_cast<int>(fx);
        int y0 = static_cast<int>(fy);
        int x1 = min(x0 + 1, src_width - 1);
        int y1 = min(y0 + 1, src_height - 1);

        float wx = fx - x0;
        float wy = fy - y0;

        // Bilinear interpolation for each channel (BGR input)
        for (int c = 0; c < 3; ++c) {
            float v00 = src[(y0 * src_width + x0) * src_channels + c];
            float v01 = src[(y0 * src_width + x1) * src_channels + c];
            float v10 = src[(y1 * src_width + x0) * src_channels + c];
            float v11 = src[(y1 * src_width + x1) * src_channels + c];

            float val = v00 * (1 - wx) * (1 - wy) +
                       v01 * wx * (1 - wy) +
                       v10 * (1 - wx) * wy +
                       v11 * wx * wy;

            // BGR to RGB conversion: swap channel 0 and 2
            int out_c = (c == 0) ? 2 : (c == 2) ? 0 : 1;

            if (out_c == 0) r = val / 255.0f;
            else if (out_c == 1) g = val / 255.0f;
            else b = val / 255.0f;
        }
    }

    // Write in CHW format (plane-by-plane)
    int pixel_idx = y * dst_width + x;
    int plane_size = dst_width * dst_height;
    dst[0 * plane_size + pixel_idx] = r;
    dst[1 * plane_size + pixel_idx] = g;
    dst[2 * plane_size + pixel_idx] = b;
}

/**
 * Simple NMS kernel for post-processing acceleration.
 * Computes IoU between detection pairs and marks suppressed detections.
 */
__global__ void nms_kernel(
    const float* __restrict__ boxes,  // [N, 4] (x1, y1, x2, y2)
    const float* __restrict__ scores,
    int* __restrict__ keep,
    int num_boxes,
    float nms_threshold) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes || keep[i] == 0) return;

    float x1_i = boxes[i * 4 + 0];
    float y1_i = boxes[i * 4 + 1];
    float x2_i = boxes[i * 4 + 2];
    float y2_i = boxes[i * 4 + 3];
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);

    for (int j = i + 1; j < num_boxes; ++j) {
        if (keep[j] == 0) continue;
        if (scores[j] > scores[i]) continue;  // Only suppress lower-scored

        float x1_j = boxes[j * 4 + 0];
        float y1_j = boxes[j * 4 + 1];
        float x2_j = boxes[j * 4 + 2];
        float y2_j = boxes[j * 4 + 3];
        float area_j = (x2_j - x1_j) * (y2_j - y1_j);

        float inter_x1 = fmaxf(x1_i, x1_j);
        float inter_y1 = fmaxf(y1_i, y1_j);
        float inter_x2 = fminf(x2_i, x2_j);
        float inter_y2 = fminf(y2_i, y2_j);

        float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
        float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
        float inter_area = inter_w * inter_h;

        float iou = inter_area / (area_i + area_j - inter_area + 1e-6f);
        if (iou > nms_threshold) {
            keep[j] = 0;
        }
    }
}

}  // namespace

/**
 * Host-callable function for CUDA preprocessing.
 */
extern "C" void cuda_preprocess(
    const unsigned char* src, float* dst,
    int src_width, int src_height, int src_channels,
    int dst_width, int dst_height) {

    // Calculate letterbox parameters
    float scale = fminf(
        static_cast<float>(dst_width) / src_width,
        static_cast<float>(dst_height) / src_height);

    int effective_w = static_cast<int>(src_width * scale);
    int effective_h = static_cast<int>(src_height * scale);
    int pad_x = (dst_width - effective_w) / 2;
    int pad_y = (dst_height - effective_h) / 2;

    float scale_x = static_cast<float>(src_width) / effective_w;
    float scale_y = static_cast<float>(src_height) / effective_h;

    // Allocate device memory
    unsigned char* d_src;
    float* d_dst;
    size_t src_size = src_width * src_height * src_channels;
    size_t dst_size = dst_width * dst_height * 3 * sizeof(float);

    cudaMalloc(&d_src, src_size);
    cudaMalloc(&d_dst, dst_size);

    cudaMemcpy(d_src, src, src_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid(
        (dst_width + block.x - 1) / block.x,
        (dst_height + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block>>>(
        d_src, d_dst,
        src_width, src_height, src_channels,
        dst_width, dst_height,
        scale_x, scale_y,
        pad_x, pad_y);

    cudaMemcpy(dst, d_dst, dst_size, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}
