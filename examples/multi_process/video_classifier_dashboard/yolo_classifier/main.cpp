/**
 * Process 2: YOLO Classifier
 *
 * Reads raw video frames from shared memory, performs YOLO object detection
 * using CUDA-accelerated inference, and publishes annotated frames with
 * bounding box overlays back to shared memory.
 */

#include <iostream>
#include <cstring>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Forward declaration for CUDA preprocessing kernel
extern "C" void cuda_preprocess(
    const unsigned char* src, float* dst,
    int src_width, int src_height, int src_channels,
    int dst_width, int dst_height);

namespace {

constexpr char kRawFrameChannel[] = "/vcf_raw_frames";
constexpr char kAnnotatedFrameChannel[] = "/vcf_annotated_frames";
constexpr char kMetadataChannel[] = "/vcf_metadata";
constexpr int kInputWidth = 640;
constexpr int kInputHeight = 640;
constexpr float kDefaultConfidence = 0.5f;
constexpr float kNmsThreshold = 0.45f;
constexpr size_t kShmHeaderSize = 64;

std::atomic<bool> g_running{true};

void SignalHandler(int /*signal*/) {
    g_running.store(false);
}

// ShmHeader matches the router definition
struct ShmHeader {
    uint64_t write_sequence;
    uint64_t frame_timestamp_ns;
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t frame_channels;
    uint32_t data_size;
    // Followed by semaphores (handled by router)
};

/**
 * Detection result structure.
 */
struct Detection {
    int class_id;
    float confidence;
    cv::Rect bbox;
};

/**
 * COCO class names for YOLO.
 */
const std::vector<std::string> kCocoClasses = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

/**
 * Color palette for drawing bounding boxes.
 */
cv::Scalar GetClassColor(int class_id) {
    static const std::vector<cv::Scalar> colors = {
        {255, 56, 56}, {255, 157, 151}, {255, 112, 31}, {255, 178, 29},
        {207, 210, 49}, {72, 249, 10}, {146, 204, 23}, {61, 219, 134},
        {26, 147, 52}, {0, 212, 187}, {44, 153, 168}, {0, 194, 255},
        {52, 69, 147}, {100, 115, 255}, {0, 24, 236}, {132, 56, 255},
        {82, 0, 133}, {203, 56, 255}, {255, 149, 200}, {255, 55, 199}
    };
    return colors[class_id % colors.size()];
}

/**
 * Shared memory reader for consuming frames from router.
 */
class ShmReader {
public:
    ShmReader(const std::string& name, size_t size)
        : name_(name), size_(size), fd_(-1), ptr_(nullptr), last_seq_(0) {}

    ~ShmReader() { Disconnect(); }

    bool Connect() {
        fd_ = shm_open(name_.c_str(), O_RDONLY, 0);
        if (fd_ == -1) return false;

        ptr_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (ptr_ == MAP_FAILED) {
            ptr_ = nullptr;
            return false;
        }
        return true;
    }

    bool HasNewFrame() const {
        if (!ptr_) return false;
        auto* header = reinterpret_cast<const ShmHeader*>(ptr_);
        return header->write_sequence > last_seq_;
    }

    cv::Mat ReadFrame() {
        if (!ptr_) return {};

        auto* header = reinterpret_cast<const ShmHeader*>(ptr_);
        uint64_t seq = header->write_sequence;
        if (seq <= last_seq_) return {};

        uint32_t w = header->frame_width;
        uint32_t h = header->frame_height;
        uint32_t ch = header->frame_channels;
        uint32_t data_size = header->data_size;

        if (w == 0 || h == 0 || data_size == 0) return {};

        // Read frame data
        const auto* data = reinterpret_cast<const unsigned char*>(ptr_) + kShmHeaderSize;
        cv::Mat frame(static_cast<int>(h), static_cast<int>(w),
                      ch == 3 ? CV_8UC3 : CV_8UC1);
        std::memcpy(frame.data, data, data_size);

        last_seq_ = seq;
        return frame;
    }

    uint64_t last_sequence() const { return last_seq_; }

    void Disconnect() {
        if (ptr_) { munmap(ptr_, size_); ptr_ = nullptr; }
        if (fd_ != -1) { close(fd_); fd_ = -1; }
    }

private:
    std::string name_;
    size_t size_;
    int fd_;
    void* ptr_;
    uint64_t last_seq_;
};

/**
 * Shared memory writer for publishing annotated frames.
 */
class ShmWriter {
public:
    ShmWriter(const std::string& name, size_t size)
        : name_(name), size_(size), fd_(-1), ptr_(nullptr), seq_(0) {}

    ~ShmWriter() { Disconnect(); }

    bool Connect() {
        fd_ = shm_open(name_.c_str(), O_RDWR, 0);
        if (fd_ == -1) return false;

        ptr_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (ptr_ == MAP_FAILED) {
            ptr_ = nullptr;
            return false;
        }
        return true;
    }

    bool WriteFrame(const cv::Mat& frame) {
        if (!ptr_ || frame.empty()) return false;

        size_t data_size = frame.total() * frame.elemSize();
        if (data_size + kShmHeaderSize > size_) return false;

        auto* header = reinterpret_cast<ShmHeader*>(ptr_);
        ++seq_;

        auto* data = reinterpret_cast<unsigned char*>(ptr_) + kShmHeaderSize;
        std::memcpy(data, frame.data, data_size);

        // Write header last (acts as memory fence for consumers)
        header->data_size = static_cast<uint32_t>(data_size);
        header->frame_channels = static_cast<uint32_t>(frame.channels());
        header->frame_height = static_cast<uint32_t>(frame.rows);
        header->frame_width = static_cast<uint32_t>(frame.cols);
        header->frame_timestamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        header->write_sequence = seq_;

        return true;
    }

    bool WriteMetadata(const std::string& json_str) {
        if (!ptr_) return false;

        size_t data_size = json_str.size();
        if (data_size + kShmHeaderSize > size_) return false;

        auto* header = reinterpret_cast<ShmHeader*>(ptr_);
        ++seq_;

        auto* data = reinterpret_cast<char*>(ptr_) + kShmHeaderSize;
        std::memcpy(data, json_str.c_str(), data_size);

        header->data_size = static_cast<uint32_t>(data_size);
        header->frame_width = 0;
        header->frame_height = 0;
        header->frame_channels = 0;
        header->frame_timestamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        header->write_sequence = seq_;

        return true;
    }

    void Disconnect() {
        if (ptr_) { munmap(ptr_, size_); ptr_ = nullptr; }
        if (fd_ != -1) { close(fd_); fd_ = -1; }
    }

private:
    std::string name_;
    size_t size_;
    int fd_;
    void* ptr_;
    uint64_t seq_;
};

/**
 * YOLO inference engine using OpenCV DNN with CUDA backend.
 */
class YoloDetector {
public:
    YoloDetector(const std::string& model_path, float confidence_threshold)
        : confidence_threshold_(confidence_threshold) {
        net_ = cv::dnn::readNetFromONNX(model_path);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "[YOLO] Loaded model: " << model_path
                  << " (CUDA backend)\n";
    }

    std::vector<Detection> Detect(const cv::Mat& frame) {
        // Preprocess: letterbox resize to 640x640
        cv::Mat blob;
        cv::dnn::blobFromImage(
            frame, blob, 1.0 / 255.0,
            cv::Size(kInputWidth, kInputHeight),
            cv::Scalar(0, 0, 0), true, false);

        net_.setInput(blob);

        // Forward pass
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        return PostProcess(frame, outputs);
    }

    double GetInferenceTimeMs() const {
        std::vector<double> times;
        net_.getPerfProfile(times);
        double total = 0;
        for (double t : times) total += t;
        return total * 1000.0 / cv::getTickFrequency();
    }

private:
    std::vector<Detection> PostProcess(
        const cv::Mat& frame, const std::vector<cv::Mat>& outputs) {

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        float x_factor = static_cast<float>(frame.cols) / kInputWidth;
        float y_factor = static_cast<float>(frame.rows) / kInputHeight;

        // YOLOv8 output format: [batch, 84, 8400] transposed
        const cv::Mat& output = outputs[0];
        int rows = output.size[2];  // Number of detections
        int dimensions = output.size[1];  // 4 (bbox) + 80 (classes)

        cv::Mat det_output = output.reshape(1, dimensions).t();

        for (int i = 0; i < rows; ++i) {
            const float* row = det_output.ptr<float>(i);
            float cx = row[0], cy = row[1], w = row[2], h = row[3];

            // Find best class
            float max_score = 0;
            int max_class = 0;
            for (int c = 4; c < dimensions; ++c) {
                if (row[c] > max_score) {
                    max_score = row[c];
                    max_class = c - 4;
                }
            }

            if (max_score < confidence_threshold_) continue;

            int x = static_cast<int>((cx - w / 2) * x_factor);
            int y = static_cast<int>((cy - h / 2) * y_factor);
            int bw = static_cast<int>(w * x_factor);
            int bh = static_cast<int>(h * y_factor);

            boxes.emplace_back(x, y, bw, bh);
            confidences.push_back(max_score);
            class_ids.push_back(max_class);
        }

        // Non-maximum suppression
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_,
                          kNmsThreshold, indices);

        std::vector<Detection> detections;
        for (int idx : indices) {
            detections.push_back({class_ids[idx], confidences[idx], boxes[idx]});
        }
        return detections;
    }

    cv::dnn::Net net_;
    float confidence_threshold_;
};

/**
 * Draw detection results on a frame.
 */
cv::Mat DrawDetections(const cv::Mat& frame,
                       const std::vector<Detection>& detections) {
    cv::Mat annotated = frame.clone();

    for (const auto& det : detections) {
        cv::Scalar color = GetClassColor(det.class_id);

        // Draw bounding box
        cv::rectangle(annotated, det.bbox, color, 2);

        // Draw label background
        std::string label = kCocoClasses[det.class_id] + " " +
                           std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        int baseline;
        cv::Size label_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(annotated,
                      cv::Point(det.bbox.x, det.bbox.y - label_size.height - 10),
                      cv::Point(det.bbox.x + label_size.width, det.bbox.y),
                      color, cv::FILLED);

        // Draw label text
        cv::putText(annotated, label,
                    cv::Point(det.bbox.x, det.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }

    return annotated;
}

/**
 * Generate JSON metadata for dashboard consumption.
 */
std::string GenerateMetadataJson(
    const std::vector<Detection>& detections,
    double inference_ms, double total_ms, uint64_t frame_seq) {

    std::string json = "{";
    json += "\"frame_seq\":" + std::to_string(frame_seq) + ",";
    json += "\"inference_ms\":" + std::to_string(inference_ms) + ",";
    json += "\"total_ms\":" + std::to_string(total_ms) + ",";
    json += "\"num_detections\":" + std::to_string(detections.size()) + ",";
    json += "\"detections\":[";

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        if (i > 0) json += ",";
        json += "{\"class\":\"" + kCocoClasses[d.class_id] + "\",";
        json += "\"confidence\":" + std::to_string(d.confidence) + ",";
        json += "\"bbox\":[" + std::to_string(d.bbox.x) + "," +
                std::to_string(d.bbox.y) + "," +
                std::to_string(d.bbox.width) + "," +
                std::to_string(d.bbox.height) + "]}";
    }

    json += "]}";
    return json;
}

}  // namespace

int main() {
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    std::cout << "=== Video Classifier Dashboard - YOLO Classifier (Process 2) ===\n";

    // Configuration
    std::string model_path = "yolov8n.onnx";
    if (const char* env = std::getenv("YOLO_MODEL")) {
        model_path = env;
    }
    float confidence = kDefaultConfidence;
    if (const char* env = std::getenv("YOLO_CONFIDENCE")) {
        confidence = std::stof(env);
    }
    size_t segment_size = 8 * 1024 * 1024;
    if (const char* env = std::getenv("SHM_SEGMENT_SIZE")) {
        segment_size = std::stoull(env);
    }

    // Connect to shared memory
    ShmReader reader(kRawFrameChannel, segment_size);
    ShmWriter frame_writer(kAnnotatedFrameChannel, segment_size);
    ShmWriter meta_writer(kMetadataChannel, 64 * 1024);

    std::cout << "[YOLO] Connecting to shared memory...\n";
    while (g_running.load()) {
        if (reader.Connect() && frame_writer.Connect() && meta_writer.Connect()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "[YOLO] Waiting for router...\n";
    }

    if (!g_running.load()) return 0;

    // Initialize YOLO detector
    std::cout << "[YOLO] Loading model: " << model_path << "\n";
    YoloDetector detector(model_path, confidence);

    // Processing loop
    uint64_t frames_processed = 0;
    double total_inference_ms = 0;
    auto start_time = std::chrono::steady_clock::now();

    std::cout << "[YOLO] Processing frames... (Ctrl+C to stop)\n";

    while (g_running.load()) {
        if (!reader.HasNewFrame()) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }

        auto frame_start = std::chrono::steady_clock::now();

        cv::Mat frame = reader.ReadFrame();
        if (frame.empty()) continue;

        // Run YOLO detection
        std::vector<Detection> detections = detector.Detect(frame);
        double inference_ms = detector.GetInferenceTimeMs();

        // Draw results
        cv::Mat annotated = DrawDetections(frame, detections);

        auto frame_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(
            frame_end - frame_start).count();

        // Publish annotated frame
        frame_writer.WriteFrame(annotated);

        // Publish metadata
        std::string metadata = GenerateMetadataJson(
            detections, inference_ms, total_ms, reader.last_sequence());
        meta_writer.WriteMetadata(metadata);

        frames_processed++;
        total_inference_ms += inference_ms;

        if (frames_processed % 50 == 0) {
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time).count();
            double fps = frames_processed / elapsed;
            double avg_ms = total_inference_ms / frames_processed;
            std::cout << "[YOLO] Processed " << frames_processed << " frames | "
                      << fps << " FPS | Avg inference: " << avg_ms << " ms\n";
        }
    }

    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    std::cout << "[YOLO] Finished. Processed " << frames_processed
              << " frames in " << elapsed << "s ("
              << frames_processed / elapsed << " FPS)\n";

    return 0;
}
