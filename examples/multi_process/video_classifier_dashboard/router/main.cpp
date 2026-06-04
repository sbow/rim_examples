/**
 * Process 0: Message Router
 *
 * Routes video frames between processes via shared memory segments.
 * Manages SHM lifecycle and coordinates publisher/subscriber connections.
 */

#include <iostream>
#include <cstring>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <mutex>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>

namespace {

constexpr size_t kDefaultSegmentSize = 8 * 1024 * 1024;  // 8 MB
constexpr int kMaxSubscribers = 8;
constexpr char kRawFrameChannel[] = "/vcf_raw_frames";
constexpr char kAnnotatedFrameChannel[] = "/vcf_annotated_frames";
constexpr char kMetadataChannel[] = "/vcf_metadata";

std::atomic<bool> g_running{true};

void SignalHandler(int /*signal*/) {
    g_running.store(false);
}

/**
 * Header stored at the beginning of each shared memory segment.
 */
struct ShmHeader {
    std::atomic<uint64_t> write_sequence;
    std::atomic<uint64_t> frame_timestamp_ns;
    std::atomic<uint32_t> frame_width;
    std::atomic<uint32_t> frame_height;
    std::atomic<uint32_t> frame_channels;
    std::atomic<uint32_t> data_size;
    sem_t write_lock;
    sem_t read_ready;
};

/**
 * Manages a single shared memory channel.
 */
class ShmChannel {
public:
    ShmChannel(const std::string& name, size_t size)
        : name_(name), size_(size), fd_(-1), ptr_(nullptr) {}

    ~ShmChannel() { Destroy(); }

    bool Create() {
        fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd_ == -1) {
            std::cerr << "[Router] Failed to create SHM: " << name_ << "\n";
            return false;
        }

        if (ftruncate(fd_, static_cast<off_t>(size_)) == -1) {
            std::cerr << "[Router] Failed to resize SHM: " << name_ << "\n";
            return false;
        }

        ptr_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (ptr_ == MAP_FAILED) {
            std::cerr << "[Router] Failed to mmap SHM: " << name_ << "\n";
            ptr_ = nullptr;
            return false;
        }

        // Initialize header
        auto* header = reinterpret_cast<ShmHeader*>(ptr_);
        header->write_sequence.store(0);
        header->frame_timestamp_ns.store(0);
        header->frame_width.store(0);
        header->frame_height.store(0);
        header->frame_channels.store(0);
        header->data_size.store(0);
        sem_init(&header->write_lock, 1, 1);
        sem_init(&header->read_ready, 1, 0);

        std::cout << "[Router] Created channel: " << name_
                  << " (" << size_ / (1024 * 1024) << " MB)\n";
        return true;
    }

    void Destroy() {
        if (ptr_) {
            auto* header = reinterpret_cast<ShmHeader*>(ptr_);
            sem_destroy(&header->write_lock);
            sem_destroy(&header->read_ready);
            munmap(ptr_, size_);
            ptr_ = nullptr;
        }
        if (fd_ != -1) {
            close(fd_);
            fd_ = -1;
        }
        shm_unlink(name_.c_str());
        std::cout << "[Router] Destroyed channel: " << name_ << "\n";
    }

    uint64_t GetSequence() const {
        if (!ptr_) return 0;
        auto* header = reinterpret_cast<ShmHeader*>(ptr_);
        return header->write_sequence.load(std::memory_order_acquire);
    }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    size_t size_;
    int fd_;
    void* ptr_;
};

/**
 * Router that manages multiple SHM channels and monitors their health.
 */
class Router {
public:
    Router() = default;

    bool Initialize() {
        size_t segment_size = kDefaultSegmentSize;
        if (const char* env = std::getenv("SHM_SEGMENT_SIZE")) {
            segment_size = std::stoull(env);
        }

        // Create channels for raw frames, annotated frames, and metadata
        channels_.emplace_back(
            std::make_unique<ShmChannel>(kRawFrameChannel, segment_size));
        channels_.emplace_back(
            std::make_unique<ShmChannel>(kAnnotatedFrameChannel, segment_size));
        channels_.emplace_back(
            std::make_unique<ShmChannel>(kMetadataChannel, 1024 * 64));

        for (auto& channel : channels_) {
            if (!channel->Create()) {
                return false;
            }
        }

        std::cout << "[Router] Initialized with " << channels_.size()
                  << " channels\n";
        return true;
    }

    void Run() {
        std::cout << "[Router] Running... (Ctrl+C to stop)\n";

        uint64_t stats_interval_ms = 2000;
        auto last_stats = std::chrono::steady_clock::now();

        std::unordered_map<std::string, uint64_t> last_sequences;
        for (auto& ch : channels_) {
            last_sequences[ch->name()] = 0;
        }

        while (g_running.load()) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_stats);

            if (elapsed.count() >= static_cast<long long>(stats_interval_ms)) {
                PrintStats(last_sequences);
                last_stats = now;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void Shutdown() {
        std::cout << "[Router] Shutting down...\n";
        channels_.clear();
        std::cout << "[Router] Shutdown complete.\n";
    }

private:
    void PrintStats(std::unordered_map<std::string, uint64_t>& last_sequences) {
        std::cout << "[Router] === Channel Stats ===\n";
        for (auto& ch : channels_) {
            uint64_t current = ch->GetSequence();
            uint64_t delta = current - last_sequences[ch->name()];
            last_sequences[ch->name()] = current;
            std::cout << "  " << ch->name() << ": seq=" << current
                      << " (+" << delta << ")\n";
        }
    }

    std::vector<std::unique_ptr<ShmChannel>> channels_;
};

}  // namespace

int main() {
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    std::cout << "=== Video Classifier Dashboard - Router (Process 0) ===\n";

    Router router;
    if (!router.Initialize()) {
        std::cerr << "[Router] Failed to initialize. Exiting.\n";
        return 1;
    }

    router.Run();
    router.Shutdown();

    return 0;
}
