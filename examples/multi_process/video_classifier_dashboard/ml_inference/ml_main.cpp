/**
 * @file ml_main.cpp
 * @brief Process 2 — C++/CUDA YOLO inference peer (RIM peer ``ml_inference``).
 *
 * Pipeline per frame:
 *   1. block on a 64 B RouterFrame notification (topic 10) from the router;
 *   2. read the raw RGB frame it points at from the source SHM ring;
 *   3. run the detector on the GPU (YOLOv8/TensorRT or the CUDA saliency
 *      fallback) and overlay boxes on a copy of the frame;
 *   4. write the annotated frame + a packed detection list into two SHM
 *      sideband rings;
 *   5. publish a topic-20 RouterFrame whose sideband descriptor + 32 B stats
 *      payload tell the dashboard what to render.
 *
 * RIM surface used: topology_loader (one shared TOML), IpcEndpoint<Uds> as a
 * datagram peer, RouterFrame v2, and the sideband descriptor fields.
 */

#include "frame_shm.hpp"
#include "pipeline.hpp"
#include "wire.hpp"

#include "ipc.hpp"
#include "router/frame.hpp"
#include "router/timestamp.hpp"   // router_now_ns() — CLOCK_MONOTONIC_RAW (ADR 0010)
#include "router/topology_loader.hpp"

#include "toml.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace rim_vcd;

namespace {

constexpr uint8_t kPeerId = 5;
volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

const char* uds_path(const PeerAddress& a) { return a.u.uds_path; }

}  // namespace

int main(int argc, char** argv) {
    const std::string config = argc > 1 ? argv[1] : "topology.toml";
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    // RIM topology: peers, routes, sideband region names — all from one file.
    LoadedTopology loaded = load_topology_from_toml_file(config);
    const RouterTopology topo = loaded.view();
    const PeerEntry* me = peer_by_id(topo, kPeerId);
    if (!me || !topo.has_listen_uds) {
        std::fprintf(stderr, "[ml] topology missing peer %d or uds listen\n", kPeerId);
        return 1;
    }
    const std::string my_path = uds_path(me->local);
    const std::string router_path = uds_path(topo.listen_uds);

    // Sideband region names: source frames come from peer 1; we own annotated +
    // detection regions on peer 5.
    size_t n_src = 0, n_self = 0;
    const SidebandRegion* src_sb = loaded.sidebands_for(1, n_src);
    const SidebandRegion* self_sb = loaded.sidebands_for(kPeerId, n_self);
    if (!src_sb || n_self < 2) {
        std::fprintf(stderr, "[ml] missing sideband declarations\n");
        return 1;
    }

    // Demo-specific knobs ride in the same TOML ([video]/[ml]).
    auto tbl = toml::parse_file(config);
    const int width = tbl["video"]["width"].value_or(640);
    const int height = tbl["video"]["height"].value_or(360);
    const std::string model = tbl["ml"]["model_path"].value_or(std::string{});
    const float conf = static_cast<float>(tbl["ml"]["conf_thresh"].value_or(0.35));
    const uint32_t frame_bytes = static_cast<uint32_t>(width) * height * 3;

    auto pipeline = make_pipeline(width, height, model, conf);
    std::fprintf(stderr, "[ml] backend=%s %dx%d conf=%.2f model=%s\n",
                 pipeline->backend(), width, height, conf,
                 model.empty() ? "(none)" : model.c_str());

    // Bind our UDS socket FIRST so the router always has a destination — the
    // kernel queues incoming datagrams even before we enter the recv loop.
    // (Bind after the SHM wait below and the router would hit ECONNREFUSED.)
    IpcEndpoint<Uds> ep;
    ep.bind(Uds::BindParams{.path = my_path});
    std::fprintf(stderr, "[ml] peer bound %s -> router %s\n",
                 my_path.c_str(), router_path.c_str());

    // Create our output rings; open the source ring (created by the Python
    // publisher) — spin until it appears.
    FrameShm annotated = FrameShm::open(self_sb[0].name, width, height, 3, 4, true);
    FrameShm detections = FrameShm::open(self_sb[1].name,
                                         kMaxDetections * kDetRecordBytes, 1, 1, 4, true);

    FrameShm source;
    bool source_ready = false;
    for (int i = 0; i < 400 && !g_stop; ++i) {
        try {
            source = FrameShm::open(src_sb[0].name, width, height, 3, 4, false);
            source_ready = true;
            break;
        } catch (const std::exception&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    if (!source_ready) {
        std::fprintf(stderr, "[ml] source ring %s never appeared\n", src_sb[0].name);
        return g_stop ? 0 : 1;
    }

    std::vector<uint8_t> src_buf(frame_bytes), dst_buf(frame_bytes);
    std::vector<uint8_t> det_buf(kMaxDetections * kDetRecordBytes);
    char rx[256];
    uint32_t out_seq = 0;

    while (!g_stop) {
        Buffer buf = Buffer::writable(rx, sizeof(rx));
        Uds::RecvResult rr{};
        if (!ep.try_recv(buf, rr)) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }
        if (buf.size < kRouterFrameSize) continue;

        RouterFrame in;
        std::memcpy(in.bytes, rx, kRouterFrameSize);
        if (in.topic_id() != kTopicRawFrame) continue;

        RawMeta meta{};
        std::memcpy(&meta, in.bytes + kRouterPayloadOffset, sizeof(meta));

        if (!source.read(in.sideband_seq(), src_buf.data())) continue;

        std::vector<Detection> d;
        const PassResult pr = pipeline->process(src_buf.data(), dst_buf.data(), d);

        const uint64_t ann_seq = annotated.write(dst_buf.data());

        // Pack detections into the metadata sideband (same seq as annotated).
        const int nb = std::min<int>(pr.num_boxes, kMaxDetections);
        std::memset(det_buf.data(), 0, det_buf.size());
        for (int i = 0; i < nb; ++i) {
            DetRecord rec{d[i].x, d[i].y, d[i].w, d[i].h, d[i].score,
                          static_cast<uint32_t>(d[i].cls)};
            std::memcpy(det_buf.data() + i * kDetRecordBytes, &rec, sizeof(rec));
        }
        detections.write(det_buf.data());

        // Publish the topic-20 notification with stats in the inline payload.
        AnnStats stats{};
        stats.width = static_cast<uint16_t>(width);
        stats.height = static_cast<uint16_t>(height);
        stats.num_boxes = static_cast<uint16_t>(nb);
        stats.infer_ms = pr.infer_ms;
        stats.overlay_ms = pr.overlay_ms;
        // Same clock (CLOCK_MONOTONIC_RAW) as the publishers, so the difference
        // is a true capture->annotate latency; clamp against clock jitter.
        const int64_t dt = static_cast<int64_t>(router_now_ns()) -
                           static_cast<int64_t>(meta.capture_ns);
        stats.e2e_ms = dt > 0 ? dt / 1.0e6f : 0.0f;
        stats.source_seq = in.seq();

        RouterFrame out;
        out.init(kPeerId);
        out.set_topic_id(kTopicAnnotatedFrame);
        out.set_seq(out_seq++);
        out.set_timestamp_ns(router_now_ns());
        out.set_flags(kFlagHasSideband);
        out.set_sideband_idx(0);
        out.set_sideband_seq(ann_seq);
        out.set_sideband_len(frame_bytes);
        out.set_payload(&stats, sizeof(stats));
        Uds::send_to(ep.fd(), router_path, out.read_only());

        if ((out_seq & 31) == 0) {
            std::fprintf(stderr, "[ml] seq=%u boxes=%d infer=%.1fms e2e=%.1fms\n",
                         out_seq, nb, pr.infer_ms, stats.e2e_ms);
        }
    }

    std::fprintf(stderr, "[ml] shutting down\n");
    return 0;
}
