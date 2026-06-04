/**
 * @file router_main.cpp
 * @brief Process 0 — the RoboticsIpcModule message router.
 *
 * This is deliberately tiny: the whole point of RIM is that standing up a
 * production-shaped, mixed-transport router is a dozen lines on top of the
 * header-only library. We load the shared @c topology.toml, let RIM decide the
 * router shape (here: a @c MixedRouterServer because the peers span UDS + UDP),
 * bind, and run the cooperative forward loop until SIGINT/SIGTERM.
 *
 * Build/usage:
 * @code
 *   rim_vcd_router <path/to/topology.toml>
 * @endcode
 */

#include "router_app.h"                  // install_router_stop_handlers, logger
#include "router/mixed_router_server.hpp"
#include "router/timestamp.hpp"          // router_now_ns (CLOCK_MONOTONIC_RAW)
#include "router/topology_loader.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

namespace {

/// Minimal leveled logger → stderr, matching the reference router_server demo.
void stderr_logger(int level, const char* msg, std::size_t len) {
    const char* tag = (level == ROUTER_LOG_ERR)  ? "[router][err]  "
                    : (level == ROUTER_LOG_WARN) ? "[router][warn] "
                                                 : "[router][info] ";
    std::string out(tag);
    out.append(msg, len);
    out.push_back('\n');
    (void)!::write(STDERR_FILENO, out.data(), out.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <topology.toml>\n", argv[0]);
        return 1;
    }

    install_router_stop_handlers();   // SIGINT/SIGTERM flip the stop flag
    router_set_log_fn(stderr_logger);

    try {
        // One file drives every process; the router reads peers + routes from it.
        LoadedTopology loaded = load_topology_from_toml_file(argv[1]);
        const RouterTopology topo = loaded.view();

        if (!topology_is_mixed(topo)) {
            router_log(ROUTER_LOG_WARN,
                       "topology is single-transport; this demo expects UDS+UDP");
        }

        MixedRouterServer server(topo);
        if (topo.has_listen_uds) {
            ::unlink(topo.listen_uds.u.uds_path);  // clear a stale socket
        }
        server.bind_router();

        std::string kinds;
        if (server.has_shm()) kinds += "SHM ";
        if (server.has_uds()) kinds += "UDS ";
        if (server.has_udp()) kinds += "UDP ";
        router_log(ROUTER_LOG_INFO, "mixed-transport router serving: " + kinds);

        RouterRunOptions opts;
        opts.poll_timeout_ms = 200;
        opts.idle_sleep_us = 200;       // ~idle: yield the core (ADR 0007)

        server.run(
            loaded.routes(), loaded.route_count(), router_now_ns,
            [&](uint8_t src, uint8_t dst, const RouterFrame& f) {
                // One line per forwarded frame; cheap because frames are sparse
                // (one per video frame, not per pixel).
                router_log(ROUTER_LOG_INFO,
                           std::string("forward ") + peer_display_name(topo, src) +
                           " -> " + peer_display_name(topo, dst) +
                           " topic=" + std::to_string(f.topic_id()) +
                           " seq=" + std::to_string(f.seq()));
            },
            opts);

        router_log(ROUTER_LOG_INFO,
                   "shutdown; forwarded=" + std::to_string(server.forwarded()) +
                   " dropped_full=" + std::to_string(server.dropped_full()));
    } catch (const std::exception& e) {
        router_log(ROUTER_LOG_ERR, std::string("fatal: ") + e.what());
        return 1;
    }
    return 0;
}
