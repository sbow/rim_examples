#include "ipc.hpp"
#include "router/frame.hpp"
#include "router/mixed_router_server.hpp"
#include "router/peer_table.hpp"
#include "router/routing.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <unistd.h>

namespace {

constexpr const char* kSensorShm = "/rim_examples_mixed_sensor";
constexpr const char* kSubShm = "/rim_examples_mixed_sub";
constexpr const char* kRouterShm = "/rim_examples_mixed_router";
constexpr const char* kRouterUds = "/tmp/rim_examples_mixed_router.sock";
constexpr const char* kNodeUds = "/tmp/rim_examples_mixed_node.sock";

constexpr uint8_t kSensorId = 1;
constexpr uint8_t kNodeId = 2;
constexpr uint8_t kSubId = 3;

constexpr uint32_t kSlotCount = 128;
constexpr uint32_t kMaxPayload = 64;
constexpr int kFrameCount = 20;

const PeerEntry kPeers[] = {
    {kSensorId, "sensor", peer_shm(kSensorShm), kSlotCount, kMaxPayload},
    {kNodeId, "node", peer_uds(kNodeUds), 0, 0},
    {kSubId, "sub", peer_shm(kSubShm), kSlotCount, kMaxPayload},
};

const RouterTopology kTopology = {
    .peers = kPeers,
    .peer_count = std::size(kPeers),
    .router_listen = peer_shm(kRouterShm),
    .has_listen_uds = true,
    .listen_uds = peer_uds(kRouterUds),
};

constexpr RouteRule kRules[] = {
    make_route(kSensorId, kNodeId),
    make_route(kNodeId, kSubId),
};

void cleanup() {
  ::shm_unlink(kSensorShm);
  ::shm_unlink(kSubShm);
  ::shm_unlink(kRouterShm);
  ::unlink(kRouterUds);
  ::unlink(kNodeUds);
}

uint64_t now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

}  // namespace

int main() {
  cleanup();

  MixedRouterServer router(kTopology);
  router.bind_router();

  IpcEndpoint<ShmSpsc> sensor;
  bind_shm_endpoint(sensor, kPeers[0], false);

  IpcEndpoint<ShmSpsc> subscriber;
  bind_shm_endpoint(subscriber, kPeers[2], false);

  IpcEndpoint<Uds> node;
  node.bind(Uds::BindParams{.path = kNodeUds});

  std::sig_atomic_t stop_router = 0;
  std::atomic<bool> stop_workers{false};
  std::atomic<int> relayed{0};
  std::atomic<int> received{0};
  std::atomic<int> wrong_source{0};

  RouterRunOptions options;
  options.idle_sleep_us = 0;

  std::thread router_thread([&]() {
    router.run(kRules, std::size(kRules), now_ns,
               [](uint8_t, uint8_t, const RouterFrame&) {}, options, &stop_router);
  });

  std::thread node_thread([&]() {
    char storage[256]{};
    while (!stop_workers.load(std::memory_order_relaxed)) {
      Buffer buf = Buffer::writable(storage, sizeof(storage));
      Uds::RecvResult recv{};
      if (!node.try_recv(buf, recv)) {
        std::this_thread::yield();
        continue;
      }
      if (buf.size < kRouterFrameSize) {
        continue;
      }
      Uds::send_to(node.fd(), kRouterUds, Buffer::read_only(storage, kRouterFrameSize));
      relayed.fetch_add(1, std::memory_order_relaxed);
    }
  });

  std::thread sub_thread([&]() {
    char storage[256]{};
    while (!stop_workers.load(std::memory_order_relaxed)) {
      Buffer buf = Buffer::writable(storage, sizeof(storage));
      ShmSpsc::RecvResult recv{};
      if (!ShmSpsc::try_recv(subscriber.handle(), buf, recv)) {
        std::this_thread::yield();
        continue;
      }
      if (buf.size < kRouterFrameSize) {
        continue;
      }
      RouterFrame frame;
      std::memcpy(frame.bytes, storage, kRouterFrameSize);
      if (frame.source() != kNodeId) {
        wrong_source.fetch_add(1, std::memory_order_relaxed);
      }
      received.fetch_add(1, std::memory_order_relaxed);
    }
  });

  for (int i = 0; i < kFrameCount; ++i) {
    RouterFrame frame;
    frame.init(kSensorId);
    frame.set_seq(static_cast<uint32_t>(i));
    frame.set_payload("mixed-demo");

    ShmSpsc::SendParams params{.payload = frame.read_only()};
    while (sensor.try_send(params, frame.read_only()) != ShmSendResult::Ok) {
      std::this_thread::yield();
    }
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  stop_workers.store(true, std::memory_order_relaxed);
  stop_router = 1;
  router_thread.join();
  node_thread.join();
  sub_thread.join();

  std::cout << "mixed transport relayed=" << relayed.load()
            << " received=" << received.load()
            << " wrong_source=" << wrong_source.load() << '\n';

  const bool ok = relayed.load() >= kFrameCount
      && received.load() >= kFrameCount
      && wrong_source.load() == 0;

  cleanup();
  return ok ? 0 : 1;
}
