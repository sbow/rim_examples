#include "ipc.hpp"
#include "router/factory.hpp"
#include "router/node.hpp"
#include "router/peer_table.hpp"
#include "router/routing.hpp"

#include <chrono>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace {

constexpr uint8_t kSourceId = 1;
constexpr uint8_t kDepthConsumerId = 2;
constexpr uint8_t kImuConsumerId = 3;
constexpr uint8_t kMapConsumerId = 4;

constexpr uint16_t kDepthTopic = 100;
constexpr uint16_t kImuTopic = 200;
constexpr uint16_t kMapTopic = 300;

constexpr const char* kRouterPath = "/tmp/rim_examples_topic_router.sock";
constexpr const char* kSourcePath = "/tmp/rim_examples_topic_source.sock";
constexpr const char* kDepthPath = "/tmp/rim_examples_topic_depth.sock";
constexpr const char* kImuPath = "/tmp/rim_examples_topic_imu.sock";
constexpr const char* kMapPath = "/tmp/rim_examples_topic_map.sock";

constexpr PeerEntry kPeers[] = {
    {kSourceId, "sensor", peer_uds(kSourcePath), 0, 0},
    {kDepthConsumerId, "depth", peer_uds(kDepthPath), 0, 0},
    {kImuConsumerId, "imu", peer_uds(kImuPath), 0, 0},
    {kMapConsumerId, "map", peer_uds(kMapPath), 0, 0},
};

constexpr RouterTopology kTopology = {
    .peers = kPeers,
    .peer_count = std::size(kPeers),
    .router_listen = peer_uds(kRouterPath),
};

constexpr RouteRule kRules[] = {
    make_topic_route(kSourceId, kDepthTopic, kDepthConsumerId),
    make_topic_route(kSourceId, kImuTopic, kImuConsumerId),
    make_topic_route(kSourceId, kMapTopic, kMapConsumerId),
};

void cleanup() {
  ::unlink(kRouterPath);
  ::unlink(kSourcePath);
  ::unlink(kDepthPath);
  ::unlink(kImuPath);
  ::unlink(kMapPath);
}

uint64_t now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool wait_for_topic(DatagramRouterClient<Uds>& client, uint16_t topic) {
  client.set_recv_timeout_ms(50);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  RouterFrame frame;
  while (std::chrono::steady_clock::now() < deadline) {
    if (client.recv_message(frame)
        && frame.source() == kSourceId
        && frame.topic_id() == topic) {
      std::cout << "topic " << topic << " payload: " << frame.payload() << '\n';
      return true;
    }
  }
  return false;
}

void publish_topic(DatagramRouterClient<Uds>& source, uint16_t topic, const char* payload) {
  RouterFrame frame;
  frame.init(kSourceId);
  frame.set_topic_id(topic);
  frame.set_payload(payload);
  source.link().send_to_router(frame);
}

}  // namespace

int main() {
  cleanup();

  std::sig_atomic_t stop = 0;
  auto router = make_datagram_router_server<Uds>(kTopology);
  bind_datagram_router_listen(router, kTopology);

  RouterRunOptions options;
  options.poll_timeout_ms = 50;
  options.idle_sleep_us = 0;

  std::thread router_thread([&]() {
    router.run(kRules, std::size(kRules), now_ns,
               [](uint8_t, uint8_t, const RouterFrame&) {}, options, &stop);
  });

  auto source = make_datagram_router_client<Uds>(kTopology, kSourceId);
  auto depth = make_datagram_router_client<Uds>(kTopology, kDepthConsumerId);
  auto imu = make_datagram_router_client<Uds>(kTopology, kImuConsumerId);
  auto map = make_datagram_router_client<Uds>(kTopology, kMapConsumerId);

  publish_topic(source, kDepthTopic, "depth-42");
  publish_topic(source, kImuTopic, "imu-43");
  publish_topic(source, kMapTopic, "map-44");

  const bool ok = wait_for_topic(depth, kDepthTopic)
      && wait_for_topic(imu, kImuTopic)
      && wait_for_topic(map, kMapTopic);

  stop = 1;
  router_thread.join();
  cleanup();

  if (!ok) {
    throw std::runtime_error("topic subscription routing failed");
  }

  return 0;
}
