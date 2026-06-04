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
#include <string>
#include <thread>
#include <unistd.h>

namespace {

constexpr uint8_t kSourceId = 1;
constexpr uint8_t kConsumerAId = 2;
constexpr uint8_t kConsumerBId = 3;

constexpr const char* kRouterPath = "/tmp/rim_examples_router_fanout.sock";
constexpr const char* kSourcePath = "/tmp/rim_examples_router_source.sock";
constexpr const char* kConsumerAPath = "/tmp/rim_examples_router_consumer_a.sock";
constexpr const char* kConsumerBPath = "/tmp/rim_examples_router_consumer_b.sock";

constexpr PeerEntry kPeers[] = {
    {kSourceId, "source", peer_uds(kSourcePath), 0, 0},
    {kConsumerAId, "consumer-a", peer_uds(kConsumerAPath), 0, 0},
    {kConsumerBId, "consumer-b", peer_uds(kConsumerBPath), 0, 0},
};

constexpr RouterTopology kTopology = {
    .peers = kPeers,
    .peer_count = std::size(kPeers),
    .router_listen = peer_uds(kRouterPath),
};

constexpr RouteRule kRules[] = {
    make_route(kSourceId, kConsumerAId, kConsumerBId),
};

void cleanup() {
  ::unlink(kRouterPath);
  ::unlink(kSourcePath);
  ::unlink(kConsumerAPath);
  ::unlink(kConsumerBPath);
}

uint64_t now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool wait_for_source_message(DatagramRouterClient<Uds>& client, RouterFrame& frame) {
  client.set_recv_timeout_ms(50);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (client.recv_message(frame) && frame.source() == kSourceId) {
      return true;
    }
  }
  return false;
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
  auto consumer_a = make_datagram_router_client<Uds>(kTopology, kConsumerAId);
  auto consumer_b = make_datagram_router_client<Uds>(kTopology, kConsumerBId);

  source.send_message(kSourceId, "fanout-message");

  RouterFrame frame_a;
  RouterFrame frame_b;

  if (!wait_for_source_message(consumer_a, frame_a)) {
    stop = 1;
    router_thread.join();
    cleanup();
    throw std::runtime_error("consumer-a did not receive source message");
  }

  if (!wait_for_source_message(consumer_b, frame_b)) {
    stop = 1;
    router_thread.join();
    cleanup();
    throw std::runtime_error("consumer-b did not receive source message");
  }

  std::cout << "fanout consumer-a payload: " << frame_a.payload() << '\n';
  std::cout << "fanout consumer-b payload: " << frame_b.payload() << '\n';

  const bool ok = frame_a.payload() == "fanout-message"
      && frame_b.payload() == "fanout-message";

  stop = 1;
  router_thread.join();
  cleanup();
  return ok ? 0 : 1;
}
