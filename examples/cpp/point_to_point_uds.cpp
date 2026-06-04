#include "ipc.hpp"

#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>

namespace {

constexpr const char* kServerPath = "/tmp/rim_examples_p2p_server.sock";
constexpr const char* kClientPath = "/tmp/rim_examples_p2p_client.sock";

void cleanup() {
  ::unlink(kServerPath);
  ::unlink(kClientPath);
}

}  // namespace

int main() {
  cleanup();

  std::sig_atomic_t stop = 0;
  UdsEchoServer server(Uds::BindParams{.path = kServerPath});
  std::thread server_thread([&]() { server.run_until(&stop, 50); });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  UdsEchoClient client(Uds::BindParams{.path = kClientPath});
  const std::string message = "hello-rim";

  char recv_storage[256]{};
  Buffer recv_buffer = Buffer::writable(recv_storage, sizeof(recv_storage));
  Buffer payload = Buffer::read_only(message.data(), message.size());

  Buffer echoed = client.exchange(
      Uds::SendParams{
          .server_path = kServerPath,
          .client_path = "",
          .payload = payload,
      },
      recv_buffer);

  std::string reply(static_cast<const char*>(echoed.data), echoed.size);
  std::cout << "point-to-point echo reply: " << reply << '\n';

  stop = 1;
  server_thread.join();
  cleanup();
  return reply == message ? 0 : 1;
}
