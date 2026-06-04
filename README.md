# rim_examples

Fully functional examples using the header-only libraries from [sbow/RoboticsIpcModule](https://github.com/sbow/RoboticsIpcModule).

## What is included

### C++ examples (built with CMake)
- `rim_point_to_point_uds`: very simple point-to-point UDS echo (no router)
- `rim_router_fanout_uds`: one source, one router, two consumers
- `rim_router_topic_subscription_uds`: topic-based subscription routing
- `rim_mixed_transport_demo`: mixed transport routing (SHM + UDS)

### Python project
- `examples/python/rim_frame_listener.py`: RouterFrame listener/decoder over UDS

### Dashboard web app
- `examples/dashboard`: Node.js + Express + WebSocket dashboard that visualizes RouterFrame traffic

## Build C++ examples

```bash
cmake -S /tmp/workspace/sbow/rim_examples -B /tmp/workspace/sbow/rim_examples/build
cmake --build /tmp/workspace/sbow/rim_examples/build -j
```

## Run C++ demos

```bash
/tmp/workspace/sbow/rim_examples/build/examples/cpp/rim_point_to_point_uds
/tmp/workspace/sbow/rim_examples/build/examples/cpp/rim_router_fanout_uds
/tmp/workspace/sbow/rim_examples/build/examples/cpp/rim_router_topic_subscription_uds
/tmp/workspace/sbow/rim_examples/build/examples/cpp/rim_mixed_transport_demo
```

## Run the Python listener

```bash
python3 -m venv /tmp/workspace/sbow/rim_examples/examples/python/.venv
source /tmp/workspace/sbow/rim_examples/examples/python/.venv/bin/activate
pip install -e /tmp/workspace/sbow/rim_examples/examples/python
python /tmp/workspace/sbow/rim_examples/examples/python/rim_frame_listener.py --path /tmp/rim_examples_python_listener.sock
```

## Run the dashboard web app

```bash
cd /tmp/workspace/sbow/rim_examples/examples/dashboard
npm install
npm start
```

Then open `http://localhost:8080`.
