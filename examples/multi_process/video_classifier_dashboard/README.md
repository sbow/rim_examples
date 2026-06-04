# Video Classifier Dashboard

A multi-process machine learning example demonstrating inter-process communication
using shared memory (SHM) for real-time video classification with a web dashboard.

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐     ┌───────────────┐
│  Process 0      │     │  Process 1           │     │  Process 2          │     │  Process 3    │
│  C++ Router     │◄───►│  Python Video Pub    │     │  C++ CUDA YOLO      │     │  Node.js Dash │
│                 │◄───►│  (SHM Publisher)     │     │  (Classifier)       │◄───►│  (Web UI)     │
│  Routes frames  │     │  Streams online video│     │  Publishes annotated│     │  Shows result │
│  between procs  │◄───►│  to shared memory    │     │  frames + boxes     │◄───►│  and stats    │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘     └───────────────┘
```

## Components

| Process | Language | Role |
|---------|----------|------|
| 0 | C++ | Message router - routes frames between processes via SHM |
| 1 | Python | Video publisher - captures online video stream, publishes raw frames |
| 2 | C++ + CUDA | YOLO classifier - runs inference, publishes annotated frames + bounding boxes |
| 3 | Node.js | Dashboard - displays processed frames and performance statistics |

## Prerequisites

- CMake 3.18+
- CUDA Toolkit 11.0+
- OpenCV 4.x (with CUDA support recommended)
- Python 3.8+ with `opencv-python`, `numpy`
- Node.js 16+
- YOLO v8 weights (auto-downloaded on first run)

## Building

```bash
# Build C++ components (router and classifier)
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j

# Install Python dependencies
pip install -r video_publisher/requirements.txt

# Install Node.js dependencies
cd dashboard && npm install && cd ..
```

## Running

Use the launch script to start all processes:

```bash
./launch.sh
```

Or start each process individually:

```bash
# Terminal 1: Router
./build/router

# Terminal 2: Video Publisher
python video_publisher/main.py --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Terminal 3: YOLO Classifier
./build/yolo_classifier

# Terminal 4: Dashboard
cd dashboard && npm start
```

Then open http://localhost:3000 in your browser to view the dashboard.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCE` | YouTube sample stream | URL of video source |
| `SHM_SEGMENT_SIZE` | 8388608 (8MB) | Shared memory segment size |
| `YOLO_MODEL` | yolov8n.onnx | YOLO model file |
| `YOLO_CONFIDENCE` | 0.5 | Detection confidence threshold |
| `DASHBOARD_PORT` | 3000 | Web dashboard port |
