# Video Classifier Dashboard — a multi-process RoboticsIpcModule showcase

A four-process, three-language pipeline that streams a **YouTube** video,
classifies it with **YOLO on CUDA**, overlays bounding boxes, and renders the
result in a live browser dashboard — all wired together by the header-only
**RoboticsIpcModule (RIM)** message fabric and **one shared `topology.toml`**.

```
 ┌────────────┐   raw RGB        ┌─────────────┐  annotated RGB   ┌──────────────┐
 │ video_src  │  ───/dev/shm──▶  │ ml_inference│  ───/dev/shm──▶  │  dashboard   │
 │ (Python)   │                  │ (C++ + CUDA)│                  │  (Node.js)   │──▶ browser
 └─────┬──────┘                  └──────┬──────┘                  └──────▲───────┘
       │ topic 10 (UDS)                 │ topic 20 (UDS)                 │ (UDP)
       │   64 B notify                  │   64 B notify + stats          │
       └───────────────▶  ┌─────────────────────────┐  ◀──────────────-─┘
                          │  router (C++)            │
                          │  MixedRouterServer       │   forwards control frames only
                          │  UDS + UDP, one process  │
                          └─────────────────────────┘
```

The clever bit (and the RIM design pattern this demonstrates): **the megabytes
of pixels never pass through the router.** The router only forwards 64 B
`RouterFrame` *notifications* whose sideband descriptor (`sideband_seq`,
`sideband_len`, `sideband_idx`) points each consumer at the right slot in a
shared-memory ring declared as a `[[peers.sideband]]` in the topology. This is
exactly the control-plane / bulk-data split from RIM's ADR 0005 / ADR 0008.

## What each process shows off

| Process | Lang | RIM feature exercised |
|---|---|---|
| `router/` | C++ | `topology_loader` + `MixedRouterServer` (UDS+UDP in one hop, Phase H) |
| `video_source/` | Python | the ctypes UDS bridge + a sideband SHM producer |
| `ml_inference/` | C++/CUDA | `IpcEndpoint<Uds>` peer, RouterFrame v2, **two** sideband regions per peer |
| `dashboard/` | Node.js | the UDP bridge + SHM sideband consumer → WebSocket |

`common/frame_shm.{hpp,py,js}` is the same lock-free "newest-frame-wins" SHM
ring ported to all three languages so they interoperate byte-for-byte.

## Quick start

```bash
cd examples/multi_process/video_classifier_dashboard
./launch.sh                                  # uses the default lofi-girl live stream
# ...or pick your own source:
./launch.sh "https://www.youtube.com/watch?v=<id>"
```

Then open <http://localhost:8080>. Press `Ctrl-C` to stop everything (the
launcher cleans up all `/dev/shm/rim_vcd_*` regions and `/tmp/rim_vcd_*.sock`
sockets).

`launch.sh` builds the C++ peers (CMake), creates a Python venv for `yt-dlp`,
runs `npm install` for the dashboard, then starts all four processes.

## Requirements

External dependencies are allowed here (unlike RIM core):

- **System:** `cmake` ≥ 3.20, a C++20 compiler, `ffmpeg`, `node` ≥ 18, `python3` ≥ 3.10.
- **GPU (optional):** CUDA toolkit. Without it the build falls back to a pure-CPU
  detector so the pipeline still runs — set `VCD_FORCE_CPU=1` to force it.
- **Python:** `yt-dlp` (installed into a local venv by the launcher).
- **Node:** `ws`, `smol-toml` (installed by `npm install`).

## Detector backends

The ML peer picks a backend at build/run time:

1. **`cuda-yolov8-trt`** — real YOLOv8 via TensorRT. Build with
   `-DVCD_WITH_TENSORRT=ON` and set `[ml].model_path` in `topology.toml` to a
   serialized engine. Export one with Ultralytics:
   ```bash
   pip install ultralytics
   yolo export model=yolov8n.pt format=engine half=True imgsz=640
   # → yolov8n.engine ; put its path in topology.toml [ml].model_path
   ```
2. **`cuda-saliency`** — no model needed; a CUDA edge-energy kernel produces
   boxes so the whole GPU path runs out of the box.
3. **`cpu-saliency`** — same detector on the CPU when no GPU is present.

Boxes are drawn into the frame by the C++ peer (`overlay.hpp`); the browser adds
crisp class/score labels from the detection sideband.

## No OpenCV

Decoding is done by `ffmpeg` (raw `rgb24` frames piped into Python); all pixel
work (overlay, RGB→RGBA) is hand-rolled. The only image library in the stack is
the GPU detector.

## Files

```
topology.toml            one config consumed by all four processes
launch.sh                build + run + teardown
common/                  cross-language SHM ring + wire payload definitions
router/router_main.cpp   process 0
video_source/            process 1 (Python)
ml_inference/            process 2 (C++/CUDA)
dashboard/               process 3 (Node + browser UI)
```
