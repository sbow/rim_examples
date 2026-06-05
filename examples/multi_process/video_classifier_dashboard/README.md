# Video Classifier Dashboard

> A four-process, three-language video pipeline wired together by the
> header-only **RoboticsIpcModule (RIM)** — and a single shared `topology.toml`.

Stream a **YouTube** video, classify it with **YOLO on CUDA**, overlay bounding
boxes, and watch the annotated result in a **live browser dashboard**. Python,
C++/CUDA, and Node.js processes all talk to each other through RIM's message
fabric without a single byte of glue code between them.

<p align="center">
  <img src="docs/dashboard.png" width="720"
       alt="The dashboard: a lofi-stream frame with YOLO bounding boxes (person 83%, chair, laptop, cup…) plus live fps, detection count, GPU inference time, end-to-end latency, and system-load panels.">
</p>

**Why it's worth a look:**

- **Zero-copy by design** — megabytes of pixels move through shared memory; only
  64-byte notifications cross the router.
- **One config, four processes, three languages** — `topology.toml` is the single
  source of truth, consumed identically by C++, Python, and Node.
- **Runs with or without a GPU** — the build auto-detects CUDA and falls back to a
  pure-CPU detector, so the demo works on any Linux box.
- **No heavyweight deps** — no OpenCV, no ROS; `ffmpeg` decodes and the rest is
  hand-rolled.

---

## Try it in 30 seconds

```bash
cd examples/multi_process/video_classifier_dashboard
./launch.sh                                   # default lofi-girl live stream
# ...or bring your own source:
./launch.sh "https://www.youtube.com/watch?v=<id>"
```

Then open **<http://localhost:8080>**. Press **`Ctrl-C`** to stop everything —
the launcher tears down all `/dev/shm/rim_vcd_*` regions and `/tmp/rim_vcd_*.sock`
sockets on the way out.

`launch.sh` does all the setup for you: builds the C++ peers with CMake, creates a
Python venv for `yt-dlp`, runs `npm install` for the dashboard, then starts all
four processes and wires them through `topology.toml`.

> No GPU? It just works — the build drops to the CPU detector automatically.
> Want to force it? `VCD_FORCE_CPU=1 ./launch.sh`.

---

## Choosing a video source

The publisher resolves the stream with `yt-dlp`, so anything `yt-dlp` can play
works: YouTube live streams, normal videos, and most other sites it supports. A
**public, always-on live stream** is the nicest default (it never "ends").

**Check a URL before launching** — list its formats with the venv's `yt-dlp`:

```bash
.venv/bin/yt-dlp -F "https://www.youtube.com/watch?v=<id>"
```

What to look for in the output:

- **At least one row appears** → the video is reachable and playable. Example:
  ```
  ID EXT RESOLUTION FPS │   TBR PROTO │ VCODEC      ACODEC
  93 mp4 640x360     30 │  962k m3u8  │ avc1.4D401E mp4a.40.2
  95 mp4 1280x720    30 │ 2448k m3u8  │ avc1.4D401F mp4a.40.2
  ```
- A row at/below your `[video].yt_format` cap (default `best[height<=720]`) so the
  selector resolves. The frame is rescaled to `[video].width`×`height` anyway, so a
  360p–720p rendition is plenty.
- **Errors instead of a table** mean it won't work as a source:
  - `No video formats found!` → usually a stale `yt-dlp`; update it (see
    [Troubleshooting](#troubleshooting)).
  - `Sign in to confirm…`, `members-only`, `Video unavailable`, region blocks →
    the video needs authentication or isn't public; pick another, or pass cookies
    via `yt-dlp`'s `--cookies-from-browser`.

**Pin an exact rendition** if you like — set `[video].yt_format` in `topology.toml`
to a format ID from the table (e.g. `yt_format = "93"` for the 640×360 stream),
or keep the `best[height<=720]` selector.

> Heads-up: recent `yt-dlp` prints `No supported JavaScript runtime could be
> found … some formats may be missing`. Extraction still works, but installing
> **Deno** silences it and can expose more formats:
> `curl -fsSL https://deno.land/install.sh | sh` (then put `~/.deno/bin` on `PATH`).

### Frame rate, resolution & smooth playback

Resolution and frame rate are set in **`topology.toml`** under `[video]` — this is
the single source of truth all peers read (the publisher only falls back to these
values if the keys are missing):

```toml
[video]
width  = 640    # frame width  (publisher rescales the source to this)
height = 360    # frame height
fps    = 15     # frames per second pulled from the source
```

Lower `fps` / `width` / `height` for less CPU/GPU load and lower bandwidth; raise
them for a crisper feed. Remember the SHM slot is sized from these — `width × height
× 3` bytes per frame — so the peers stay byte-compatible automatically.

> **Why playback is smooth (the `-re` flag).** HLS streams arrive as multi-second
> segments. Without pacing, ffmpeg races through each freshly-downloaded segment
> *faster than real time*, then freezes while fetching the next one — a "chunky"
> sprint/stall cycle. The publisher passes ffmpeg **`-re`** (in
> `video_source/publisher.py`, `start_ffmpeg`) to emit frames at the input's native
> rate, which both smooths delivery and lets the next segment prefetch in the
> background. If a particular source is still choppy, lower `[video].fps` or try a
> source with shorter HLS segments.

---

## How it works

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

**The key idea — and the RIM pattern this demonstrates: the pixels never touch
the router.** It forwards only 64-byte `RouterFrame` *notifications*. Each one
carries a sideband descriptor (`sideband_seq`, `sideband_len`, `sideband_idx`)
that points the consumer at the right slot in a shared-memory ring, declared as a
`[[peers.sideband]]` in the topology. That's the **control-plane / bulk-data
split** from RIM's ADR 0005 / ADR 0008 — tiny messages for coordination, shared
memory for the heavy payload.

`common/frame_shm.{hpp,py,js}` is the same lock-free, "newest-frame-wins" SHM ring
ported to all three languages, so they interoperate **byte-for-byte**.

### What each process demonstrates

| Process | Language | RIM feature exercised |
|---|---|---|
| `router/` | C++ | `topology_loader` + `MixedRouterServer` (UDS + UDP in one hop, Phase H) |
| `video_source/` | Python | the ctypes UDS bridge + a sideband SHM producer |
| `ml_inference/` | C++/CUDA | `IpcEndpoint<Uds>` peer, RouterFrame v2, **two** sideband regions per peer |
| `dashboard/` | Node.js | the UDP bridge + SHM sideband consumer → WebSocket |

---

## Requirements

External dependencies are allowed here (unlike RIM core):

| Category | Needs |
|---|---|
| **System** | `cmake` ≥ 3.20, a C++20 compiler, `ffmpeg`, `node` ≥ 18 **and `npm`** (separate packages on Debian/Ubuntu), `python3` ≥ 3.10 |
| **GPU** *(optional)* | CUDA toolkit. Without it the build uses the CPU detector so the pipeline still runs. |
| **Python** | `yt-dlp` (installed into a local venv by the launcher) |
| **Node** | `ws`, `smol-toml` (installed by `npm install`) |

---

## Building manually

`launch.sh` builds for you, but you can drive CMake directly — useful when you
want to enable TensorRT, force the CPU path, or target a specific GPU arch.

```bash
cd examples/multi_process/video_classifier_dashboard

# default build: auto-detects CUDA, falls back to the CPU detector if none
cmake -S . -B build -DRIM_DIR=../../../../RoboticsIpcModule -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# → build/rim_vcd_router  and  build/rim_vcd_ml
```

### Build flags

Supply these as `-D<FLAG>=<VALUE>` on the `cmake -S . -B build` configure line:

| Flag | Default | Effect |
|---|---|---|
| `-DRIM_DIR=<path>` | fetched from GitHub | Use a local RoboticsIpcModule checkout instead of `FetchContent`. Point it at your RIM repo (the sibling `../../../../RoboticsIpcModule` if you cloned both side by side). |
| `-DCMAKE_BUILD_TYPE=Release` | unset | Optimized build. Recommended for the GPU path. |
| `-DVCD_WITH_TENSORRT=ON` | `OFF` | Build the real **YOLOv8 TensorRT** detector (`yolo_trt.cu`). Requires CUDA **and** the TensorRT dev libraries (links `nvinfer` + `cudart`). Set `[ml].model_path` in `topology.toml` to a serialized `.engine`. |
| `-DVCD_FORCE_CPU=ON` | `OFF` | Skip CUDA entirely and build the CPU-only saliency detector, even on a machine that has a GPU. |
| `-DCMAKE_CUDA_ARCHITECTURES=<arch>` | `native` (CMake ≥ 3.24) | Target GPU compute capability. E.g. `86` for an RTX 3060, `87` for Jetson Orin, `75` for Turing. Set this if `native` detection fails or you cross-compile. |

> The env var `VCD_FORCE_CPU=1` understood by `launch.sh` simply maps to
> `-DVCD_FORCE_CPU=ON` on the configure line above.

### Worked examples

```bash
# Real YOLOv8 + TensorRT on an RTX 3060 (compute capability 8.6)
cmake -S . -B build -DRIM_DIR=../../../../RoboticsIpcModule \
      -DCMAKE_BUILD_TYPE=Release \
      -DVCD_WITH_TENSORRT=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j

# GPU build without TensorRT (CUDA saliency kernel — no model needed)
cmake -S . -B build -DRIM_DIR=../../../../RoboticsIpcModule -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CPU-only, force the fallback detector
cmake -S . -B build -DRIM_DIR=../../../../RoboticsIpcModule -DVCD_FORCE_CPU=ON
cmake --build build -j
```

If you change a flag, re-run the `cmake -S . -B build …` configure step (CMake
caches options in `build/CMakeCache.txt`); for a clean slate, `rm -rf build` first.

---

## Detector backends

The ML peer picks a backend at build/run time:

1. **`cuda-yolov8-trt`** — real YOLOv8 via TensorRT. Build with
   `-DVCD_WITH_TENSORRT=ON` (see [Building manually](#building-manually)) and set
   `[ml].model_path` in `topology.toml` to a serialized engine. Export one with
   Ultralytics:
   ```bash
   pip install ultralytics
   yolo export model=yolov8n.pt format=engine half=True imgsz=640
   # → yolov8n.engine ; put its path in topology.toml [ml].model_path
   ```
   (If the FP16 export fails, see [Troubleshooting](#troubleshooting) below.)
2. **`cuda-saliency`** — no model needed; a CUDA edge-energy kernel produces boxes,
   so the whole GPU path runs out of the box.
3. **`cpu-saliency`** — the same detector on the CPU when no GPU is present.

Boxes are drawn into the frame by the C++ peer (`overlay.hpp`); the browser adds
crisp class/score labels from the detection sideband.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `BuilderFlag … has no attribute 'FP16'` during `yolo export … format=engine` | **TensorRT 10+/11 removed the `FP16` builder flag** (precision is now set via strongly-typed networks). The Ultralytics version installed still calls the old API when `half=True`. | Export without FP16: `yolo export model=yolov8n.pt format=engine imgsz=640` (FP32 engine). See the [precision options](#tensorrt-fp16-export-fails) below for FP16 alternatives. |
| `Command 'ffmpeg' not found` | `ffmpeg` isn't installed. | `sudo apt install ffmpeg` |
| `npm: command not found` (node works fine) | On Debian/Ubuntu `npm` ships in a **separate package** from `node`. | `sudo apt install npm` |
| `yt-dlp … No video formats found!` / `This live stream … not available` | The video source's `yt-dlp` is **out of date** (YouTube changes often), or the stream ended / is restricted. | `.venv/bin/pip install -U yt-dlp` (or `… -U --pre 'yt-dlp[default]'` for nightly); or pick another public URL: `./launch.sh <youtube_url>`. The launcher now auto-updates yt-dlp each run when online. |
| Playback sprints faster-than-realtime, then freezes, repeatedly | HLS segment pacing — ffmpeg bursts through a segment, then stalls fetching the next. | The publisher already passes ffmpeg `-re` to pace at native rate (see [Frame rate, resolution & smooth playback](#frame-rate-resolution--smooth-playback)). If still choppy, lower `[video].fps` or try a shorter-segment source. |
| Build prints `no CUDA compiler found — using CPU fallback` | No CUDA toolkit / `nvcc` not on `PATH`. | Expected on CPU-only machines. Install the CUDA toolkit and ensure `nvcc` is on `PATH` for the GPU path, or accept the CPU detector. |
| Engine fails to deserialize at runtime (TensorRT version error) | The `.engine` was built with a **different TensorRT major version** than the one the C++ peer links against. | Rebuild the engine with the same TensorRT version you build/link `rim_vcd_ml` with. |
| Dashboard won't bind / port already in use | Something else is on `8080`. | `VCD_HTTP_PORT=9090 ./launch.sh` and open that port instead. |
| Pipeline stalls after a crash; nothing updates | Stale shared-memory regions or sockets from a previous run. | `rm -f /dev/shm/rim_vcd_* /tmp/rim_vcd_*.sock` then relaunch. (The launcher normally cleans these up on exit.) |

### TensorRT FP16 export fails

The most common snag. Newer TensorRT (10/11) dropped the explicit `FP16`/`INT8`
builder flags, so Ultralytics' `half=True` export path raises
`AttributeError: … BuilderFlag … has no attribute 'FP16'`. Your options, easiest
first:

1. **Export FP32** — drop `half=True`. Works today, costs some throughput:
   ```bash
   yolo export model=yolov8n.pt format=engine imgsz=640
   ```
2. **Use the ONNX you already have** — the same `yolo export … format=onnx` (or the
   ONNX emitted during the engine export) runs on the GPU via `onnxruntime-gpu`,
   including a TensorRT execution provider that gives near-engine speed and FP16
   without the broken builder-flag path.
3. **Match Ultralytics to your TensorRT** — use an Ultralytics release that supports
   your installed TensorRT's strongly-typed API (check their changelog), then retry
   with `half=True`.

For YOLOv8n on a desktop GPU, the FP32-vs-FP16 difference is usually absorbed by
the video framerate ceiling — start with option 1 to get running, optimize later.

---

## Design notes

**No OpenCV.** Decoding is done by `ffmpeg` (raw `rgb24` frames piped into Python);
all pixel work (overlay, RGB→RGBA) is hand-rolled. The only image library anywhere
in the stack is the GPU detector itself.

---

## Project layout

```
topology.toml            one config consumed by all four processes
launch.sh                build + run + teardown
common/                  cross-language SHM ring + wire payload definitions
router/router_main.cpp   process 0 — C++ router
video_source/            process 1 — Python video publisher
ml_inference/            process 2 — C++/CUDA YOLO/saliency detector
dashboard/               process 3 — Node.js server + browser UI
```
