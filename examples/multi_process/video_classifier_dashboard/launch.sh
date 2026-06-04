#!/usr/bin/env bash
# ============================================================================
# launch.sh — bring up the whole video_classifier_dashboard pipeline.
#
#   ./launch.sh [YOUTUBE_URL]
#
# Starts four processes (router, ML inference, dashboard, video source), wires
# them through the shared topology.toml, and tears everything down cleanly on
# Ctrl-C. Open http://localhost:8080 once it is up.
#
# Env overrides:
#   RIM_DIR        path to a RoboticsIpcModule checkout (default: sibling repo)
#   VCD_HTTP_PORT  dashboard port (default 8080)
#   VCD_FORCE_CPU  set to 1 to skip CUDA and build the CPU detector
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

RIM_DIR="${RIM_DIR:-$(cd "$HERE/../../../../RoboticsIpcModule" 2>/dev/null && pwd || true)}"
export RIM_DIR
TOPO="$HERE/topology.toml"
YOUTUBE_URL="${1:-}"

log() { printf '\033[1;36m[launch]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[launch] %s\033[0m\n' "$*" >&2; exit 1; }

command -v cmake  >/dev/null || die "cmake not found"
command -v node   >/dev/null || die "node not found"
command -v python3>/dev/null || die "python3 not found"
command -v ffmpeg >/dev/null || die "ffmpeg not found (apt install ffmpeg)"
[ -d "${RIM_DIR:-}" ] || die "RoboticsIpcModule not found; set RIM_DIR"

# --- cleanup any leftovers from a previous (crashed) run ---------------------
cleanup_shm() { rm -f /dev/shm/rim_vcd_* /tmp/rim_vcd_*.sock 2>/dev/null || true; }
cleanup_shm

# --- 1. build the C++ peers --------------------------------------------------
CMAKE_ARGS=(-S "$HERE" -B "$HERE/build" -DRIM_DIR="$RIM_DIR" -DCMAKE_BUILD_TYPE=Release)
[ "${VCD_FORCE_CPU:-0}" = "1" ] && CMAKE_ARGS+=(-DVCD_FORCE_CPU=ON)
log "configuring + building C++ (router, ml)…"
cmake "${CMAKE_ARGS[@]}" >/dev/null
cmake --build "$HERE/build" -j >/dev/null
log "C++ build done"

# --- 2. python venv for the publisher ---------------------------------------
if [ ! -d "$HERE/.venv" ]; then
  log "creating python venv + installing yt-dlp…"
  python3 -m venv "$HERE/.venv"
  "$HERE/.venv/bin/pip" -q install --upgrade pip >/dev/null
  "$HERE/.venv/bin/pip" -q install -r "$HERE/video_source/requirements.txt"
fi
PY="$HERE/.venv/bin/python"

# --- 3. node deps for the dashboard -----------------------------------------
if [ ! -d "$HERE/dashboard/node_modules" ]; then
  log "installing dashboard npm deps…"
  (cd "$HERE/dashboard" && npm install --silent --no-audit --no-fund)
fi

# --- 4. launch all four processes -------------------------------------------
PIDS=()
shutdown() {
  log "stopping…"
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  cleanup_shm
  log "bye"
}
trap shutdown INT TERM EXIT

log "starting router"
"$HERE/build/rim_vcd_router" "$TOPO" & PIDS+=($!)
sleep 0.4

log "starting dashboard  ->  http://localhost:${VCD_HTTP_PORT:-8080}"
(cd "$HERE/dashboard" && node server.js "$TOPO") & PIDS+=($!)
sleep 0.3

log "starting ml inference"
"$HERE/build/rim_vcd_ml" "$TOPO" & PIDS+=($!)
sleep 0.3

log "starting video source (Python SHM publisher)"
if [ -n "$YOUTUBE_URL" ]; then
  "$PY" "$HERE/video_source/publisher.py" --config "$TOPO" --url "$YOUTUBE_URL" & PIDS+=($!)
else
  "$PY" "$HERE/video_source/publisher.py" --config "$TOPO" & PIDS+=($!)
fi

log "running — press Ctrl-C to stop"
wait
