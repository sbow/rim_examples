#!/bin/bash
#
# Launch script for Video Classifier Dashboard
# Starts all four processes in the correct order.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PIDS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cleanup() {
    echo -e "\n${YELLOW}Shutting down all processes...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null
    echo -e "${GREEN}All processes stopped.${NC}"
}

trap cleanup EXIT INT TERM

echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Video Classifier Dashboard - Launcher      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""

# Check build
if [ ! -f "${BUILD_DIR}/router" ] || [ ! -f "${BUILD_DIR}/yolo_classifier" ]; then
    echo -e "${YELLOW}Building C++ components...${NC}"
    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}"
    cmake --build "${BUILD_DIR}" -j
fi

# Check Node.js dependencies
if [ ! -d "${SCRIPT_DIR}/dashboard/node_modules" ]; then
    echo -e "${YELLOW}Installing dashboard dependencies...${NC}"
    (cd "${SCRIPT_DIR}/dashboard" && npm install)
fi

# Download YOLO model if needed
YOLO_MODEL="${YOLO_MODEL:-yolov8n.onnx}"
if [ ! -f "${SCRIPT_DIR}/${YOLO_MODEL}" ]; then
    echo -e "${YELLOW}Downloading YOLO model...${NC}"
    curl -L -o "${SCRIPT_DIR}/${YOLO_MODEL}" \
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/${YOLO_MODEL}"
fi

echo ""

# Process 0: Router
echo -e "${GREEN}[1/4] Starting Router (Process 0)...${NC}"
"${BUILD_DIR}/router" &
PIDS+=($!)
sleep 1

# Process 1: Video Publisher
echo -e "${GREEN}[2/4] Starting Video Publisher (Process 1)...${NC}"
python3 "${SCRIPT_DIR}/video_publisher/main.py" \
    --source "${VIDEO_SOURCE:-https://www.youtube.com/watch?v=dQw4w9WgXcQ}" \
    --fps "${TARGET_FPS:-30}" &
PIDS+=($!)
sleep 1

# Process 2: YOLO Classifier
echo -e "${GREEN}[3/4] Starting YOLO Classifier (Process 2)...${NC}"
YOLO_MODEL="${SCRIPT_DIR}/${YOLO_MODEL}" "${BUILD_DIR}/yolo_classifier" &
PIDS+=($!)
sleep 1

# Process 3: Dashboard
echo -e "${GREEN}[4/4] Starting Dashboard (Process 3)...${NC}"
(cd "${SCRIPT_DIR}/dashboard" && node server.js) &
PIDS+=($!)

echo ""
echo -e "${GREEN}All processes started!${NC}"
echo -e "Dashboard: ${BLUE}http://localhost:${DASHBOARD_PORT:-3000}${NC}"
echo -e "Press Ctrl+C to stop all processes."
echo ""

# Wait for any process to exit
wait -n "${PIDS[@]}" 2>/dev/null || true
echo -e "${RED}A process exited unexpectedly. Shutting down...${NC}"
