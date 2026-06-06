cmake_minimum_required(VERSION 3.20)
project(rim_vcd LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Locate the header-only RoboticsIpcModule. Prefer a local checkout (set
# -DRIM_DIR=/path/to/RoboticsIpcModule); otherwise fetch it.
# ----------------------------------------------------------------------------
set(RIM_DIR "" CACHE PATH "Path to a local RoboticsIpcModule checkout")
if(RIM_DIR AND EXISTS "${RIM_DIR}/ipc/src")
  set(RIM_SRC "${RIM_DIR}")
  message(STATUS "rim_vcd: using local RIM at ${RIM_SRC}")
else()
  include(FetchContent)
  FetchContent_Declare(rim
    GIT_REPOSITORY https://github.com/sbow/RoboticsIpcModule.git
    GIT_TAG main GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(rim)
  set(RIM_SRC "${rim_SOURCE_DIR}")
endif()

set(RIM_INC "${RIM_SRC}/ipc/src" "${RIM_SRC}/third_party/tomlplusplus")

# ----------------------------------------------------------------------------
# CUDA is the intended backend; the CPU fallback keeps the demo runnable on
# machines without a GPU. -DVCD_WITH_TENSORRT=ON adds the YOLOv8 TRT detector.
# ----------------------------------------------------------------------------
option(VCD_FORCE_CPU "Force the CPU fallback detector" OFF)
option(VCD_WITH_TENSORRT "Build the YOLOv8 TensorRT backend" OFF)

include(CheckLanguage)
set(VCD_HAVE_CUDA OFF)
if(NOT VCD_FORCE_CPU)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(VCD_HAVE_CUDA ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
      else()
        set(CMAKE_CUDA_ARCHITECTURES 72 75 86 87)  # Jetson Xavier/Orin + desktop
      endif()
    endif()
    message(STATUS "rim_vcd: CUDA backend enabled (arch ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(STATUS "rim_vcd: no CUDA compiler found — using CPU fallback")
  endif()
endif()

# ----------------------------------------------------------------------------
# Locate TensorRT (only when VCD_WITH_TENSORRT=ON). On Jetson/JetPack and apt
# installs the SDK lives under /usr and is found automatically. For a tarball
# install pass -DTENSORRT_DIR=/opt/TensorRT-x.y.z (the dir containing
# include/NvInfer.h and lib/libnvinfer.so), or export TENSORRT_DIR.
# ----------------------------------------------------------------------------
if(VCD_WITH_TENSORRT)
  if(NOT VCD_HAVE_CUDA)
    message(FATAL_ERROR "VCD_WITH_TENSORRT=ON needs CUDA — do not combine it with VCD_FORCE_CPU")
  endif()
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR} ENV TENSORRT_DIR
    PATH_SUFFIXES include
    PATHS /usr/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu)
  find_library(TENSORRT_NVINFER NAMES nvinfer
    HINTS ${TENSORRT_DIR} ENV TENSORRT_DIR
    PATH_SUFFIXES lib lib64 targets/x86_64-linux-gnu/lib lib/x86_64-linux-gnu lib/aarch64-linux-gnu
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER)
    message(FATAL_ERROR
      "VCD_WITH_TENSORRT=ON but the TensorRT C++ SDK was not found.\n"
      "  NvInfer.h    : ${TENSORRT_INCLUDE_DIR}\n"
      "  libnvinfer   : ${TENSORRT_NVINFER}\n"
      "Install TensorRT and/or pass -DTENSORRT_DIR=/path/to/TensorRT (the dir "
      "holding include/NvInfer.h and lib/libnvinfer.so). On Jetson/JetPack it is "
      "preinstalled under /usr. NOTE: the pip 'tensorrt' wheels are runtime-only "
      "(no headers) and cannot satisfy a C++ build.")
  endif()
  message(STATUS "rim_vcd: TensorRT headers=${TENSORRT_INCLUDE_DIR} lib=${TENSORRT_NVINFER}")
endif()

# ----------------------------------------------------------------------------
# Process 0 — router
# ----------------------------------------------------------------------------
add_executable(rim_vcd_router router/router_main.cpp)
target_compile_features(rim_vcd_router PRIVATE cxx_std_20)
target_include_directories(rim_vcd_router PRIVATE ${RIM_INC})
target_link_libraries(rim_vcd_router PRIVATE pthread rt)

# ----------------------------------------------------------------------------
# Process 2 — ML inference peer
# ----------------------------------------------------------------------------
set(ML_SOURCES ml_inference/ml_main.cpp)
if(VCD_HAVE_CUDA)
  list(APPEND ML_SOURCES ml_inference/pipeline_cuda.cu)
  if(VCD_WITH_TENSORRT)
    list(APPEND ML_SOURCES ml_inference/yolo_trt.cu)
  endif()
else()
  list(APPEND ML_SOURCES ml_inference/pipeline_cpu.cpp)
endif()

add_executable(rim_vcd_ml ${ML_SOURCES})
target_compile_features(rim_vcd_ml PRIVATE cxx_std_20)
target_include_directories(rim_vcd_ml PRIVATE ${RIM_INC} common ml_inference)
target_link_libraries(rim_vcd_ml PRIVATE pthread rt)

if(VCD_HAVE_CUDA)
  set_target_properties(rim_vcd_ml PROPERTIES CUDA_STANDARD 17)
  target_compile_definitions(rim_vcd_ml PRIVATE VCD_HAVE_CUDA)
  if(VCD_WITH_TENSORRT)
    target_compile_definitions(rim_vcd_ml PRIVATE VCD_WITH_TENSORRT)
    target_include_directories(rim_vcd_ml PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(rim_vcd_ml PRIVATE ${TENSORRT_NVINFER} cudart)
  endif()
endif()
