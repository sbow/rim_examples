cmake_minimum_required(VERSION 3.18)
project(video_classifier_dashboard LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV 4 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Router (Process 0)
add_executable(router router/main.cpp)
target_include_directories(router PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(router PRIVATE ${OpenCV_LIBS} rt pthread)

# YOLO Classifier (Process 2)
add_executable(yolo_classifier
    yolo_classifier/main.cpp
    yolo_classifier/inference.cu
)
target_include_directories(yolo_classifier PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(yolo_classifier PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    rt
    pthread
)
set_target_properties(yolo_classifier PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
