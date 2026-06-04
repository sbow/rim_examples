cmake_minimum_required(VERSION 3.20)
project(rim_examples LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
  rim
  GIT_REPOSITORY https://github.com/sbow/RoboticsIpcModule.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(rim)

add_subdirectory(examples/cpp)
