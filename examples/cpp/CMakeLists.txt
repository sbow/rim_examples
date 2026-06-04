set(RIM_INCLUDE_DIR "${rim_SOURCE_DIR}/ipc/src")

function(add_rim_example target source_file)
  add_executable(${target} ${source_file})
  target_compile_features(${target} PRIVATE cxx_std_20)
  target_include_directories(${target} PRIVATE ${RIM_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE pthread rt)
endfunction()

add_rim_example(rim_point_to_point_uds point_to_point_uds.cpp)
add_rim_example(rim_router_fanout_uds router_fanout_uds.cpp)
add_rim_example(rim_router_topic_subscription_uds router_topic_subscription_uds.cpp)
add_rim_example(rim_mixed_transport_demo mixed_transport_demo.cpp)
