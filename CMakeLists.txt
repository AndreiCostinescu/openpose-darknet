cmake_minimum_required(VERSION 3.17)
project(openpose-darknet)

set(CMAKE_CXX_STANDARD 11)

set(DARKNET_DIR "/home/andrei/CLionProjects/darknet-andrei")
include_directories("${DARKNET_DIR}/include")
set(DARKNET_LIB "${DARKNET_DIR}/cmake-build-release/libdarknet.so")

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "Adding OpenposeDarknet")
add_executable(OpenposeDarknet src/main_modif.cpp src/run_darknet.c src/run_darknet.h)
target_link_libraries(OpenposeDarknet ${DARKNET_LIB} ${realsense2_LIBRARY} ${OpenCV_LIBS})
target_compile_definitions(OpenposeDarknet PRIVATE -DOPENCV)
target_compile_definitions(OpenposeDarknet PUBLIC -DOPENCV)

message(STATUS "Adding OpenposeDarknetOriginal")
add_executable(OpenposeDarknetOriginal src/main.cpp src/run_darknet.c src/run_darknet.h)
target_link_libraries(OpenposeDarknetOriginal ${DARKNET_LIB} ${realsense2_LIBRARY} ${OpenCV_LIBS})
target_compile_definitions(OpenposeDarknetOriginal PRIVATE -DOPENCV)
target_compile_definitions(OpenposeDarknetOriginal PUBLIC -DOPENCV)