cmake_minimum_required(VERSION 3.18)
project(hotswap_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT_DIR "$ENV{HOME}/work/libs/onnxruntime-osx-arm64-1.29.0")

add_executable(hotswap
    src/main.cpp
)

# Use target_include_directories and add both ONNX root and sub-include paths
target_include_directories(hotswap PRIVATE
    ${ONNXRUNTIME_ROOT_DIR}/include
    ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime
    ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime/core/session
    include
)

target_link_directories(hotswap PRIVATE
    ${RFDETR_ROOT}/build/src
    ${ONNXRUNTIME_ROOT_DIR}/lib
)

target_link_libraries(hotswap PRIVATE
    onnxruntime
)