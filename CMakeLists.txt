cmake_minimum_required(VERSION 3.28)
project(cpp_demo)

set(CMAKE_CXX_STANDARD 20)

set(ONNXRUNTIME_ROOT "/Users/lihongji/dev/dependency/onnxruntime")

include_directories("${ONNXRUNTIME_ROOT}/include")

add_executable(cpp_demo ./src/main.cpp
        src/utils.cpp)

target_link_libraries(cpp_demo "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
