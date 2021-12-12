cmake_minimum_required(VERSION 3.7)
project(ustc_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers -g -O2")
#set(CMAKE_CUDA_FLAGS " -g --default-stream per-thread -O2")
set(CMAKE_CUDA_FLAGS " -g  -O2")
#set_source_files_properties(lib/aes_faster_c/aes.c lib/aes_faster_c/wrapper.c lib/md5.c PROPERTIES LANGUAGE CXX )

include_directories(./cuda/include)

set(SOURCE_FILES graph.cpp pb/onnx.proto3.pb.cc pb/model_file.proto3.pb.cc log.cpp helper.cpp engine.cpp profiler.cpp cuda_test.cu visualizer.cpp trans.cpp misc.cpp optimizer.cpp)

add_library(dnn ${SOURCE_FILES})
target_link_libraries(dnn protobuf pthread cudnn cublas)
set_target_properties(dnn PROPERTIES CUDA_ARCHITECTURES "70")

add_executable(a.out main.cpp)
target_link_libraries(a.out protobuf pthread cudnn cublas dnn)

add_executable(b.out test.cpp)
target_link_libraries(b.out protobuf pthread cudnn cublas dnn)
#set_target_properties(a.out PROPERTIES CUDA_ARCHITECTURES "70")
