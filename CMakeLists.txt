cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project (myproject)
enable_language(CUDA)
include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

aux_source_directory(. DIR_SRCS)

add_subdirectory(src)
 
add_library(gpu_conv_lib STATIC src/layer/gpu-new-forward-basic.cu src/layer/gpu-new-forward.h src/layer/gpu-utils.cu src/layer/gpu-utils.h src/layer/gpu-new-forward-optimize.cu)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(demo ${DIR_SRCS})

target_link_libraries(demo gpu_conv_lib MiniDNNLib ${CUDA_LIBRARIES})
