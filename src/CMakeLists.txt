
aux_source_directory(. DIR_LIB_SRCS)
aux_source_directory(./layer DIR_LIB_SRCS)
aux_source_directory(./loss DIR_LIB_SRCS)
aux_source_directory(./optimizer DIR_LIB_SRCS)

add_library (MiniDNNLib ${DIR_LIB_SRCS})


find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})



