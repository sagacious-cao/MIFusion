cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME TensorRT_MaskRCNN)
project(${PROJECT_NAME})

add_compile_options(-std=c++11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/cmake")
# find_package(Glog REQUIRED)
# find_package(GFlags REQUIRED)
# find_package(Boost REQUIRED COMPONENTS date_time system thread)
set(OpenCV_DIR "/usr/share/OpenCV/")
find_package(OpenCV 3 REQUIRED)
find_package(CUDA QUIET REQUIRED)

# file(GLOB srcs "/home/sagacious/tensorrt/samples/common/*.cpp")

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS} 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/../thirdparty/TensorRT-8.2.2.1/include
    # /home/ubuntu/data/caoruizhi/git/tandem/tandem/thirdparty/TensorRT-8.2.2.1/include
    # "/home/sagacious/tensorrt/samples/common"
    
    )
 
set(CUDA_ARCH_BIN "30 35 50 52 61 62 72" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              
 
SET(CMAKE_BUILD_TYPE DEBUG)
 
# #设置编译器版本
set(CMAKE_CXX_STANDARD 14)
 

add_library(maskrcnn-tensorrt
    ${PROJECT_SOURCE_DIR}/src/logger.cpp
    ${PROJECT_SOURCE_DIR}/src/maskrcnn.cpp
)

target_link_libraries(maskrcnn-tensorrt
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/../thirdparty/TensorRT-8.2.2.1/lib/libnvinfer.so
    ${PROJECT_SOURCE_DIR}/../thirdparty/TensorRT-8.2.2.1/lib/libnvinfer_plugin.so
    ${PROJECT_SOURCE_DIR}/../thirdparty/TensorRT-8.2.2.1/lib/libnvparsers.so
    ${PROJECT_SOURCE_DIR}/../thirdparty/TensorRT-8.2.2.1/lib/libnvonnxparser.so
    
    # /home/ubuntu/data/caoruizhi/git/tandem/tandem/thirdparty/TensorRT-8.2.2.1/lib/libnvinfer.so
    # "/home/ubuntu/data/caoruizhi/git/tandem/tandem/thirdparty/TensorRT-8.2.2.1/lib/libnvinfer_plugin.so"
    # "/home/ubuntu/data/caoruizhi/git/tandem/tandem/thirdparty/TensorRT-8.2.2.1/lib/libnvparsers.so"
    # "/home/ubuntu/data/caoruizhi/git/tandem/tandem/thirdparty/TensorRT-8.2.2.1/lib/libnvonnxparser.so"    
)


# #设置要生成可执行程序的文件名
# add_executable(tensorrtMaskrcnn
#             ${PROJECT_SOURCE_DIR}/src/sampleUffMaskRCNN.cpp
#                                   )
 
# #设置链接库
# target_link_libraries(
#     tensorrtMaskrcnn 
#     maskrcnn-tensorrt
#     #"/usr/lib/x86_64-linux-gnu/libnvinfer.so"
#     #"/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so"
#     #"/usr/lib/x86_64-linux-gnu/libnvparsers.so"
#     #"/usr/lib/x86_64-linux-gnu/libnvonnxparser.so"
#     )
