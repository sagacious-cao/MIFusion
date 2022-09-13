#cmake_minimum_required(VERSION 2.8)
cmake_minimum_required(VERSION 3.5)
project(gSLICr)
  
IF(MSVC_IDE)
  set(OpenCV_STATIC OFF)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package (Eigen3 REQUIRED)
include_directories( "/usr/include/eigen3"  ${EIGEN3_INCLUDE_DIR})



include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(ORUtils)

set(GSLICR_LIB
gSLICr_Lib/engines/gSLICr_core_engine.h
gSLICr_Lib/engines/gSLICr_seg_engine.h
gSLICr_Lib/engines/gSLICr_seg_engine_GPU.h
gSLICr_Lib/engines/gSLICr_seg_engine_shared.h
gSLICr_Lib/engines/gSLICr_core_engine.cpp
gSLICr_Lib/engines/gSLICr_seg_engine.cpp
gSLICr_Lib/engines/gSLICr_seg_engine_GPU.cu
gSLICr_Lib/objects/gSLICr_settings.h
gSLICr_Lib/objects/gSLICr_spixel_info.h
gSLICr_Lib/gSLICr_defines.h
gSLICr_Lib/gSLICr.h
gSLICr_Lib/gSLICrTools.cpp
gSLICr_Lib/gSLICrTools.h
gSLICr_Lib/segmentationCuda.cu
gSLICr_Lib/segmentationCuda.h

)

list(APPEND "-std=c++11 -ftree-vectorize")
SOURCE_GROUP(engines FILES ${GSLICR_LIB})

cuda_add_library(gSLICr_lib
			${GSLICR_LIB}
			# NVTimer.h
			OPTIONS -gencode arch=compute_86,code=compute_86)
target_link_libraries(gSLICr_lib ${CUDA_LIBRARY})

# add_executable(demo demo.cpp)
# target_link_libraries(demo gSLICr_lib ${OpenCV_LIBS})
