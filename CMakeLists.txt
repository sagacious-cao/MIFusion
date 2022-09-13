SET(PROJECT_NAME MIFusion)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
#set(CMAKE_VERBOSE_MAKEFILE ON)


set(BUILD_TYPE Release)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# required libraries
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread)
find_package(Torch REQUIRED)

# optional libraries
# find_package(Pangolin 0.2 QUIET)
find_package(Pangolin QUIET)
set(OpenCV_DIR "/usr/share/OpenCV/")
find_package(OpenCV 3 REQUIRED)
find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR})


# libdr
add_subdirectory(libdr/dr_fusion)
add_subdirectory(libdr/dr_mvsnet)
add_subdirectory(libdr/cuda_coarse_tracker)

add_subdirectory(libmaskrcnn)
add_subdirectory(libgSLICr)

# flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "${SSE_FLAGS} -O3 -g -std=c++0x -march=native -msse2 -msse3")

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

# DVO source files
File(GLOB_RECURSE dvo_src_core ${PROJECT_SOURCE_DIR}/src/dvo-core/src/core/*.cpp)

# Comment
# Sources files
set(tandem_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.cpp
        ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptimize.cpp
        ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptPoint.cpp
        ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemDebugStuff.cpp
        ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemMarginalize.cpp
        ${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.cpp
        ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.cpp
        ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.cpp
        ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseRGBDInitializer.cpp
        ${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.cpp
        ${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.cpp
        ${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector2.cpp
        ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
        ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
        ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
        ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
        ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
        ${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
        ${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp
        ${PROJECT_SOURCE_DIR}/src/util/commandline.cpp
        ${PROJECT_SOURCE_DIR}/src/util/memoryManager.cpp
        ${PROJECT_SOURCE_DIR}/src/tandem/tandem_backend.cpp

        
        ${dvo_src_core}
        ${PROJECT_SOURCE_DIR}/src/dvo-core/src/dense_tracking.cpp
        ${PROJECT_SOURCE_DIR}/src/dvo-core/src/dense_tracking_config.cpp
        ${PROJECT_SOURCE_DIR}/src/dvo-core/src/dense_tracking_impl.cpp
        ${PROJECT_SOURCE_DIR}/src/dvo-core/src/util/histogram.cpp
        ${PROJECT_SOURCE_DIR}/src/dvo-core/src/util/id_generator.cpp
        )

include_directories(
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
        ${PROJECT_SOURCE_DIR}/thirdparty/sse2neon
        ${EIGEN3_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/src/dvo-core/include

        ${PROJECT_SOURCE_DIR}/libmaskrcnn/include
        ${PROJECT_SOURCE_DIR}/thirdparty/TensorRT-8.2.2.1/include

        ${PROJECT_SOURCE_DIR}/libgSLICr/ORUtils
        ${PROJECT_SOURCE_DIR}/libgSLICr/gSLICr_Lib/
        ${PROJECT_SOURCE_DIR}/libgSLICr/gSLICr_Lib/engines
        ${PROJECT_SOURCE_DIR}/libgSLICr/gSLICr_Lib/objects
)

cuda_add_library(utils_cuda  ${PROJECT_SOURCE_DIR}/src/tandem/utils_cuda.cu)


# decide if we have pangolin
if (Pangolin_FOUND)
    message("--- found PANGOLIN, compiling tandem_pangolin library.")
    include_directories(${Pangolin_INCLUDE_DIRS})
    set(tandem_pangolin_SOURCE_FILES
            ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
            ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp
            ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/DrFrameDisplay.cpp)

    set(HAS_PANGOLIN 1)
else ()
    message("--- could not find PANGOLIN, not compiling tandem_pangolin library.")
    message("    this means there will be no 3D display / GUI available for tandem_dataset.")
    set(tandem_pangolin_SOURCE_FILES)
    set(HAS_PANGOLIN 0)
endif ()

# openCV
message("--- found OpenCV (${OpenCV_INCLUDE_DIRS}), compiling tandem_opencv library.")
include_directories(${OpenCV_INCLUDE_DIRS})
set(tandem_opencv_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
        ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)
set(HAS_OPENCV 1)


# no ziplib.
set(LIBZIP_LIBRARY "")


# compile main library.
include_directories(${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR})
add_library(tandem ${tandem_SOURCE_FILES} ${tandem_opencv_SOURCE_FILES} ${tandem_pangolin_SOURCE_FILES})
target_link_libraries(tandem tbb)

target_link_libraries(tandem dr-mvsnet)
target_link_libraries(tandem dr-fusion)
target_compile_definitions(tandem PUBLIC MVSNET)
target_link_libraries(tandem cuda_coarse_tracker)
target_link_libraries(tandem maskrcnn-tensorrt)
target_link_libraries(tandem gSLICr_lib)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # OSX
    set(BOOST_THREAD_LIBRARY boost_thread-mt)
else ()
    set(BOOST_THREAD_LIBRARY boost_thread)
endif ()

# build main executable (only if we have both OpenCV and Pangolin)
message("--- compiling mifusion_dataset.")
add_executable(mifusion_dataset ${PROJECT_SOURCE_DIR}/src/mifusion.cpp)
target_link_libraries(mifusion_dataset tandem boost_system cxsparse ${BOOST_THREAD_LIBRARY} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS} utils_cuda)
if (MVSNET)
    target_link_libraries(mifusion_dataset dr-mvsnet)
    target_link_libraries(mifusion_dataset dr-fusion)
    target_compile_definitions(mifusion_dataset PUBLIC MVSNET)
endif ()


