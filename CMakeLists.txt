# set cmake_install_prefix path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# check if building type is not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# bsp
if(NOT BSP_MSP_DIR)
    set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/axpi_pro_bsp_sdk/msp/out)
endif()

message(STATUS "BSP_MSP_DIR = ${BSP_MSP_DIR}")

# set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake)
cmake_minimum_required(VERSION 3.0)
project(CLIP-ONNX-AX650-CPP)

# 添加 FLAGS 检查代码是否有明显 bug
# include(cmake/overlook.cmake)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(NOT OpenCV_DIR)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/third_party/libopencv-4.6-aarch64-none/lib/cmake/opencv4)
endif()

message(STATUS "OPENCV_DIR Path: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src/Runner/)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${BSP_MSP_DIR}/include)
link_directories(${BSP_MSP_DIR}/lib)

option(BUILD_WITH_AX650 "build with ax650" OFF)
option(BUILD_WITH_AX620E "build with ax620e" OFF)
option(BUILD_VIDEO_DEMO "build video demo" OFF)

if(BUILD_WITH_AX620E)
    set(BUILD_WITH_AX650 ON)
    add_compile_definitions(AXERA_TARGET_CHIP_AX620E)
endif()


if(BUILD_WITH_AX650)
    set(AXERA_TARGET_CHIP "AX650")
    add_compile_definitions(BUILD_WITH_AX650)
    add_compile_definitions(AXERA_TARGET_CHIP_AX650)
endif()

message(STATUS "BUILD_WITH_AX650 : ${BUILD_WITH_AX650}")
message(STATUS "BUILD_WITH_AX620E : ${BUILD_WITH_AX620E}")
message(STATUS "BUILD_VIDEO_DEMO : ${BUILD_VIDEO_DEMO}")


add_executable(main 
                src/main.cpp 
                src/Runner/ax_model_runner_ax650.cpp)


target_link_libraries(main ${OpenCV_LIBS})

if(BUILD_WITH_AX650)
    target_link_libraries(main ax_engine ax_interpreter ax_sys)
endif()

install(TARGETS main DESTINATION bin)
