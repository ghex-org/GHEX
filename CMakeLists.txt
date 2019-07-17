cmake_minimum_required(VERSION 3.12.4)

cmake_policy(SET CMP0048 NEW)

project(GHEX VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(MPI REQUIRED)
find_package(GridTools REQUIRED)
find_package(Boost COMPONENTS mpi REQUIRED)

message(">>>>>>>>>>>>>>>>${GridTools_MODULE_PATH}")
list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

add_library(GHEX_libs INTERFACE)
target_include_directories(GHEX_libs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
target_link_libraries(GHEX_libs INTERFACE GridTools::gridtools)

enable_testing()

set(GTEST_DIR ${GridTools_DIR}/_deps/googletest-src)

find_path(GTEST_INC_DIR
    "gtest/gtest.h"
    HINTS
    ${GTEST_DIR}/include
    ${GTEST_DIR}/googletest/include)
if(NOT GTEST_INC_DIR)
    message(FATAL_ERROR "Couldn't find gtest/gtest.h under folder GTEST_DIR = " ${GTEST_DIR})
endif()

find_path(GTEST_SOURCE_DIR
    gtest-all.cc
    HINTS
    ${GTEST_DIR}/src
    ${GTEST_DIR}/googletest/src
)
if(NOT GTEST_SOURCE_DIR)
    message(FATAL_ERROR "Couldn't find gtest-all.cc under folder GTEST_DIR = " ${GTEST_DIR})
endif()

set(GTEST_LIB_SOURCES ${GTEST_SOURCE_DIR}/gtest-all.cc)
set(GTEST_INCLUDE_DIRS ${GTEST_INC_DIR} ${GTEST_SOURCE_DIR} ${GTEST_SOURCE_DIR}/..)

add_subdirectory(tests)
add_subdirectory(benchmarks)
