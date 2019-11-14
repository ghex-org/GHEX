cmake_minimum_required(VERSION 3.12.4)

cmake_policy(SET CMP0048 NEW)

set(USE_GPU "OFF" CACHE BOOL "use cuda")
set(USE_MPI_WITH_UCX_IN_TESTS "OFF" CACHE BOOL "use ucx for unit test")
set(USE_HYBRID_TESTS "ON" CACHE BOOL "run gpu+cpu tests")

if(USE_GPU)
    project(GHEX VERSION 0.1 LANGUAGES CXX CUDA)
else()
    project(GHEX VERSION 0.1 LANGUAGES CXX)
endif()


set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic") # Last option is because of a problem with GCC9. Try to remove with more releases of the compiler
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(GHEX_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${GHEX_MODULE_PATH}")
message(">>>>>>>>>>>>>>>>${CMAKE_MODULE_PATH}")

find_package(MPI REQUIRED)
find_package(GridTools REQUIRED HINTS ${GridTools_DIR})
find_package(Boost)
find_package(PMIx)
find_package(UCP)

message(">>>>>>>>>>>>>>>>${GridTools_MODULE_PATH}")
list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

add_library(GHEX_libs INTERFACE)
target_include_directories(GHEX_libs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
target_link_libraries(GHEX_libs INTERFACE GridTools::gridtools)

enable_testing()

set(GHEX_GTEST_DIR ${GridTools_DIR}/_deps/googletest-src CACHE FILEPATH "Place where to find googletest installation")

find_path(GTEST_INC_DIR
    "gtest/gtest.h"
    HINTS
    ${GHEX_GTEST_DIR}/include
    ${GHEX_GTEST_DIR}/googletest/include)
if(NOT GTEST_INC_DIR)
    message(FATAL_ERROR "Couldn't find gtest/gtest.h under folder GHEX_GTEST_DIR = " ${GHEX_GTEST_DIR})
endif()

find_path(GTEST_SOURCE_DIR
    gtest-all.cc
    HINTS
    ${GHEX_GTEST_DIR}/src
    ${GHEX_GTEST_DIR}/googletest/src
)
if(NOT GTEST_SOURCE_DIR)
    message(FATAL_ERROR "Couldn't find gtest-all.cc under folder GHEX_TEST_DIR = " ${GHEX_GTEST_DIR})
endif()

set(GTEST_LIB_SOURCES ${GTEST_SOURCE_DIR}/gtest-all.cc)
set(GTEST_INCLUDE_DIRS ${GTEST_INC_DIR} ${GTEST_SOURCE_DIR} ${GTEST_SOURCE_DIR}/..)

add_library(gtest_main_mt ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES} ./utils/gtest_main.cpp )
target_include_directories(gtest_main_mt PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_link_libraries(gtest_main_mt MPI::MPI_CXX GridTools::gridtools )

add_library(gtest_main_bench ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES} ./utils/gtest_main_bench.cpp )
target_link_libraries(gtest_main_bench MPI::MPI_CXX GridTools::gridtools )
target_include_directories(gtest_main_bench PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})

add_library(gtest_main_bench_mt ${GTEST_LIB_SOURCES} ${GTEST_MAIN_SOURCES} ./utils/gtest_main_bench.cpp )
target_link_libraries(gtest_main_bench_mt MPI::MPI_CXX GridTools::gridtools )
target_include_directories(gtest_main_bench_mt PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
target_compile_definitions(gtest_main_bench_mt PRIVATE GHEX_BENCHMARKS_USE_MULTI_THREADED_MPI)

add_subdirectory(tests)
add_subdirectory(benchmarks)
