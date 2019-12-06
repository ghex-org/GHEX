cmake_minimum_required(VERSION 3.12.4)

project(GHEX VERSION 0.1 LANGUAGES CXX)
if(USE_GPU)
  enable_language(CUDA)
endif()

cmake_policy(SET CMP0048 NEW)
enable_testing()

set(USE_GPU "OFF" CACHE BOOL "use cuda")
set(USE_MPI_WITH_UCX_IN_TESTS "OFF" CACHE BOOL "use ucx for unit test")
set(USE_HYBRID_TESTS "ON" CACHE BOOL "run gpu+cpu tests")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic") # Last option is because of a problem with GCC9. Try to remove with more releases of the compiler

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(GHEX_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${GHEX_MODULE_PATH}")

find_package(MPI REQUIRED)
find_package(Boost 1.66 REQUIRED)
find_package(GridTools REQUIRED HINTS ${GridTools_DIR})
# NOT USED:
# find_package(PMIx)
# find_package(UCP)

list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

add_library(ghexlib INTERFACE)
add_library(GHEX::ghexlib ALIAS ghexlib)
target_include_directories(ghexlib INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )
target_link_libraries(ghexlib INTERFACE GridTools::gridtools)
target_compile_features(ghexlib INTERFACE cxx_std_14)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)

  # https://github.com/google/googletest/issues/2429
  add_library(GTest::gtest ALIAS gtest)
endif()

add_library(gtest_main_mt utils/gtest_main.cpp)
target_link_libraries(gtest_main_mt MPI::MPI_CXX GridTools::gridtools GTest::gtest)

add_library(gtest_main_bench utils/gtest_main_bench.cpp )
target_link_libraries(gtest_main_bench MPI::MPI_CXX GridTools::gridtools GTest::gtest)

add_library(gtest_main_bench_mt ./utils/gtest_main_bench.cpp)
target_link_libraries(gtest_main_bench_mt MPI::MPI_CXX GridTools::gridtools GTest::gtest)
target_compile_definitions(gtest_main_bench_mt PRIVATE GHEX_BENCHMARKS_USE_MULTI_THREADED_MPI)

add_subdirectory(tests)
add_subdirectory(benchmarks)

include(GNUInstallDirs)
install(TARGETS ghexlib EXPORT GHEX-targets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT GHEX-targets
  FILE GHEX-targets.cmake
  NAMESPACE GHEX::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
  )
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/GHEXConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GHEXConfig.cmake
  INSTALL_DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake)
write_basic_package_version_file(GHEXConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
  )
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/GHEXConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GHEXConfigVersion.cmake
  DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake)
