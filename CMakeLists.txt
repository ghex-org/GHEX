cmake_minimum_required(VERSION 3.12.4)

project(GHEX VERSION 0.1 LANGUAGES CXX)
if(USE_GPU)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_EXTENSIONS OFF)    
endif()

cmake_policy(SET CMP0048 NEW)

set(USE_GPU "OFF" CACHE BOOL "use cuda")
set(USE_MPI_WITH_UCX_IN_TESTS "OFF" CACHE BOOL "use ucx for unit test")
set(USE_HYBRID_TESTS "ON" CACHE BOOL "run gpu+cpu tests")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic") # Last option is because of a problem with GCC9. Try to remove with more releases of the compiler

set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(GHEX_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${GHEX_MODULE_PATH}")

find_package(MPI REQUIRED)
find_package(Boost REQUIRED)
find_package(GridTools REQUIRED HINTS ${GridTools_DIR})
list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

set(GHEX_USE_UCP OFF CACHE BOOL "Set to true to use UCP library")
if (GHEX_USE_UCP)
    find_package(UCP REQUIRED)
endif()

set(GHEX_USE_PMIX OFF CACHE BOOL "Set to true to use PMIx process management")
if (GHEX_USE_PMIX)
    find_package(PMIx REQUIRED)
endif()

set(GHEX_SKIP_MPICXX OFF CACHE BOOL "True if your compiler wrapper includes MPI already (as CRAY PE for instance)")
if (GHEX_SKIP_MPICXX)
    set(MPI_CXX_SKIP_MPICXX ON)
else()
    set(MPI_CXX_SKIP_MPICXX OFF)
endif()

set(GHEX_BUILD_TESTS OFF CACHE BOOL "True if tests shall be built")
set(GHEX_BUILD_BENCHMARKS OFF CACHE BOOL "True if benchmarks shall be built")

add_library(ghexlib INTERFACE)
add_library(GHEX::ghexlib ALIAS ghexlib)
target_include_directories(ghexlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
target_link_libraries(ghexlib INTERFACE GridTools::gridtools MPI::MPI_CXX)
if (GHEX_USE_UCP)
    target_link_libraries(ghexlib INTERFACE UCP::libucp)
    if (GHEX_USE_PMIX)
        target_link_libraries(ghexlib INTERFACE PMIx::libpmix)
    endif()
endif()
target_compile_features(ghexlib INTERFACE cxx_std_14)

enable_testing()

if (GHEX_BUILD_TESTS OR GHEX_BUILD_BENCHMARKS)
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

    if (GHEX_BUILD_TESTS)
        add_library(gtest_main_mt ./utils/gtest_main.cpp)
        target_link_libraries(gtest_main_mt GTest::gtest ghexlib)
    
        if (GHEX_USE_UCP)
            add_library(gtest_main_ucx ./utils/gtest_main_ucx.cpp)
            target_link_libraries(gtest_main_ucx GTest::gtest ghexlib)
        endif()

        add_subdirectory(tests)
    endif()
  
    if (GHEX_BUILD_BENCHMARKS)
        add_library(gtest_main_bench ./utils/gtest_main_bench.cpp)
        target_link_libraries(gtest_main_bench GTest::gtest ghexlib)
 
        add_library(gtest_main_bench_mt ./utils/gtest_main_bench.cpp)
        target_compile_definitions(gtest_main_bench_mt PRIVATE GHEX_BENCHMARKS_USE_MULTI_THREADED_MPI)
        target_link_libraries(gtest_main_bench_mt GTest::gtest ghexlib)

        add_subdirectory(benchmarks)
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS ghexlib EXPORT GHEX-targets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT GHEX-targets
  FILE GHEX-targets.cmake
  NAMESPACE GHEX::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
  )
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/GHEXConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/GHEXConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
write_basic_package_version_file(GHEXConfigVersion.cmake VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/GHEXConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/GHEXConfigVersion.cmake
        ${CMAKE_CURRENT_LIST_DIR}/cmake/FindUCP.cmake
        ${CMAKE_CURRENT_LIST_DIR}/cmake/FindPMIx.cmake
    DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake)
