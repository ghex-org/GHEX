cmake_minimum_required(VERSION 3.12.4)

project(GHEX VERSION 0.1 LANGUAGES CXX)
if(USE_GPU)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

cmake_policy(SET CMP0048 NEW)

set(USE_GPU "OFF" CACHE BOOL "use cuda")
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

# GridTools uses BUILD_TESTING, cache it and reset after (not required with GT 2.0)
set(build_testing ${BUILD_TESTING})
set(install_tools ${INSTALL_TOOLS})
set(BUILD_TESTING OFF)
set(INSTALL_TOOLS OFF)
set(GT_INSTALL_EXAMPLES OFF)
include(FetchContent)
FetchContent_Declare(GridTools
    URL https://github.com/GridTools/gridtools/archive/v1.1.3.tar.gz
)
FetchContent_GetProperties(GridTools)
if(NOT GridTools_POPULATED)
    FetchContent_Populate(GridTools)
    add_subdirectory(${gridtools_SOURCE_DIR} ${gridtools_BINARY_DIR})
endif()
set(BUILD_TESTING ${build_testing})
set(INSTALL_TOOLS ${install_tools})

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

set(GHEX_ENABLE_ATLAS_BINDINGS OFF CACHE BOOL "Set to true to build with Atlas bindings")
if (GHEX_ENABLE_ATLAS_BINDINGS)
    find_package(eckit REQUIRED HINTS ${eckit_DIR})
    find_package(Atlas REQUIRED HINTS ${Atlas_DIR})
    # Temporary workaround to fix missing dependency in Atlas target: eckit
    target_link_libraries(atlas INTERFACE eckit)
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
if (GHEX_ENABLE_ATLAS_BINDINGS)
    target_link_libraries(ghexlib INTERFACE atlas)
endif()
target_compile_features(ghexlib INTERFACE cxx_std_14)

# Enable adding of tests etc
enable_testing()

# Enable CDash dashboard testing integration
include(CTest)

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

    if (GHEX_ENABLE_ATLAS_BINDINGS)
        add_library(gtest_main_atlas ./utils/gtest_main_atlas.cpp)
        target_link_libraries(gtest_main_atlas GTest::gtest ghexlib)

        add_library(gtest_main_bench_atlas ./utils/gtest_main_bench_atlas.cpp)
        target_link_libraries(gtest_main_bench_atlas GTest::gtest ghexlib)
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
