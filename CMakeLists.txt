cmake_minimum_required(VERSION 3.21)

if (NOT ${CMAKE_VERSION} VERSION_LESS 3.27)
    # new in 3.27: additionally use uppercase <PACKAGENAME>_ROOT
    # environment and CMake variables for find_package
    cmake_policy(SET CMP0144 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
endif()

set(GHEX_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${GHEX_MODULE_PATH}")

include(ghex_version)
make_version(GHEX 0 3 0)

project(GHEX VERSION ${GHEX_VERSION} LANGUAGES C CXX)

# ---------------------------------------------------------------------
# CMake setup, C++ version, build type, modules, etc
# ---------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE) # AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Position Independent Code (PIC) option
set(POSITION_INDEPENDENT_CODE ON)

# Set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set the library and archive output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---------------------------------------------------------------------
# GHEX options
# ---------------------------------------------------------------------
option(GHEX_GIT_SUBMODULE "Check submodules during build" OFF)
option(GHEX_USE_BUNDLED_LIBS "Use bundled 3rd party libraries" OFF)
set(GHEX_NO_RMA OFF CACHE BOOL "Disable in-node RMA completely")
set(GHEX_USE_XPMEM OFF CACHE BOOL "Set to true to use xpmem shared memory")
set(GHEX_USE_XPMEM_ACCESS_GUARD OFF CACHE BOOL "Use xpmem to synchronize rma access")
mark_as_advanced(GHEX_USE_XPMEM_ACCESS_GUARD)
set(GHEX_ENABLE_PARMETIS_BINDINGS OFF CACHE BOOL "Set to true to build with ParMETIS bindings")
set(GHEX_ENABLE_ATLAS_BINDINGS OFF CACHE BOOL "Set to true to build with Atlas bindings")
set(GHEX_BUILD_FORTRAN OFF CACHE BOOL "True if FORTRAN bindings shall be built")
set(GHEX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Set to true to build Python bindings")
set(GHEX_WITH_TESTING OFF CACHE BOOL "True if tests shall be built")

# ---------------------------------------------------------------------
# Common includes
# ---------------------------------------------------------------------
include(ghex_device)
include(ghex_compile_options)
include(ghex_copy_files)
include(ghex_reg_test)

# ---------------------------------------------------------------------
# Define main library
# ---------------------------------------------------------------------
include(ghex_common)

# ---------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------
include(ghex_external_dependencies)
target_link_libraries(ghex_common INTERFACE MPI::MPI_CXX)
target_link_libraries(ghex_common INTERFACE Boost::boost)
target_link_libraries(ghex_common INTERFACE GridTools::gridtools)
target_link_libraries(ghex_common INTERFACE oomph::oomph)
target_link_libraries(ghex PRIVATE MPI::MPI_CXX)
if (UNIX AND NOT APPLE)
target_link_libraries(ghex PUBLIC ${LIBRT})
endif()
target_link_libraries(ghex INTERFACE oomph::oomph)
if (GHEX_USE_XPMEM)
    target_link_libraries(ghex_common INTERFACE XPMEM::libxpmem)
    target_link_libraries(ghex PRIVATE XPMEM::libxpmem)
endif()

# ---------------------------------------------------------------------
# include paths
# ---------------------------------------------------------------------
target_include_directories(ghex_common INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)

# ---------------------------------------------------------------------
# main src subdir
# ---------------------------------------------------------------------
add_subdirectory(src)

# ---------------------------------------------------------------------
# generate config file
# ---------------------------------------------------------------------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/ghex/config.hpp @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/include/ghex/config.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ghex)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ghex_version.txt.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.txt @ONLY)

# ---------------------------------------------------------------------
# fortran bindings
# ---------------------------------------------------------------------
include(ghex_fortran)

# ---------------------------------------------------------------------
# python bindings
# ---------------------------------------------------------------------
include(ghex_python)

# bindings
add_subdirectory(bindings)

# ---------------------------------------------------------------------
# testing
# ---------------------------------------------------------------------
if (GHEX_WITH_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

# ---------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------
install(EXPORT ghex-targets
    FILE ghex-targets.cmake
    NAMESPACE GHEX::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ghex_config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/GHEXConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)

write_basic_package_version_file(GHEXConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/GHEXConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/GHEXConfigVersion.cmake
        ${CMAKE_CURRENT_LIST_DIR}/cmake/FindXPMEM.cmake
    DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake)

install(
    FILES "${PROJECT_SOURCE_DIR}/LICENSE"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/ghex
    COMPONENT license
)

export(EXPORT ghex-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/ghex-targets.cmake")
