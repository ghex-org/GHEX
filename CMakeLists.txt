cmake_minimum_required(VERSION 3.17)
#cmake_policy(SET CMP0048 NEW)

project(GHEX VERSION 0.2 LANGUAGES CXX)

# ---------------------------------------------------------------------
# CMake setup, C++ version, build type, modules, etc
# ---------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE) # AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
set(GHEX_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${GHEX_MODULE_PATH}")


# ---------------------------------------------------------------------
# Define main library
# ---------------------------------------------------------------------
include(ghex_common)

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

# ---------------------------------------------------------------------
# fortran bindings
# ---------------------------------------------------------------------
include(ghex_fortran)

# ---------------------------------------------------------------------
# python bindings
# ---------------------------------------------------------------------

# bindings
add_subdirectory(bindings)

# ---------------------------------------------------------------------
# testing
# ---------------------------------------------------------------------
set(GHEX_WITH_TESTING OFF CACHE BOOL "True if tests shall be built")
if (GHEX_WITH_TESTING)
    enable_testing()
    # Enable CDash dashboard testing integration
    include(CTest)
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

export(EXPORT ghex-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/ghex-targets.cmake")
