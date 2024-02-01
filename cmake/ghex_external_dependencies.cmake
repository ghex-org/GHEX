include(CMakeDependentOption)
include(ghex_git_submodule)
include(ghex_external_project)

if(GHEX_GIT_SUBMODULE)
    update_git_submodules()
endif()

# ---------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------
find_package(MPI REQUIRED COMPONENTS CXX)

# ---------------------------------------------------------------------
# Boost setup
# ---------------------------------------------------------------------
find_package(Boost REQUIRED)

# ---------------------------------------------------------------------
# LibRt setup
# ---------------------------------------------------------------------
if(UNIX AND NOT APPLE)
find_library(LIBRT rt REQUIRED)
endif()

# ---------------------------------------------------------------------
# GridTools setup
# ---------------------------------------------------------------------
cmake_dependent_option(GHEX_USE_BUNDLED_GRIDTOOLS "Use bundled gridtools." ON
    "GHEX_USE_BUNDLED_LIBS" OFF)
if(GHEX_USE_BUNDLED_GRIDTOOLS)
    check_git_submodule(GridTools ext/gridtools)
    set(BUILD_GMOCK OFF)
    set(GT_BUILD_TESTING OFF)
    set(GT_INSTALL_EXAMPLES OFF)
    set(GT_TESTS_ENABLE_PYTHON_TESTS OFF)
    set(GT_ENABLE_BINDINGS_GENERATION OFF)
    add_subdirectory(ext/gridtools EXCLUDE_FROM_ALL)
else()
    find_package(GridTools REQUIRED)
endif()

# ---------------------------------------------------------------------
# oomph setup
# ---------------------------------------------------------------------
set(GHEX_TRANSPORT_BACKEND "MPI" CACHE STRING "Choose the backend type: MPI | UCX | LIBFABRIC")
set_property(CACHE GHEX_TRANSPORT_BACKEND PROPERTY STRINGS "MPI" "UCX" "LIBFABRIC")
cmake_dependent_option(GHEX_USE_BUNDLED_OOMPH "Use bundled oomph." ON "GHEX_USE_BUNDLED_LIBS" OFF)
if(GHEX_USE_BUNDLED_OOMPH)
    set(OOMPH_GIT_SUBMODULE OFF CACHE BOOL "")
    set(OOMPH_USE_BUNDLED_LIBS ON CACHE BOOL "")
    if(GHEX_TRANSPORT_BACKEND STREQUAL "LIBFABRIC")
        set(OOMPH_WITH_LIBFABRIC ON CACHE BOOL "Build with LIBFABRIC backend")
    elseif(GHEX_TRANSPORT_BACKEND STREQUAL "UCX")
        set(OOMPH_WITH_UCX ON CACHE BOOL "Build with UCX backend")
    endif()
    if(GHEX_USE_GPU)
        set(HWMALLOC_ENABLE_DEVICE ON CACHE BOOL "True if GPU support shall be enabled")
        if (GHEX_GPU_TYPE STREQUAL "NVIDIA")
            set(HWMALLOC_DEVICE_RUNTIME "cuda" CACHE STRING "Choose the type of the gpu runtime.")
        elseif (GHEX_GPU_TYPE STREQUAL "AMD")
            set(HWMALLOC_DEVICE_RUNTIME "hip" CACHE STRING "Choose the type of the gpu runtime.")
        endif()
    endif()
    check_git_submodule(oomph ext/oomph)
    add_subdirectory(ext/oomph)
    if(TARGET oomph_mpi)
        add_library(oomph::oomph_mpi ALIAS oomph_mpi)
    endif()
    if(TARGET oomph_ucx)
        add_library(oomph::oomph_ucx ALIAS oomph_ucx)
    endif()
    if(TARGET oomph_libfabric)
        add_library(oomph::oomph_libfabric ALIAS oomph_libfabric)
    endif()
else()
    find_package(oomph REQUIRED)
endif()

function(ghex_link_to_oomph target)
    if (GHEX_TRANSPORT_BACKEND STREQUAL "LIBFABRIC")
        target_link_libraries(${target} PRIVATE oomph::oomph_libfabric)
    elseif (GHEX_TRANSPORT_BACKEND STREQUAL "UCX")
        target_link_libraries(${target} PRIVATE oomph::oomph_ucx)
    else()
        target_link_libraries(${target} PRIVATE oomph::oomph_mpi)
    endif()
endfunction()

# ---------------------------------------------------------------------
# xpmem setup
# ---------------------------------------------------------------------
if (GHEX_USE_XPMEM)
    find_package(XPMEM REQUIRED)
endif()

# ---------------------------------------------------------------------
# parmetis setup
# ---------------------------------------------------------------------
if (GHEX_ENABLE_PARMETIS_BINDINGS)
    set(METIS_INCLUDE_DIR "" CACHE STRING "METIS include directory")
    set(METIS_LIB_DIR "" CACHE STRING "METIS library directory")
    set(PARMETIS_INCLUDE_DIR "" CACHE STRING "ParMETIS include directory")
    set(PARMETIS_LIB_DIR "" CACHE STRING "ParMETIS library directory")
endif()

# ---------------------------------------------------------------------
# atlas setup
# ---------------------------------------------------------------------
if (GHEX_ENABLE_ATLAS_BINDINGS)
    find_package(eckit REQUIRED HINTS ${eckit_DIR})
    find_package(Atlas REQUIRED HINTS ${Atlas_DIR})
    set(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND "KFIRST" CACHE STRING "GridTools CPU storage traits: KFIRST | IFIRST.")
    set_property(CACHE GHEX_ATLAS_GT_STORAGE_CPU_BACKEND PROPERTY STRINGS "KFIRST" "IFIRST")
    # Temporary workaround to fix missing dependency in Atlas target: eckit
    target_link_libraries(atlas INTERFACE eckit)
    target_link_libraries(ghex_common INTERFACE atlas)

    if (GHEX_ATLAS_GT_STORAGE_CPU_BACKEND STREQUAL "KFIRST")
        set(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_KFIRST ON)
        set(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_IFIRST OFF)
    elseif(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND STREQUAL "IFIRST")
        set(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_KFIRST OFF)
        set(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_IFIRST ON)
    else()
        set(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_KFIRST OFF)
        set(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_IFIRST OFF)
    endif()
endif()

# ---------------------------------------------------------------------
# google test setup
# ---------------------------------------------------------------------
cmake_dependent_option(GHEX_USE_BUNDLED_GTEST "Use bundled googletest lib." ON
    "GHEX_USE_BUNDLED_LIBS" OFF)
if (GHEX_WITH_TESTING)
    if(GHEX_USE_BUNDLED_GTEST)
        add_external_cmake_project(
            NAME googletest-ghex
            PATH ext/googletest
            INTERFACE_NAME ext-gtest-ghex
            LIBS libgtest.a libgtest_main.a
            CMAKE_ARGS
                "-DCMAKE_BUILD_TYPE=release"
                "-DBUILD_SHARED_LIBS=OFF"
                "-DBUILD_GMOCK=OFF")
        # on some systems we need link explicitly against threads
        if (TARGET ext-gtest-ghex)
            find_package (Threads)
            target_link_libraries(ext-gtest-ghex INTERFACE Threads::Threads)
        endif()
    else()
        # Use system provided google test
        find_package(GTest REQUIRED)
        add_library(ext-gtest-ghex INTERFACE)
        if (${CMAKE_VERSION} VERSION_LESS "3.20.0")
            target_link_libraries(ext-gtest-ghex INTERFACE GTest::GTest GTest::Main)
        else()
            target_link_libraries(ext-gtest-ghex INTERFACE GTest::gtest GTest::gtest_main)
        endif()
    endif()
endif()
