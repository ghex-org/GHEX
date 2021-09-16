include(ghex_compile_options)

# ---------------------------------------------------------------------
# interface library
# ---------------------------------------------------------------------
add_library(ghex_common INTERFACE)
add_library(GHEX::ghex ALIAS ghex_common)

# ---------------------------------------------------------------------
# shared library
# ---------------------------------------------------------------------
add_library(ghex SHARED)
add_library(GHEX::lib ALIAS ghex)
target_link_libraries(ghex PUBLIC ghex_common)
ghex_target_compile_options(ghex)

# ---------------------------------------------------------------------
# device setup
# ---------------------------------------------------------------------
include(ghex_device)
if (ghex_gpu_mode STREQUAL "hip")
    target_link_libraries(ghex PUBLIC hip::device)
endif()

# ---------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------
find_package(MPI REQUIRED)
target_link_libraries(ghex_common INTERFACE MPI::MPI_CXX)
target_link_libraries(ghex PRIVATE MPI::MPI_CXX)

# ---------------------------------------------------------------------
# Boost setup
# ---------------------------------------------------------------------
find_package(Boost REQUIRED)
target_link_libraries(ghex_common INTERFACE Boost::boost)

# ---------------------------------------------------------------------
# LibRt setup
# ---------------------------------------------------------------------
find_library(LIBRT rt REQUIRED)
target_link_libraries(ghex PUBLIC ${LIBRT})

# ---------------------------------------------------------------------
# GridTools setup
# ---------------------------------------------------------------------
include(ghex_gridtools)
target_link_libraries(ghex_common INTERFACE GridTools::gridtools)

# ---------------------------------------------------------------------
# oomph setup
# ---------------------------------------------------------------------
include(ghex_oomph)
target_link_libraries(ghex_common INTERFACE oomph::oomph)

# ---------------------------------------------------------------------
# xpmem setup
# ---------------------------------------------------------------------
set(GHEX_USE_XPMEM OFF CACHE BOOL "Set to true to use xpmem shared memory")
if (GHEX_USE_XPMEM)
    find_package(XPMEM REQUIRED)
    target_link_libraries(ghex_common INTERFACE XPMEM::libxpmem)
    target_link_libraries(ghex PRIVATE XPMEM::libxpmem)
endif()
set(GHEX_USE_XPMEM_ACCESS_GUARD OFF CACHE BOOL "Use xpmem to synchronize rma access")
mark_as_advanced(GHEX_USE_XPMEM_ACCESS_GUARD)

# ---------------------------------------------------------------------
# parmetis setup
# ---------------------------------------------------------------------
set(GHEX_ENABLE_PARMETIS_BINDINGS OFF CACHE BOOL "Set to true to build with ParMETIS bindings")
if (GHEX_ENABLE_PARMETIS_BINDINGS)
    set(METIS_INCLUDE_DIR "" CACHE STRING "METIS include directory")
    set(METIS_LIB_DIR "" CACHE STRING "METIS library directory")
    set(PARMETIS_INCLUDE_DIR "" CACHE STRING "ParMETIS include directory")
    set(PARMETIS_LIB_DIR "" CACHE STRING "ParMETIS library directory")
endif()

# ---------------------------------------------------------------------
# atlas setup
# ---------------------------------------------------------------------
set(GHEX_ENABLE_ATLAS_BINDINGS OFF CACHE BOOL "Set to true to build with Atlas bindings")
if (GHEX_ENABLE_ATLAS_BINDINGS)
    find_package(eckit REQUIRED HINTS ${eckit_DIR})
    find_package(Atlas REQUIRED HINTS ${Atlas_DIR})
    # Temporary workaround to fix missing dependency in Atlas target: eckit
    target_link_libraries(atlas INTERFACE eckit)
    target_link_libraries(ghex_common INTERFACE atlas)
endif()

# ---------------------------------------------------------------------
# install rules
# ---------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ghex_common ghex
    EXPORT ghex-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
