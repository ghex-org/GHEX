
set(c_cxx_lang "$<COMPILE_LANGUAGE:C,CXX>")
set(cuda_lang "$<COMPILE_LANGUAGE:CUDA>")

# ---------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------
find_package(MPI REQUIRED)
target_link_libraries(ghex INTERFACE MPI::MPI_CXX)

# ---------------------------------------------------------------------
# Boost setup
# ---------------------------------------------------------------------
find_package(Boost REQUIRED)
target_link_libraries(ghex INTERFACE Boost::boost)

# ---------------------------------------------------------------------
# LibRt setup
# ---------------------------------------------------------------------
find_library(LIBRT rt REQUIRED)
target_link_libraries(ghex INTERFACE ${LIBRT})

# ---------------------------------------------------------------------
# GridTools setup
# ---------------------------------------------------------------------
include(ghex_gridtools)
target_link_libraries(ghex INTERFACE GridTools::gridtools)

# ---------------------------------------------------------------------
# oomph setup
# ---------------------------------------------------------------------
include(ghex_oomph)
target_link_libraries(ghex INTERFACE oomph::oomph)

# ---------------------------------------------------------------------
# xpmem setup
# ---------------------------------------------------------------------
set(GHEX_USE_XPMEM OFF CACHE BOOL "Set to true to use xpmem shared memory")
if (GHEX_USE_XPMEM)
    find_package(XPMEM REQUIRED)
endif()
set(GHEX_USE_XPMEM_ACCESS_GUARD OFF CACHE BOOL "Use xpmem to synchronize rma access")
mark_as_advanced(GHEX_USE_XPMEM_ACCESS_GUARD)
if (GHEX_USE_XPMEM_ACCESS_GUARD)
    target_compile_definitions(ghex INTERFACE GHEX_USE_XPMEM_ACCESS_GUARD)
endif()
