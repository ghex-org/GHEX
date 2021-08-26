
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
