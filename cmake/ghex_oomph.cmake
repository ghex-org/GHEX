#TODO: make some sort of plugin loader to avoid linking to hwmalloc, oomph_<backend> and ghex
#      and instead only link to libghex
set(GHEX_TRANSPORT_BACKEND "MPI" CACHE STRING "Choose the backend type: MPI | UCX | LIBFABRIC")
set_property(CACHE GHEX_TRANSPORT_BACKEND PROPERTY STRINGS "MPI" "UCX" "LIBFABRIC")

set(_oomph_repository "https://github.com/boeschf/oomph.git")
set(_oomph_tag        "main")
if(NOT _oomph_already_fetched)
    message(STATUS "Fetching oomph ${_oomph_tag} from ${_oomph_repository}")
endif()
include(FetchContent)
FetchContent_Declare(
    oomph
    GIT_REPOSITORY ${_oomph_repository}
    GIT_TAG        ${_oomph_tag}
)

if (GHEX_TRANSPORT_BACKEND STREQUAL "LIBFABRIC")
    set(OOMPH_WITH_MPI OFF CACHE INTERNAL "")  # Forces the value
    set(OOMPH_WITH_UCX OFF CACHE INTERNAL "")  # Forces the value
    set(OOMPH_WITH_LIBFABRIC ON CACHE INTERNAL "")  # Forces the value
    target_link_libraries(ghex PUBLIC oomph::libfabric)
elseif (GHEX_TRANSPORT_BACKEND STREQUAL "UCX")
    set(OOMPH_WITH_MPI OFF CACHE INTERNAL "")  # Forces the value
    set(OOMPH_WITH_UCX ON CACHE INTERNAL "")  # Forces the value
    set(OOMPH_WITH_LIBFABRIC OFF CACHE INTERNAL "")  # Forces the value
    target_link_libraries(ghex PUBLIC oomph::ucx)
else()
    set(OOMPH_WITH_MPI ON CACHE INTERNAL "")  # Forces the value
    set(OOMPH_WITH_UCX OFF CACHE INTERNAL "")  # Forces the value
    set(OOMPH_WITH_LIBFABRIC OFF CACHE INTERNAL "")  # Forces the value
    target_link_libraries(ghex PUBLIC oomph::mpi)
endif()

if (GHEX_USE_GPU)
    set(OOMPH_USE_GPU ON CACHE INTERNAL "")  # Forces the value
    if (ghex_gpu_mode STREQUAL "hip")
        set(OOMPH_GPU_TYPE "AMD" CACHE INTERNAL "")  # Forces the value
    elseif (ghex_gpu_mode STREQUAL "cuda")
        set(OOMPH_GPU_TYPE "NVIDIA" CACHE INTERNAL "")  # Forces the value
    else()
        set(OOMPH_GPU_TYPE "EMULATE" CACHE INTERNAL "")  # Forces the value
    endif()
else()
    set(OOMPH_USE_GPU OFF CACHE INTERNAL "")  # Forces the value
    set(OOMPH_GPU_TYPE "AUTO" CACHE INTERNAL "")  # Forces the value
endif()

FetchContent_MakeAvailable(oomph)
set(_oomph_already_fetched ON CACHE INTERNAL "")
